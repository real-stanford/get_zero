import pyrealsense2 as rs
import numpy as np
import cv2
from get_zero.deploy.leap.util.ar_tag import CUBE_ID, ARUCO_DICT
import time
from argparse import ArgumentParser

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
	From https://stackoverflow.com/questions/76802576/how-to-estimate-pose-of-single-marker-in-opencv-python-4-8-0

    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

class Camera:
	def __init__(self):
		raise NotImplementedError
	
	def get_rgb(self):
		raise NotImplementedError

	def get_intrinsic_mat(self):
		raise NotImplementedError

	def get_ar_tag_pose(self, id, aruco_dict, marker_length=0.1, show_viewer=False, block=False): # 0.0762m is 3 in
		color_img = None
		gray_img = None
		def fetch_im():
			nonlocal color_img, gray_img
			color_img = cv2.cvtColor(self.get_rgb(), cv2.COLOR_RGB2BGR) # cv2 uses BGR color ordering
			gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

		detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
		distortion = np.zeros((1,5))

		fetch_im()

		while True:
			if show_viewer:
				im_vis = np.copy(color_img)
				
			corners, ids, rejected = detector.detectMarkers(gray_img)
			found = ids is not None and id in ids
			if found:
				index = list(ids).index(id)
				corner_index = corners[index][0]
				center = np.mean(corner_index, axis=0)
				rvec, tvec, markerPoints = my_estimatePoseSingleMarkers([corner_index], marker_length, self.get_intrinsic_mat(), distortion)

			if show_viewer and ids is not None:
				cv2.aruco.drawDetectedMarkers(im_vis, corners, ids)
				for i in range(len(ids)):
					rvec_vis, tvec_vis, markerPoints = my_estimatePoseSingleMarkers(corners[i], marker_length, self.get_intrinsic_mat(), distortion)
					cv2.drawFrameAxes(im_vis, self.get_intrinsic_mat(), distortion, rvec_vis[0], tvec_vis[0], marker_length, 2)
			
			if show_viewer:
				cv2.imshow('AR tag detection', im_vis)
				cv2.waitKey(1)

			if found:
				return True, rvec[0][:, 0], tvec[0][:, 0], corner_index[1], center
			elif not block:
				break
			else:
				fetch_im()
		
		return False, None, None, None, None

	def record_video(self, duration_secs: float, fpath: str):
		frame_size = self.get_rgb().shape[:2]
		frame_size = (frame_size[1], frame_size[0])
		video_writer = cv2.VideoWriter(fpath, cv2.VideoWriter_fourcc(*'mp4v'), self.get_video_fps(), frame_size, 1)

		try:
			for _ in range(int(duration_secs * self.get_video_fps())):
				color_image = self.get_rgb()
				color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
				video_writer.write(color_image)
		finally:
			video_writer.release()

	def get_video_fps(self) -> int:
		raise NotImplementedError
	
	def camera_viewer(self):
		while True:
			image = self.get_rgb()
			bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			cv2.imshow('camera viewer', bgr_image)
			cv2.waitKey(1)

class RealsenseCamera(Camera):
	def __init__(self, id=0):
		self.record_fps = 30

		# get available cameras
		realsense_ctx = rs.context()
		serial_num = realsense_ctx.devices[min(id, len(realsense_ctx.devices) - 1)].get_info(rs.camera_info.serial_number)

		config = rs.config()
		config.enable_device(serial_num)
		config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, self.record_fps)
		self.pipe = rs.pipeline()
		self.profile = self.pipe.start(config)
		K = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics()
		self.K = np.array([[K.fx, 0,    K.ppx],
					       [0,    K.fy, K.ppy],
						   [0,    0,    1]])

	def get_rgb(self):
		while True:
			frames = self.pipe.wait_for_frames()
			color_frame = frames.get_color_frame()
			if color_frame is None:
				continue
			im = np.asanyarray(color_frame.get_data())
			return im
	
	def get_intrinsic_mat(self):
		return self.K
	
	def get_video_fps(self) -> int:
		return self.record_fps
	
class MeasureRotation:
	def __init__(self, cam: Camera, ar_tag_id, aruco_dict, use_viewer=False, reset_threshold=10):
		self.cam = cam
		self.last_theta = None
		self.completed_theta = 0
		self.start_theta = None
		self.final_theta = 0
		self.use_viewer = use_viewer
		self.ar_tag_id = ar_tag_id
		self.aruco_dict = aruco_dict
		self.steps_since_last_detection = 0
		self.reset_threshold = reset_threshold
		self.num_resets = 0
		self.valid_steps = 0
		self.in_reset_mode = True

	def reset_env(self):
		if self.last_theta is not None and self.start_theta is not None:
			self.completed_theta += self.last_theta - self.start_theta # commit what was able to be rotation pre reset
		self.start_theta = None
		self.last_theta = None
		self.steps_since_last_detection = 0

	def step(self, block=False):
		found, _, _, corner_pos, center_pos = self.cam.get_ar_tag_pose(self.ar_tag_id, self.aruco_dict, show_viewer=self.use_viewer, block=block)
		
		reset_detected = self.steps_since_last_detection >= self.reset_threshold and not found

		# fall/reset detection
		if reset_detected:
			# this means the cube fell off, so we need to update start_theta
			if not self.in_reset_mode:
				# this is the case for the start of the reset
				self.num_resets += 1
				self.in_reset_mode = True
				self.completed_theta += self.last_theta - self.start_theta # commit what was able to be rotation pre fall
				self.start_theta = None
				self.last_theta = None
		else:
			self.valid_steps += 1

		if found:
			# compute current theta
			diff = corner_pos - center_pos
			x, y = diff
			theta = np.arctan2(x, y).item()

			if theta < 0:
				theta = 2 * np.pi + theta

			if self.start_theta is None:
				self.start_theta = theta
				self.last_theta = theta

			self.in_reset_mode = False

			# if theta passes threshold then update completed_theta
			if theta < 0.5 * np.pi and self.last_theta > 1.5 * np.pi:
				self.completed_theta += 2 * np.pi
			elif theta > 1.5 * np.pi and self.last_theta < 0.5 * np.pi:
				self.completed_theta -= 2 * np.pi

			self.final_theta = self.completed_theta + theta - self.start_theta
			self.last_theta = theta
			self.steps_since_last_detection = 0
		else:
			self.steps_since_last_detection += 1
			reset_detected = False
		
		return self.final_theta, self.valid_steps, self.steps_since_last_detection, self.num_resets, self.in_reset_mode
	
	def get_final_metrics(self) -> float:
		return self.final_theta, self.valid_steps, self.num_resets

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--mode', type=str, required=True)
	parser.add_argument('--cam_id', type=int, default=0)
	args = parser.parse_args()

	cam = RealsenseCamera(args.cam_id)

	if args.mode == 'AR':
		np.set_printoptions(precision=2, floatmode='fixed')

		measurer = MeasureRotation(cam, CUBE_ID, ARUCO_DICT, use_viewer=True)
		while True:
			print(measurer.step())
			time.sleep(0.1)
	elif args.mode == 'record':
		cam.record_video(5, 'test.mp4')
	elif args.mode == 'viewer':
		cam.camera_viewer()
	else:
		raise NotImplementedError
