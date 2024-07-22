import cv2

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
CUBE_ID = 0

if __name__ == '__main__':
    size = 200
    for name, id in zip(['cube'], [CUBE_ID]):
        marker_im = cv2.aruco.generateImageMarker(ARUCO_DICT, id, size)
        cv2.imwrite(f"../assets/aruco_4x4_100_id_{id}_{name}.png", marker_im)
