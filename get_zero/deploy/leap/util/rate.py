import time

class Rate:
    """Drop-in replacement for rospy rate, but without ROS dependency. Has code directly copied from rospy.Rate, but modified to use `time` package rather than ROS time"""
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_dur = 1/hz

    def _remaining(self, curr_time):
        # detect time jumping backwards
        if self.last_time > curr_time:
            self.last_time = curr_time

        # calculate remaining time
        elapsed = curr_time - self.last_time
        return self.sleep_dur - elapsed

    def remaining(self):
        curr_time = time.time()
        return self._remaining(curr_time)

    def sleep(self):
        curr_time = time.time()
        try:
            time.sleep(self._remaining(curr_time))
        except ValueError:
            self.last_time = time.time()
        self.last_time = self.last_time + self.sleep_dur

        # detect time jumping forwards, as well as loops that are
        # inherently too slow
        if curr_time - self.last_time > self.sleep_dur * 2:
            self.last_time = curr_time
