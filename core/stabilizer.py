import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self, border_crop_ratio=0.04):
        self.ref_gray = None
        self.ref_pts = None
        self.border_crop_ratio = border_crop_ratio
        self.feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=20, blockSize=7)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def stabilize(self, frame):
        h, w = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.ref_gray is None:
            self.ref_gray = curr_gray.copy()
            self.ref_pts = cv2.goodFeaturesToTrack(self.ref_gray, **self.feature_params)
            return self._crop_border(frame, w, h)

        if self.ref_pts is None or len(self.ref_pts) < 15:
            self.ref_pts = cv2.goodFeaturesToTrack(self.ref_gray, **self.feature_params)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.ref_gray, curr_gray, self.ref_pts, None, **self.lk_params
        )

        idx = np.where(status == 1)[0]
        if len(idx) < 10:
            return self._crop_border(frame, w, h)

        good_ref_pts = self.ref_pts[idx]
        good_curr_pts = curr_pts[idx]

        matrix, _ = cv2.estimateAffinePartial2D(
            good_curr_pts, good_ref_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if matrix is None:
            return self._crop_border(frame, w, h)

        stabilized_frame = cv2.warpAffine(
            frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return self._crop_border(stabilized_frame, w, h)

    def _crop_border(self, img, w, h):
        if self.border_crop_ratio <= 0:
            return img
        dx = int(w * self.border_crop_ratio)
        dy = int(h * self.border_crop_ratio)
        cropped = img[dy:h - dy, dx:w - dx]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)