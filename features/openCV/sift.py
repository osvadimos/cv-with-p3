import cv2
from skimage import io
import random
import numpy as np

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


image = cv2.imread("../pants.jpg")
image_rot = rotateImage(cv2.imread("../pants.jpg"), 50)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(gray, None)
kp_rot, desc_rot = sift.detectAndCompute(gray_rot, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc, desc_rot, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])
# Shuffle the matched keypoints
random.shuffle(good)
# cv2.drawMatchesKnn expects list of lists as matches.
image_match = cv2.drawMatchesKnn(image, kp, image_rot, kp_rot, good[:10], flags=2,
                   outImg=None)
cv2.imwrite('sift_matches.jpg', image_match)

io.imshow(image_match)
io.show()
exit()
