import cv2
import numpy as np
from matplotlib import pyplot as plt


roi = cv2.imread('_2.png')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('image\\1.png')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

cv2.imshow("h", hsvt[:, :, 0])
cv2.imshow("s", hsvt[:, :, 1])
cv2.imshow("v", hsvt[:, :, 2])

# calculating object histogram
roihist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
counts, bins = np.histogram(hsv[:,:,1])
plt.hist(bins[:-1], bins, weights=counts)
plt.show()
# normalize histogram and apply backprojection
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [1], roihist, [0, 256], 1)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)

# threshold and binary AND
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)

# res = np.vstack((target,thresh,res))
cv2.imshow("", thresh)
cv2.waitKey()
