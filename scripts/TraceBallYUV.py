import numpy as np
import cv2 as cv
from time import sleep

roi = cv.imread('_1.png')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2YUV)

# cv.imshow("y", hsvt[:,:,0])
# cv.imshow("u", hsvt[:,:,2])
# cv.imshow("v", hsvt[:,:,1])

# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)

for i in range(0, 350):
    target = cv.imread('image/{}.png'.format(i))
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2YUV)
    dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    # Now convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    cv.filter2D(dst,-1,disc,dst)
    # threshold and binary AND
    ret,thresh = cv.threshold(dst,200,255,0)
    thresh = cv.merge((thresh,thresh,thresh))
    # res = cv.bitwise_and(target,thresh)
    # res = np.vstack((thresh))
    # cv.imshow("", res)
    cv.imshow("", thresh)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.waitKey()
# cv.imwrite('res.jpg',res)