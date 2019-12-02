import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
template = cv.imread('_1.png')
template = cv.cvtColor(template, cv.COLOR_BGR2HSV)
template = template[:,:,1]
w, h = template.shape[::-1]


for i in range(0, 350):
    img_rgb = cv.imread('image/{}.png'.format(i))
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)
    img_gray = img_hsv[:,:,1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv.imshow("", img_rgb)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.waitKey()