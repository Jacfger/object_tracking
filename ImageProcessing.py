# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

import cv2
import numpy as np

img = cv2.imread('sampling_image.png')
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
u = yuv[:,:,1]
v = yuv[:,:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

hist, xedges, yedges = np.histogram2d(u.ravel(), v.ravel(), bins=(10,10))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
plt.title("X vs. Y Amplitudes for ____ Data")
plt.xlabel("My X data source")
plt.ylabel("My Y data source")
plt.savefig("Your_title_goes_here")
plt.show()