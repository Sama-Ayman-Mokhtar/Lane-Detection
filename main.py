import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test_images/test1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


fig = plt.figure(figsize=(8.5, 4.8))

gs1 = plt.GridSpec(3, 3, left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, wspace = 0.05, hspace = 0.05)
ax1 = fig.add_subplot(gs1[:-1, :-1])
ax1.imshow(img)
ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)

ax2 = fig.add_subplot(gs1[-1, 0])
ax2.imshow(img)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)

ax3 = fig.add_subplot(gs1[-1, 1])
ax3.imshow(img)
ax3.axes.xaxis.set_visible(False)
ax3.axes.yaxis.set_visible(False)

ax4 = fig.add_subplot(gs1[-1, -1])
ax4.imshow(img)
ax4.axes.xaxis.set_visible(False)
ax4.axes.yaxis.set_visible(False)

ax5 = fig.add_subplot(gs1[1, -1])
ax5.imshow(img)
ax5.axes.xaxis.set_visible(False)
ax5.axes.yaxis.set_visible(False)

ax6 = fig.add_subplot(gs1[0, -1])
ax6.imshow(img)
ax6.axes.xaxis.set_visible(False)
ax6.axes.yaxis.set_visible(False)
plt.show()