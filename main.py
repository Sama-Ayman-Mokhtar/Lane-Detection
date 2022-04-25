import cv2
import numpy as np
import matplotlib.pyplot as plt
import Functions

myImage = cv2.imread("test_images/test1.jpg")
myImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2RGB)



straightLinesImage = cv2.imread("test_images/straight_lines1.jpg")
straightLinesImage = cv2.cvtColor(straightLinesImage, cv2.COLOR_BGR2RGB)
#

height = straightLinesImage.shape[0]
width = straightLinesImage.shape[1]

source_topLeft = (592, 450)         
source_bottomLeft = (180, height)      
source_bottomRight = (1130, height)    
source_topRight = (687, 450)  
sourcePoints = np.array([[source_topLeft, source_bottomLeft, source_bottomRight, source_topRight]]).astype('float32')

destination_topLeft = (240, 0)         
destination_bottomLeft = (240, height)      
destination_bottomRight = (1040, height)    
destination_topRight = (1040, 0) 
destiantionPoints = np.array([[destination_topLeft, destination_bottomLeft, destination_bottomRight, destination_topRight]]).astype('float32')

transformMatrix = cv2.getPerspectiveTransform(sourcePoints, destiantionPoints)
inverseTransformMatix = cv2.getPerspectiveTransform(destiantionPoints, sourcePoints)

# test straight line
src_image1 = np.copy(straightLinesImage)
src_image1 = cv2.polylines(src_image1, sourcePoints.astype('int32'), 1, (255,0,0), thickness=6)

transfered_image1 = np.copy(straightLinesImage)
transfered_image1 = Functions.perspectiveTransform(transfered_image1, transformMatrix)
transfered_image1 = cv2.polylines(transfered_image1, destiantionPoints.astype('int32'), 1, (255,0,0), thickness=6)



#test curved lines
src_image2 = cv2.imread("test_images/test2.jpg")
src_image2 = cv2.cvtColor(src_image2, cv2.COLOR_BGR2RGB)
#
thresholdImage = Functions.hsl_and_verticalEdges_wThreshold(src_image2)

transfered_image2 = np.copy(thresholdImage)
transfered_image2 = Functions.perspectiveTransform(transfered_image2, transformMatrix)

H_edgeDetectionImage = Functions.edgeDetection_1D_wThreshold(myImage, 'horizontal', 3, (20,200))
V_edgeDetectionImage = Functions.edgeDetection_1D_wThreshold(myImage, 'vertical', 3, (20,200))

hslImage = Functions.hls_wThreshold(myImage, (150, 255))

combined = Functions.hsl_and_verticalEdges_wThreshold(myImage)

trail = Functions.testingSpeedEnhancement(myImage, transformMatrix)

fig = plt.figure(figsize=(8.5, 4.8))

gs1 = plt.GridSpec(3, 3, left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, wspace = 0.05, hspace = 0.05)
ax1 = fig.add_subplot(gs1[:-1, :-1])
ax1.imshow(myImage)
ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)

ax2 = fig.add_subplot(gs1[-1, 0])
ax2.imshow(src_image1)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)

ax3 = fig.add_subplot(gs1[-1, 1])
ax3.imshow(transfered_image1)
ax3.axes.xaxis.set_visible(False)
ax3.axes.yaxis.set_visible(False)

ax4 = fig.add_subplot(gs1[-1, -1])
ax4.imshow(thresholdImage, cmap = 'gray')
ax4.axes.xaxis.set_visible(False)
ax4.axes.yaxis.set_visible(False)

ax5 = fig.add_subplot(gs1[1, -1])
ax5.imshow(transfered_image2, cmap='gray')
ax5.axes.xaxis.set_visible(False)
ax5.axes.yaxis.set_visible(False)

ax6 = fig.add_subplot(gs1[0, -1])
ax6.imshow(trail, cmap = 'gray')
ax6.axes.xaxis.set_visible(False)
ax6.axes.yaxis.set_visible(False)


plt.show()
