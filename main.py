import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import Functions

myImage = cv2.imread("test_images/test3.jpg")
myImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2RGB)

with open('dist_pickle.p', 'rb') as f:
    parameters = pickle.load(f)
    cameraMatrix = parameters['mtx']
    distortionCoefficients = parameters['dist']

undistort = Functions.undistortImage(myImage, cameraMatrix, distortionCoefficients)

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
src_image2 = cv2.imread("test_images/test5.jpg")
src_image2 = cv2.cvtColor(src_image2, cv2.COLOR_BGR2RGB)
#
thresholdImage = Functions.hsl_and_verticalEdges_wThreshold(src_image2)

transfered_image2 = np.copy(thresholdImage)
transfered_image2 = Functions.perspectiveTransform(transfered_image2, transformMatrix)

H_edgeDetectionImage = Functions.edgeDetection_1D_wThreshold(myImage, 'horizontal', 3, (20,200))
V_edgeDetectionImage = Functions.edgeDetection_1D_wThreshold(myImage, 'vertical', 3, (20,200))

hslImage = Functions.hls_wThreshold(myImage, (150, 255))

combined = Functions.hsl_and_verticalEdges_wThreshold(myImage)

trail = Functions.testingSpeedEnhancement(myImage, cameraMatrix, distortionCoefficients, transformMatrix)

histogram = np.sum(transfered_image2[int(height/2):,:], axis=0)

pro_test_image_A = Functions.testingSpeedEnhancement(myImage, cameraMatrix, distortionCoefficients, transformMatrix)
left_x, left_y, right_x, right_y = Functions.slidingWindow(pro_test_image_A)
left_fit, right_fit = Functions.fitCurve(left_x, left_y, right_x, right_y)

LX, LY, RX, RY = Functions.slidingWindow(trail)
LF, RF = Functions.fitCurve(LX, LY, RX, RY)
final_test_image_C = Functions.markLane(Functions.undistortImage(myImage, cameraMatrix, distortionCoefficients), LF, RF, inverseTransformMatix)
print(Functions.calcCameraOffset(LF, RF, pro_test_image_A.shape), "m")
#laneCurvature = Functions.calcRadiusCurvature(left_fit, right_fit, pro_test_image_A.shape)
#print(laneCurvature, 'm')

fig = plt.figure(figsize=(8.5, 4.8))

gs1 = plt.GridSpec(3, 3, left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, wspace = 0.05, hspace = 0.05)
ax1 = fig.add_subplot(gs1[:-1, :-1])
ax1.imshow(final_test_image_C)
ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)

ax2 = fig.add_subplot(gs1[-1, 0])
ax2.imshow(final_test_image_C)
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
ax6.set_xlim([0, width])
ax6.plot(histogram)
#ax6.imshow(trail, cmap = 'gray')
#ax6.axes.xaxis.set_visible(False)
#ax6.axes.yaxis.set_visible(False)


plt.show()
