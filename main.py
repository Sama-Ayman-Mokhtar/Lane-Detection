import cv2
import numpy as np
import matplotlib.pyplot as plt

myImage = cv2.imread("test_images/test1.jpg")
myImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2RGB)

def perspectiveTransform(myImage, transformMatix):
    imageSize = (myImage.shape[1], myImage.shape[0])
    transformedImage = cv2.warpPerspective(myImage, transformMatix, imageSize, flags=cv2.INTER_LINEAR)
    return transformedImage

straightLinesImage = cv2.imread("test_images/straight_lines1.jpg")
straightLinesImage = cv2.cvtColor(straightLinesImage, cv2.COLOR_BGR2RGB)
#straightLinesImage = undistort_image(straight_lines, mtx, dist)

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

src_image = np.copy(straightLinesImage)
src_image = cv2.polylines(src_image, sourcePoints.astype('int32'), 1, (255,0,0), thickness=6)

transfered_image = np.copy(straightLinesImage)
transfered_image = perspectiveTransform(transfered_image, transformMatrix)
transfered_image = cv2.polylines(transfered_image, destiantionPoints.astype('int32'), 1, (255,0,0), thickness=6)

def edgeDetection_1D_wThreshold(myImage, orientation='horizontal', kernelSize = 3, threshold = (0, 255)):
    grayMyImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

    if orientation == 'horizontal':
        sobelImage = cv2.Sobel(grayMyImage, cv2.CV_64F, 0, 1, kernelSize)
    elif orientation == 'vertical':
        sobelImage = cv2.Sobel(grayMyImage, cv2.CV_64F, 1, 0, kernelSize)

    absSobelImage = np.abs(sobelImage)

    scaledSobelImage = np.uint8(255*absSobelImage/np.max(absSobelImage))

    thresholdSobelImage =  np.zeros_like(scaledSobelImage)
    thresholdSobelImage[(scaledSobelImage >= threshold[0]) & (scaledSobelImage <= threshold[1])] = 1
    
    return thresholdSobelImage

def hls_wThreshold(myImage, thresh= (0,255)):
    hlsImage = cv2.cvtColor(myImage, cv2.COLOR_RGB2HLS)
    s_channel_hslImage = hlsImage[:,:,2]

    thresholdSobelImage = np.zeros_like(s_channel_hslImage)
    thresholdSobelImage[(s_channel_hslImage > thresh[0]) & (s_channel_hslImage <= thresh[1])] = 1
    
    return thresholdSobelImage

def hsl_and_verticalEdges_wThreshold(myImage):

    thresh_x = (20, 200)
    thresh_hls = (150, 255)
    
    thresholdVerticalEdges = edgeDetection_1D_wThreshold(myImage, 'vertical', 5, (20, 200))
    thresholdHLS = hls_wThreshold(myImage, (150, 255))
    
    # Combined binary output
    combined_binary = np.zeros_like(thresholdVerticalEdges)
    combined_binary[(thresholdHLS == 1) | (thresholdVerticalEdges == 1)] = 1

    return combined_binary

H_edgeDetectionImage = edgeDetection_1D_wThreshold(myImage, 'horizontal', 3, (20,200))
V_edgeDetectionImage = edgeDetection_1D_wThreshold(myImage, 'vertical', 3, (20,200))

hslImage = hls_wThreshold(myImage, (150, 255))

combined = hsl_and_verticalEdges_wThreshold(myImage)

fig = plt.figure(figsize=(8.5, 4.8))

gs1 = plt.GridSpec(3, 3, left = 0.05, bottom = 0.05, right = 0.95, top = 0.95, wspace = 0.05, hspace = 0.05)
ax1 = fig.add_subplot(gs1[:-1, :-1])
ax1.imshow(straightLinesImage)
ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)

ax2 = fig.add_subplot(gs1[-1, 0])
ax2.imshow(src_image)
ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)

ax3 = fig.add_subplot(gs1[-1, 1])
ax3.imshow(transfered_image)
ax3.axes.xaxis.set_visible(False)
ax3.axes.yaxis.set_visible(False)

ax4 = fig.add_subplot(gs1[-1, -1])
ax4.imshow(hslImage, cmap='gray')
ax4.axes.xaxis.set_visible(False)
ax4.axes.yaxis.set_visible(False)

ax5 = fig.add_subplot(gs1[1, -1])
ax5.imshow(combined, cmap='gray')
ax5.axes.xaxis.set_visible(False)
ax5.axes.yaxis.set_visible(False)

ax6 = fig.add_subplot(gs1[0, -1])
ax6.imshow(myImage)
ax6.axes.xaxis.set_visible(False)
ax6.axes.yaxis.set_visible(False)


plt.show()
