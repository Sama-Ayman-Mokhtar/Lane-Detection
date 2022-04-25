import cv2
import numpy as np
import matplotlib.pyplot as plt

def undistortImage(myImage, cameraMatrix, distortionCoefficients):
    return cv2.undistort(myImage, cameraMatrix, distortionCoefficients, None, cameraMatrix)

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

def testingSpeedEnhancement(myImage, transformMatrix):
    processedImage = hsl_and_verticalEdges_wThreshold(myImage)
    processedImage = perspectiveTransform(processedImage, transformMatrix)
    return processedImage

def perspectiveTransform(myImage, transformMatix):
    imageSize = (myImage.shape[1], myImage.shape[0])
    transformedImage = cv2.warpPerspective(myImage, transformMatix, imageSize, flags=cv2.INTER_LINEAR)
    return transformedImage

def slidingWindow(perspectiveTransformImage, windowsNUM = 10, margin = 100, threshold = 50):
    
    h = perspectiveTransformImage.shape[0]
    w = perspectiveTransformImage.shape[1]
      
    histogram = np.sum(perspectiveTransformImage[int(h/2):, :], axis=0)
       
    listLeftLaneIndices = []
    listRightLaneIndices = [] 
    xCurrLeft = np.argmax(histogram[:int(w/2)])                #leftPeak
    xCurrRight = np.argmax(histogram[int(w/2):]) + int(w/2)    #rightPeak
    
    winHeight = int(h/windowsNUM) 
    
    nonZero2DIndices = perspectiveTransformImage.nonzero()
    nonZeroYindices = nonZero2DIndices[0]
    nonZeroXindicies = nonZero2DIndices[1]

    for win in range(windowsNUM):

        winYlowerBound = h - (win + 1) * winHeight      
        winYupperBound = h - win * winHeight       
        
        winXlowerBoundLeft = xCurrLeft - margin       
        winXupperBoundLeft =  xCurrLeft + margin
        winXlowerBoundRight = xCurrRight - margin
        winXupperBoundRight = xCurrRight + margin

        nonZeroLeft_withinWindow = ((nonZeroYindices >= winYlowerBound) & (nonZeroYindices < winYupperBound) 
                          & (nonZeroXindicies >= winXlowerBoundLeft) & (nonZeroXindicies < winXupperBoundLeft)).nonzero()[0]

        nonZeroRight_withinWindow = ((nonZeroYindices >= winYlowerBound) & (nonZeroYindices < winYupperBound) 
                          & (nonZeroXindicies >= winXlowerBoundRight)  & (nonZeroXindicies < winXupperBoundRight)).nonzero()[0]
        
        listLeftLaneIndices.append(nonZeroLeft_withinWindow)
        listRightLaneIndices.append(nonZeroRight_withinWindow)
        
        if len(nonZeroLeft_withinWindow) > threshold:
            xCurrLeft = np.mean(nonZeroXindicies[nonZeroLeft_withinWindow]).astype('int')

        if len(nonZeroRight_withinWindow) > threshold:
            xCurrRight = np.mean(nonZeroXindicies[nonZeroRight_withinWindow]).astype('int')
        
    listLeftLaneIndices = np.concatenate(listLeftLaneIndices)
    listRightLaneIndices = np.concatenate(listRightLaneIndices)
    
    LX = nonZeroXindicies[listLeftLaneIndices]
    LY = nonZeroYindices[listLeftLaneIndices]
    RX = nonZeroXindicies[listRightLaneIndices]
    RY = nonZeroYindices[listRightLaneIndices]

    return LX, LY, RX, RY