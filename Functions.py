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

def perspectiveTransform(myImage, transformMatix):
    imageSize = (myImage.shape[1], myImage.shape[0])
    transformedImage = cv2.warpPerspective(myImage, transformMatix, imageSize, flags=cv2.INTER_LINEAR)
    return transformedImage

def testingSpeedEnhancement(myImage, cameraMatrix, distortionCoefficients ,transformMatrix):
    processedImage = undistortImage(myImage, cameraMatrix, distortionCoefficients)
    processedImage = hsl_and_verticalEdges_wThreshold(processedImage)
    processedImage = perspectiveTransform(processedImage, transformMatrix)
    return processedImage



def slidingWindow(perspectiveTransformImage, windowsNUM = 10, margin = 100, threshold = 50):
    
    h = perspectiveTransformImage.shape[0]
    w = perspectiveTransformImage.shape[1]
      
    histogram = np.sum(perspectiveTransformImage[int(h/2):, :], axis=0)
       
    listLeftLaneIndices = []
    listRightLaneIndices = [] 
    xCurrLeft = np.argmax(histogram[:int(w/2)])                #leftPeak
    xCurrRight = np.argmax(histogram[int(w/2):]) + int(w/2)    #rightPeak
    
    winHeight = int(h/windowsNUM) 
    
    nonZero2DIndices = np.nonzero(perspectiveTransformImage)
    nonZeroYindices = nonZero2DIndices[0]
    nonZeroXindicies = nonZero2DIndices[1]

    for win in range(windowsNUM):

        winYlowerBound = h - (win + 1) * winHeight      
        winYupperBound = h - win * winHeight       
        
        winXlowerBoundLeft = xCurrLeft - margin       
        winXupperBoundLeft =  xCurrLeft + margin
        winXlowerBoundRight = xCurrRight - margin
        winXupperBoundRight = xCurrRight + margin

        nonZeroLeft_withinWindow = np.nonzero(((nonZeroYindices >= winYlowerBound) & (nonZeroYindices < winYupperBound) 
                          & (nonZeroXindicies >= winXlowerBoundLeft) & (nonZeroXindicies < winXupperBoundLeft)))[0]

        nonZeroRight_withinWindow = np.nonzero(((nonZeroYindices >= winYlowerBound) & (nonZeroYindices < winYupperBound) 
                          & (nonZeroXindicies >= winXlowerBoundRight)  & (nonZeroXindicies < winXupperBoundRight)))[0]
        
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

def fitCurve(LX, LY, RX, RY):

    leftCurve = np.polyfit(LY, LX, 2)
    rightCurve =  np.polyfit(RY, RX, 2)
    return leftCurve, rightCurve

def generatePlottingValues(leftCurve, rightCurve, imageShape):
    pointYaxis = np.linspace(0, imageShape[0]-1, imageShape[0])
    
    leftXvalues =  np.polyval(leftCurve, pointYaxis)
    rightXvalues = np.polyval(rightCurve, pointYaxis)
    
    return pointYaxis, leftXvalues, rightXvalues


def markLane(myImage, leftCurve, rightCurve, Inverse_transformMatix):
    zerosImage = np.zeros_like(myImage[:,:,0]).astype('uint8')
    colorImageZero = np.dstack((zerosImage, zerosImage, zerosImage))
    
    y, xLeft, xRight = generatePlottingValues(leftCurve, rightCurve, myImage.shape)


    pointLeft = np.array([np.transpose(np.vstack([xLeft, y]))])
    pointRight = np.array([np.flipud(np.transpose(np.vstack([xRight, y])))])
    points = np.hstack((pointLeft, pointRight))
    
    cv2.fillPoly(colorImageZero, np.int_([points]), (0,255, 0))
    
    lanesInverseTransferedImage = cv2.warpPerspective(colorImageZero, Inverse_transformMatix, (myImage.shape[1], myImage.shape[0])) 
    
    # Combine the result with the original image
    combined = cv2.addWeighted(myImage, 1, lanesInverseTransferedImage, 0.3, 0)
   
    return combined