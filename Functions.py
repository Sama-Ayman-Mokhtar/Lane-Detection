import cv2
import numpy as np
import matplotlib.pyplot as plt

def undistortImage(myImage, cameraMatrix, distortionCoefficientsortionCoefficients):
    return cv2.undistort(myImage, cameraMatrix, distortionCoefficientsortionCoefficients, None, cameraMatrix)

def edgeDetection_1D_wThreshold(mymyImage, orientation='horizontal', kernelSize = 3, threshold = (0, 255)):
    grayMymyImage = cv2.cvtColor(mymyImage, cv2.COLOR_BGR2GRAY)

    if orientation == 'horizontal':
        sobelmyImage = cv2.Sobel(grayMymyImage, cv2.CV_64F, 0, 1, kernelSize)
    elif orientation == 'vertical':
        sobelmyImage = cv2.Sobel(grayMymyImage, cv2.CV_64F, 1, 0, kernelSize)

    absSobelmyImage = np.abs(sobelmyImage)

    scaledSobelmyImage = np.uint8(255*absSobelmyImage/np.max(absSobelmyImage))

    thresholdSobelmyImage =  np.zeros_like(scaledSobelmyImage)
    thresholdSobelmyImage[(scaledSobelmyImage >= threshold[0]) & (scaledSobelmyImage <= threshold[1])] = 1
    
    return thresholdSobelmyImage

def hls_wThreshold(mymyImage, thresh= (0,255)):
    hlsmyImage = cv2.cvtColor(mymyImage, cv2.COLOR_RGB2HLS)
    s_channel_hslmyImage = hlsmyImage[:,:,2]

    thresholdSobelmyImage = np.zeros_like(s_channel_hslmyImage)
    thresholdSobelmyImage[(s_channel_hslmyImage > thresh[0]) & (s_channel_hslmyImage <= thresh[1])] = 1
    
    return thresholdSobelmyImage

def hsl_and_verticalEdges_wThreshold(mymyImage):

    thresh_x = (20, 200)
    thresh_hls = (150, 255)
    
    thresholdVerticalEdges = edgeDetection_1D_wThreshold(mymyImage, 'vertical', 5, (20, 200))
    thresholdHLS = hls_wThreshold(mymyImage, (150, 255))
    
    # Combined binary output
    combined_binary = np.zeros_like(thresholdVerticalEdges)
    combined_binary[(thresholdHLS == 1) | (thresholdVerticalEdges == 1)] = 1

    return combined_binary

def perspectiveTransform(mymyImage, transformMatix):
    myImageSize = (mymyImage.shape[1], mymyImage.shape[0])
    transformedmyImage = cv2.warpPerspective(mymyImage, transformMatix, myImageSize, flags=cv2.INTER_LINEAR)
    return transformedmyImage

def testingSpeedEnhancement(mymyImage, cameraMatrix, distortionCoefficientsortionCoefficients ,transformMatrix):
    processedmyImage = undistortImage(mymyImage, cameraMatrix, distortionCoefficientsortionCoefficients)
    processedmyImage = hsl_and_verticalEdges_wThreshold(processedmyImage)
    processedmyImage = perspectiveTransform(processedmyImage, transformMatrix)
    return processedmyImage

def slidingWindow(perspectiveTransformmyImage, windowsNUM = 10, margin = 100, threshold = 50):
    
    h = perspectiveTransformmyImage.shape[0]
    w = perspectiveTransformmyImage.shape[1]
      
    histogram = np.sum(perspectiveTransformmyImage[int(h/2):, :], axis=0)
       
    listLeftLaneIndices = []
    listRightLaneIndices = [] 
    xCurrLeft = np.argmax(histogram[:int(w/2)])                #leftPeak
    xCurrRight = np.argmax(histogram[int(w/2):]) + int(w/2)    #rightPeak
    
    winHeight = int(h/windowsNUM) 
    
    nonZero2DIndices = np.nonzero(perspectiveTransformmyImage)
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

def generatePlottingValues(leftCurve, rightCurve, myImageShape):
    pointYaxis = np.linspace(0, myImageShape[0]-1, myImageShape[0])
    
    leftXvalues =  np.polyval(leftCurve, pointYaxis)
    rightXvalues = np.polyval(rightCurve, pointYaxis)
    
    return pointYaxis, leftXvalues, rightXvalues

def markLane(mymyImage, leftCurve, rightCurve, Inverse_transformMatix):
    zerosmyImage = np.zeros_like(mymyImage[:,:,0]).astype('uint8')
    colormyImageZero = np.dstack((zerosmyImage, zerosmyImage, zerosmyImage))
    
    y, xLeft, xRight = generatePlottingValues(leftCurve, rightCurve, mymyImage.shape)


    pointLeft = np.array([np.transpose(np.vstack([xLeft, y]))])
    pointRight = np.array([np.flipud(np.transpose(np.vstack([xRight, y])))])
    points = np.hstack((pointLeft, pointRight))
    
    cv2.fillPoly(colormyImageZero, np.int_([points]), (0,255, 0))
    
    lanesInverseTransferedmyImage = cv2.warpPerspective(colormyImageZero, Inverse_transformMatix, (mymyImage.shape[1], mymyImage.shape[0])) 
    
    # Combine the outputImage with the original myImage
    combined = cv2.addWeighted(mymyImage, 1, lanesInverseTransferedmyImage, 0.3, 0)
   
    return combined

def calcCameraOffset(leftCurve, rightCurve, myImageShape):
    h = myImageShape[0]  
    w = myImageShape[1]
    
    leftXvalue = np.polyval(leftCurve, h)
    rightXvalue = np.polyval(rightCurve, h)
    
    mid = w/2
    laneWidthPixels = np.abs(rightXvalue - leftXvalue)
    offset = (mid - np.mean([leftXvalue, rightXvalue])) * (3.7 / laneWidthPixels)

    return offset

def calcRadiusCurvature(leftCurve, rightCurve, imageShape):
    
    plot_y = np.linspace(0, 719, num=720)
    leftXvalues = np.polyval(leftCurve, plot_y)
    rightXvalues = np.polyval(rightCurve, plot_y)
    
    left_fit_m = np.polyfit(plot_y*(30/720), leftXvalues*(3.7/700), 2)
    right_fit_m = np.polyfit(plot_y*(30/720), rightXvalues*(3.7/700), 2)
    
    leftRadiusOfCurvature = ((1 + (2*left_fit_m[0]*imageShape[0]*(30/720) + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    rightRadiusOfCurvature = ((1 + (2*right_fit_m[0]*imageShape[0]*(30/720) + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])
    radiusOfCurvature = np.mean([leftRadiusOfCurvature, rightRadiusOfCurvature])

    return radiusOfCurvature

def pipline(myImage, lane, cameraMatrix, distortionCoefficients, transformMatrix, InverseTransformMatrix):

    undistortedImage = undistortImage(myImage, cameraMatrix, distortionCoefficients)
    hls_VedgesImage = hsl_and_verticalEdges_wThreshold(undistortedImage)
    birdViewImage = perspectiveTransform(hls_VedgesImage, transformMatrix)  
    LX, LY, RX, RY = slidingWindow(birdViewImage)
    LF, RF = fitCurve(LX, LY, RX, RY)
    
    if lane.getLeftCurve() is None:
        lane.setLeftCurve(LF)
        lane.setRightCurve(RF)
    else:
        lane.setLeftCurve(0.8 * lane.getLeftCurve() + 0.2 * LF)
        lane.setRightCurve(0.8 * lane.getRightCurve() + 0.2 * RF)
 
    radiusOfCurvature = calcRadiusCurvature(lane.getLeftCurve(), lane.getRightCurve(), myImage.shape)
    
    if lane.getRadiusOfCurvature() is None:
        lane.setRadiusOfCurvature(radiusOfCurvature)
    else:
        lane.setRadiusOfCurvature(0.95 * lane.getRadiusOfCurvature() + 0.05 * radiusOfCurvature)

    lane.setLaneOffset(calcCameraOffset(lane.getLeftCurve(), lane.getRightCurve(), myImage.shape))

    outputImage = markLane(undistortedImage, lane.getLeftCurve(), lane.getRightCurve(), InverseTransformMatrix)
    
    fontStyle = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    cv2.putText(outputImage, 'radius of curvature: {:.2f} m'.format(lane.getRadiusOfCurvature()), (100 ,100),fontStyle, 1, (255,0,0), 2)
    cv2.putText(outputImage, 'lane offset: {:.2f} m'.format(lane.getLaneOffset()), (200 ,200), fontStyle, 1, (255,0,0), 2)
    
    return outputImage