import constants
import cv2
import numpy as np


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

def hsl_and_verticalEdges_wThreshold(myImage):

    thresh_x = (20, 200)
    thresh_hls = (150, 255)
    
    thresholdVerticalEdges = edgeDetection_1D_wThreshold(myImage, 'vertical', 5, (20, 200))
    thresholdHLS = hls_wThreshold(myImage, (150, 255))
    
    # Combined binary output
    combined_binary = np.zeros_like(thresholdVerticalEdges)
    combined_binary[(thresholdHLS == 1) | (thresholdVerticalEdges == 1)] = 255

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
    
    outImg = np.dstack((perspectiveTransformmyImage, perspectiveTransformmyImage, perspectiveTransformmyImage))*255

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

        cv2.rectangle(outImg,(winXlowerBoundLeft,winYlowerBound),(winXupperBoundLeft,winYupperBound),(0,255,0), 2) 
        cv2.rectangle(outImg,(winXlowerBoundRight,winYlowerBound),(winXupperBoundRight,winYupperBound),(0,255,0), 2)
        
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

    leftCurve = np.polyfit(LY, LX, 2)
    rightCurve = np.polyfit(RY, RX, 2)

    outImg[nonZeroYindices[listLeftLaneIndices], nonZeroXindicies[listLeftLaneIndices]] = [255, 0, 0]
    outImg[nonZeroYindices[listRightLaneIndices], nonZeroXindicies[listRightLaneIndices]] = [0, 0, 255]
    
    # Generate x and y values for plotting
    '''
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

   
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    '''

    return LX, LY, RX, RY, outImg

def fitCurve(LX, LY, RX, RY):

    leftCurve = np.polyfit(LY, LX, 2)
    rightCurve =  np.polyfit(RY, RX, 2)
    return leftCurve, rightCurve

def generatePlottingValues(leftCurve, rightCurve, myImageShape):
    pointYaxis = np.linspace(0, myImageShape[0]-1, myImageShape[0])
    
    leftXvalues =  np.polyval(leftCurve, pointYaxis)
    rightXvalues = np.polyval(rightCurve, pointYaxis)
    
    return pointYaxis, leftXvalues, rightXvalues

def markLane(myImage, leftCurve, rightCurve, Inverse_transformMatix):
    zerosmyImage = np.zeros_like(myImage[:,:,0]).astype('uint8')
    colormyImageZero = np.dstack((zerosmyImage, zerosmyImage, zerosmyImage))
    
    y, xLeft, xRight = generatePlottingValues(leftCurve, rightCurve, myImage.shape)

    pointLeft = np.array([np.transpose(np.vstack([xLeft, y]))])
    pointRight = np.array([np.flipud(np.transpose(np.vstack([xRight, y])))])
    points = np.hstack((pointLeft, pointRight))
    

    cv2.fillPoly(colormyImageZero, np.int_([points]), (0,255, 0))
    cv2.polylines(colormyImageZero, np.int_([pointRight]), False, (0,0,255),10)
    cv2.polylines(colormyImageZero, np.int_([pointLeft]), False, (255,0,0),10)

    lanesInverseTransferedmyImage = cv2.warpPerspective(colormyImageZero, Inverse_transformMatix, (myImage.shape[1], myImage.shape[0])) 
    
    combined = cv2.addWeighted(myImage, 1, lanesInverseTransferedmyImage, 0.3, 0)
   
    return combined, colormyImageZero

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
    LX, LY, RX, RY, step4_slidingWinImg = slidingWindow(birdViewImage)
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

    markedImageCombined, markLaneImg = markLane(undistortedImage, lane.getLeftCurve(), lane.getRightCurve(), InverseTransformMatrix)
    fontStyle = cv2.FONT_HERSHEY_SIMPLEX

    if constants.DEBUG_MODE == "true":
        height, width = 1080, 1920
        outputImage = np.zeros((height,width),'uint8')
        outputImage = np.dstack((outputImage, outputImage, outputImage))

        step1_marking_source = cv2.polylines(undistortedImage, constants.sourcePoints.astype('int32'), 1, (255,0,0), thickness=6)
        cv2.putText(step1_marking_source, 'Undistort', (50 ,100),fontStyle, 4, (255,0,0), 3)
        
        step2_hsl_edge_img = np.zeros_like(myImage)
        step2_hsl_edge_img[:,:,0] = hls_VedgesImage
        step2_hsl_edge_img[:,:,1] = hls_VedgesImage
        step2_hsl_edge_img[:,:,2] = hls_VedgesImage
        cv2.putText(step2_hsl_edge_img, 'HSL and Vertical Edges', (50 ,100),fontStyle,2 , (255,255,0), 3)
        
        step3_BirdView_img = np.zeros_like(myImage)
        step3_BirdView_img[:,:,0] = birdViewImage
        step3_BirdView_img[:,:,1] = birdViewImage
        step3_BirdView_img[:,:,2] = birdViewImage
        cv2.putText(step3_BirdView_img, 'Bird View', (450 ,600),fontStyle, 4, (255,255,0), 3)

        cv2.putText(step4_slidingWinImg, 'Sliding Window', (450 ,650),fontStyle, 2, (255,255,0), 3)

        step5_markLaneImg = markLaneImg
        cv2.putText(step5_markLaneImg, 'Curve Fitting', (450 ,650),fontStyle, 2, (0,0,0), 3)

        outputImage[0:720,0:1280] = cv2.resize(markedImageCombined, (1280,720), interpolation=cv2.INTER_AREA)
        outputImage[0:360,1280:1920] = cv2.resize(step5_markLaneImg, (640,360), interpolation=cv2.INTER_AREA) #6
        outputImage[360:720,1280:1920] = cv2.resize(step4_slidingWinImg, (640,360), interpolation=cv2.INTER_AREA) #5
        outputImage[720:1080,1280:1920] = cv2.resize(step3_BirdView_img, (640,360), interpolation=cv2.INTER_AREA) #4
        outputImage[720:1080,640:1280] = cv2.resize(step2_hsl_edge_img, (640,360), interpolation=cv2.INTER_AREA) #3
        outputImage[720:1080,0:640] = cv2.resize(step1_marking_source, (640,360), interpolation=cv2.INTER_AREA) #2
        
    if constants.DEBUG_MODE == "false":
         outputImage = markedImageCombined


    cv2.putText(outputImage, 'radius of curvature: {:.2f} m'.format(lane.getRadiusOfCurvature()), (450 ,600),fontStyle, 1, (0,0,0), 3)
    cv2.putText(outputImage, 'lane offset: {:.2f} m'.format(lane.getLaneOffset()), (450 ,700), fontStyle, 1, (0,0,0), 3)
    
    return outputImage