from pickle import TRUE
import cv2
import numpy as np

DEBUG_MODE = True

height, width = 720, 1280

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