import numpy as np
import cv2
import os
os.sys.path
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

#loading yolo weigths and configurations
weights_path = os.path.join("yolo", "yolov3.weights")
config_path = os.path.join("yolo", "yolov3.cfg")

#loading the neural net
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#getting layer names
names = net.getLayerNames() 
#'yolo_82' 'yolo_94' 'yolo_106'
layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

#Reading the 80-label file "COCO"
labels_path = os.path.join("yolo","coco.names")
labels = open(labels_path).read().strip().split("\n")

def detectCars(img):

    #Image height and width
    (H, W) = img.shape[:2]

    #loading the image as blob and feeding it ot the neural network
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), crop = False, swapRB = False)
    net.setInput(blob)

    #Supplying the network with the 3 layer names
    layers_output = net.forward(layers_names)
    
    boxes = []       
    confidences = [] 
    classIDs = []    
    
    for output in layers_output:
        for detection in output:
            #Scores are in the vector ranging from 5 -> 85
            scores = detection[5:]
            #The class id is the object that got the max score
            classID = np.argmax(scores)
            #The maximum score is the confidence of the detected object
            confidence = scores[classID]

            #Only if confidence is above 85%, consider it
            if (confidence > 0.85):
                '''
                The first two items in the vector
                are centers of box in x and y direction
                while the second two 
                are the height and width of the box'''
                box = detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype("int")

                # getting the top-left corner of the box
                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))
                # appending the box to the boxes list
                boxes.append([x, y, int(bw), int(bh)])
                # appending its confidence to the confidences list
                confidences.append(float(confidence))
                # appendings its classID to the classes list
                classIDs.append(classID)

    ''' 
    applying non maximum suppression so that an
    object does not get detected several times
    from different boxes'''
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
  
    #If no car is detected, return the image as is
    if len(idxs) == 0:
        return img

    #If a car or more got detected, draw a box around each car
    for i in idxs.flatten():
        (x, y) = [boxes[i][0], boxes[i][1]]
        (w, h) = [boxes[i][2], boxes[i][3]]
        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(img, "{}: {}".format(labels[classIDs[i]], confidences[i]), \
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 139,139), 2 )
    
    return img

#path to input video
video_path = os.path.join("test_videos", "project_video.mp4")

#applying the solution to video frames and combining them into video
inputVideo = VideoFileClip(video_path)
process_video = lambda process_frame:process_frame()
project_clip = inputVideo.fl_image(lambda frame: detectCars(frame))
test_clip = project_clip
test_clip.write_videofile("output.mp4", audio=False)