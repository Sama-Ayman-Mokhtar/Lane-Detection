import Lane
import pickle
import constants
import Functions
from moviepy.editor import VideoFileClip


with open('dist_pickle.p', 'rb') as f:
    parameters = pickle.load(f)
    cameraMatrix = parameters['mtx']
    distortionCoefficients = parameters['dist']

lane = Lane.Lane()
if constants.DEBUG_MODE:
    outputVideo = 'Result_Debug_Mode_ON.mp4'
else:
    outputVideo = 'Result_Debug_Mode_OFF.mp4'

inputVideo = VideoFileClip("test_videos/project_video.mp4")
process_video = lambda process_frame:process_frame()
project_clip = inputVideo.fl_image(lambda frame: Functions.pipline(frame, lane, cameraMatrix, distortionCoefficients, constants.transformMatrix, constants.inverseTransformMatix))
test_clip = project_clip
test_clip.write_videofile(outputVideo, audio=False)
