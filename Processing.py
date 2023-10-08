import cv2

from Holistic import transform_video

video = cv2.VideoCapture('./Raw_Data/1/S1V1C1M1A Clip1.mp4')
transform_video(video, 'output.mp4')