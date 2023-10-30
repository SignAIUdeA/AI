import pandas as pd
import cv2
from Preprocessing.holistic import transform_video
import numpy as np
from progress.bar import Bar

df = pd.read_csv("../Metadata/data.csv")
df.set_index("File",inplace=True)


def process_data():
    with Bar('Processing...',max=df.size) as bar:
        for file_name, class_name in df.iterrows():
            cap = cv2.VideoCapture("../Raw_Data/{}/{}".format(class_name.iloc[0],file_name))
            frame_width = int(cap.get(3)) 
            frame_height = int(cap.get(4)) 
            size = (frame_width, frame_height)
            name = "../Data_Augmented_Raw/{}/{}.mp4".format(class_name.iloc[0],file_name[:-4])
            mirror_video(cap, name,size)
            cap = cv2.VideoCapture("../Data_Augmented_Raw/{}/{}.mp4".format(class_name.iloc[0],file_name[:-4]))
            result = transform_video(cap)
            np.save("../Data_Augmented_Clean/{}/{}.npy".format(str(class_name.iloc[0]),str(file_name[:-4])),result)
            bar.next()


def mirror_img(image):
    return cv2.flip(image,1)

def mirror_video(video,file_name,size):
    result = cv2.VideoWriter(file_name,  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 
    while True:
        ret, frame = video.read()
        if ret:
            result.write(mirror_img(frame))
        else:
            break

process_data()