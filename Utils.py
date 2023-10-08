import cv2
import os
from Holistic import transform_video
import pandas as pd
import numpy as np


def create_dataframe():
    classes = ['1','2','3','4','5']
    data = []
    for c in classes:
        for f in os.listdir(f'./Raw_Data/{c}'):
            data.append({ 
                "File": f,
                "Class": c
            })
    data = pd.DataFrame(data)
    
    data.to_csv('data.csv', index=False)

create_dataframe()


"""
base_directory = './Raw_Data/'

for i in range(1,6):
    counter = 0
    for index, row in df.iterrows():
        try:
            video_path = os.path.join(base_directory,str(i),row[i])
            cap = cv2.VideoCapture(video_path)
        except e:
            print(e)
        if not cap.isOpened():
            print("Error al abrir el video.")
            print(video_path,"/")
            break
            
        frames = []
        
        while True:
            ret, frame = cap.read()

            if not ret or len(frames)==30:
                break
                
            frames.append(frame)
        print(len(frames))
        cap.release()
        
        video_array = np.stack(frames)
        
        output_directory = os.path.join("MP_Data",str(i),str(counter))
        
        output_filename = f'{0}.npy'
        
        np.save(os.path.join(output_directory,output_filename),video_array)
        print(os.path.join(output_directory,output_filename))
        counter += 1

video = cv2.VideoCapture('./Raw_Data/1/S1V1C1M1A Clip1.mp4')
transform_video(video, 'output.mp4')

"""