import pandas as pd
import cv2
import numpy as np
from Preprocessing.holistic import transform_video
from progress.bar import Bar

df = pd.read_csv("../Metadata/data.csv")
df.set_index("File",inplace=True)

def process_data():
    with Bar('Processing...',max=df.size) as bar:
        for file_name, class_name in df.iterrows():
            result = transform_video(cv2.VideoCapture("../Raw_Data/{}/{}".format(class_name.iloc[0],file_name)))
            np.save("../Clean_Data/{}/{}.npy".format(str(class_name.iloc[0]),str(file_name[:-4])),result)
            bar.next()

