import os
import pandas as pd

def create_dataframe():
    classes = ['1','2','3','4','5']
    data = []
    for c in classes:
        for f in os.listdir(f'../Raw_Data/{c}'):
            data.append({ 
                "File": f,
                "Class": c
            })
    data = pd.DataFrame(data)
    data.to_csv('../Metadata/data.csv', index=False)

def mp4_to_npy_dataframe():
    data = pd.read_csv("../Metadata/data.csv")
    len_data = len(data)
    for file_row in range(len_data):
        data.at[file_row, "File"] = data.at[file_row,"File"][:-4]+".npy"
    data.to_csv('../Metadata/data_curated.csv', index=False)
