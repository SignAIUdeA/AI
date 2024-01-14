import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from typing import List
import random

def load_data_methodology_cutting(df,size,val_split_label):
    """
    df: dataframe with the file names and the class names
    size: size of the cut of the data
    val_split_label: label of the validation data -> M1, M2, M3, M4, M5

    return: X, y, X_val, y_val
    """
    
    
    X_list = []
    y_list = []
    X_val_list = []
    y_val_list = []
    for file_name, class_name in df.iterrows():
        if val_split_label in file_name:
            X_val_list.append(np.load(f"../Clean_Data/{class_name.iloc[0]}/{file_name}")[:size,:])
            X_val_list.append(np.load(f"../Data_Augmented_Clean/{class_name.iloc[0]}/{file_name}")[:size,:])
            y_val_list.append(class_name.iloc[0])
            y_val_list.append(class_name.iloc[0])
        else:
            X_list.append(np.load(f"../Clean_Data/{class_name.iloc[0]}/{file_name}")[:size,:])
            X_list.append(np.load(f"../Data_Augmented_Clean/{class_name.iloc[0]}/{file_name}")[:size,:])
            y_list.append(class_name.iloc[0])
            y_list.append(class_name.iloc[0])

    X = np.stack(X_list)
    y = np.array(y_list)
    X_val = np.stack(X_val_list)
    y_val = np.array(y_val_list)
    return X, to_categorical(y), X_val, to_categorical(y_val)


def load_data_methodology_random_cutting(df : pd.DataFrame ,val_split_labels: List[str], size: int, seed: int = 42):
    """
    df: dataframe with the file names and the class names
    size: size of the cut of the data
    val_split_label: label of the validation data -> M1, M2, M3, M4, M5

    return: X, y, X_val, y_val
    """
    
    
    X_list = []
    X_val_list = []
    y_val_list = []
    y_list = []

    for file_name, class_name in df.iterrows():
        for val_split_label in val_split_labels:
            x_original = np.load(f"../Clean_Data/{class_name.iloc[0]}/{file_name}")[:,:]
            x_augmented = np.load(f"../Data_Augmented_Clean/{class_name.iloc[0]}/{file_name}")[:,:]
            size_x = x_original.shape[0]

            if size_x > size:
                start = random.randint(0,size_x-size)
                x_original = x_original[start:start+size,:]
                x_augmented = x_augmented[start:start+size,:]
            else:
                x_original = np.pad(x_original,((0,size-size_x),(0,0)),mode="constant")
                x_augmented = np.pad(x_augmented,((0,size-size_x),(0,0)),mode="constant")

            if val_split_label in file_name:
                X_val_list.append(x_original)
                X_val_list.append(x_augmented)
                y_val_list.append(class_name.iloc[0])
                y_val_list.append(class_name.iloc[0])
            else:
                X_list.append(x_original)
                X_list.append(x_augmented)
                y_list.append(class_name.iloc[0])
                y_list.append(class_name.iloc[0])

   
    X = np.stack(X_list)
    y = np.array(y_list)
    X_val = np.stack(X_val_list)
    y_val = np.array(y_val_list)


    return X, to_categorical(y), X_val, to_categorical(y_val)
