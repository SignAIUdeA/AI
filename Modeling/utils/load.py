import numpy as np
from progress.bar import Bar
from tensorflow.keras.utils import to_categorical

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
    with Bar("Processing...", max = df.size) as bar:
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
            bar.next()
    X = np.stack(X_list)
    y = np.array(y_list)
    X_val = np.stack(X_val_list)
    y_val = np.array(y_val_list)
    return X, to_categorical(y), X_val, to_categorical(y_val)