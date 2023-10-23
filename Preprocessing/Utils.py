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
    
    data.to_csv('../data.csv', index=False)
