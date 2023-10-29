from Preprocessing.transform import process_data
from Preprocessing.utils import create_dataframe, mp4_to_npy_dataframe

print("Creating dataframes...")
create_dataframe()
mp4_to_npy_dataframe()
print("Done creating dataframes.")
print("Start to process data...")
process_data()