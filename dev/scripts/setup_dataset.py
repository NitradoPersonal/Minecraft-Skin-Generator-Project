# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
import os
import shutil
# REPLACE THIS PATH TO WHERE YOU WANT TO ST-
# ORE THE TRAINING DATA

#path="/home/kaggle/data/mcskin"
def get_dataset():
    alxmamaev_minecraft_skins_path = kagglehub.dataset_download('alxmamaev/minecraft-skins')
    return alxmamaev_minecraft_skins_path
#os.makedirs("/home/kaggle/data/mcskin/", exist_ok=True, mode=0o7)
#shutil.move(alxmamaev_minecraft_skins_path, "/home/kaggle/data/mcskin")

print('Data source import complete.')
