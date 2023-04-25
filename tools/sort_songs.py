import os
from tqdm import tqdm
import pandas as pd
import shutil


def read_excel_database(filename):
    df = pd.read_csv(filename)
    id = df["index"]
    mood = df['mood']
    return id, mood


def copy_mp3_file(filedir, database_filepath, outpath):
    name, emotion = read_excel_database(database_filepath)
    name_emotion_dict = {name[i]: emotion[i] for i in range(len(name))}
    nb = len(os.listdir(filedir))
    pbar = tqdm(total=nb, unit="file")
    
    for file in os.listdir(filedir):
        if(file.endswith(".mp3")):
            name_ = file[:-4]
            emotion = name_emotion_dict[name_]
            shutil.copy2(f'{filedir}{file}', f'{outpath}{emotion}/{file}')
        pbar.update()
    pbar.close()


if __name__=="__main__":
    outpath = "../database/songs_divided/"
    filedir = "../database/songs/"
    database_filepath = "../database/MoodyLyrics4Q.csv"
    copy_mp3_file(filedir, database_filepath, outpath)