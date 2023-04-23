import os
import numpy as np
import librosa
from tqdm import tqdm
from scipy import signal
from sort_songs import read_excel_database
from extract_features_from_audio import cut_musical_piece


def save_melgram(filedir, database_filepath, outpath):
    name, emotion = read_excel_database(database_filepath)
    name_emotion_dict = {name[i]: emotion[i] for i in range(len(name))}
    nb = len(os.listdir(filedir))
    pbar = tqdm(total=nb, unit="file")
    
    for file in os.listdir(filedir):
        if(file.endswith(".mp3")):
            name_ = file[:-4]
            emotion = name_emotion_dict[name_]
            x, sr = librosa.load(filedir+file, sr=44100)
            x = librosa.util.normalize(x)
            x = cut_musical_piece(x, sr, 30)
            melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, n_fft=2048, hop_length=1024)
            melgram = librosa.amplitude_to_db(melspectogram, ref=1.0)[np.newaxis,np.newaxis,:,:]
            outfile = outpath + emotion + '/' + name_+'.npy'
            np.save(outfile,melgram)
        pbar.update()
    pbar.close()


if __name__=="__main__":
    outpath = "../database/melgrams/melgrams_2048_nfft_1024_hop_128_mel/"
    filedir = "../database/songs/"
    database_filepath = "../database/MoodyLyrics4Q.csv"
    save_melgram(filedir, database_filepath, outpath)