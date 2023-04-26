import os
import pylab
import librosa
import numpy as np
from tqdm import tqdm
from scipy import signal
from matplotlib import cm
from utils import read_database, cut_musical_piece


def save_melgram(filedir, database_filepath, outpath, file_format='mp3'):
    name, emotion = read_database.read_excel_database(database_filepath)
    name_emotion_dict = {name[i]: emotion[i] for i in range(len(name))}
    nb = len(os.listdir(filedir))
    pbar = tqdm(total=nb, unit="file")
    
    for file in os.listdir(filedir):
        if(file.endswith(".mp3")):
            name_ = file[:-4]
            emotion = name_emotion_dict[name_]
            x, sr = librosa.load(filedir+file, sr=44100)
            x = librosa.util.normalize(x)
            x = cut_musical_piece.cut_musical_piece(x, sr, 30)
            if (file_format == 'mp3'):
                melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=96, n_fft=2048, hop_length=512)
                melgram = librosa.amplitude_to_db(melspectogram, ref=1)[np.newaxis,np.newaxis,:,:]
                outfile = outpath + emotion + '/' + name_+'.npy'
                np.save(outfile,melgram)
            if (file_format == 'jpg'):
                melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=96, n_fft=2048, hop_length=512, power=2)
                log_power = librosa.power_to_db(melspectogram, ref=np.max)
                pylab.figure(figsize=(5,5))
                pylab.axis('off') 
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
                librosa.display.specshow(log_power, cmap=cm.jet)
                pylab.savefig(outpath+emotion+'/' + name_+'.jpg', bbox_inches=None, pad_inches=0)
                pylab.close()
        pbar.update()
    pbar.close()


if __name__=="__main__":
    outpath = "../database/melgrams/melgrams_2048_nfft_512_hop_jpg/"
    filedir = "../database/songs/"
    database_filepath = "../database/MoodyLyrics4Q.csv"
    save_melgram(filedir, database_filepath, outpath, 'jpg')