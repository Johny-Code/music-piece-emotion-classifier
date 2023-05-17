import os
import sys
import pylab
import librosa
import numpy as np
sys.path.append("../utils/")

from read_database import read_excel_database
from cut_musical_piece import cut_musical_piece
from tqdm import tqdm
from scipy import signal
from matplotlib import cm


def save_melgram(filedir, database_filepath, outpath, emotions, file_format='mp3'):
    name, _, _, emotion = read_excel_database(database_filepath)
    name_emotion_dict = {name[i]: emotion[i] for i in range(len(name))}
    nb = len(os.listdir(filedir))
    os.makedirs(outpath, exist_ok=True)
    [os.makedirs(os.path.join(outpath, emotion), exist_ok=True) for emotion in emotions]
    pbar = tqdm(total=nb, unit="file")

    for file in os.listdir(filedir):
        if (file.endswith(".mp3")):
            name_ = file[:-4]
            emotion = name_emotion_dict[name_]
            full_name = filedir + file
            x, sr = librosa.load(full_name, sr=44100)
            x = librosa.util.normalize(x)
            x = cut_musical_piece(x, sr, 30)
            if (file_format == 'mp3'):
                melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=96, n_fft=2048, hop_length=512)
                melgram = librosa.amplitude_to_db(melspectogram, ref=1)[np.newaxis, np.newaxis, :, :]
                outfile = os.path.join(outpath, emotion, name_ + '.npy')
                np.save(outfile, melgram)
            if (file_format == 'jpg'):
                melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, n_fft=2048, hop_length=1024, power=2)
                log_power = librosa.power_to_db(melspectogram, ref=np.max)
                pylab.figure(figsize=(5, 5))
                pylab.axis('off')
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
                librosa.display.specshow(log_power, cmap=cm.jet)
                pylab.savefig(os.path.join(outpath, emotion, name_ + ".jpg"), bbox_inches=None, pad_inches=0)
                pylab.close()
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    emotions = ["happy", "sad", "angry", "relaxed"]
    outpath = "../database/melgrams/melgrams_2048_nfft_1024_hop_128_mel_jpg/"
    filedir = "../database/songs/"
    database_filepath = "../database/MoodyLyrics4Q_cleaned.csv"
    save_melgram(filedir, database_filepath, outpath, emotions, 'jpg')
