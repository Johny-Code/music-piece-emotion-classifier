import os
import sys
import pylab
import librosa
import numpy as np
sys.path.append("../utils/")

from read_database import read_extended_excel_database
from cut_musical_piece import cut_musical_piece
from tqdm import tqdm
from scipy import signal
from matplotlib import cm
from PIL import Image


def save_melgram(filedir, database_filepath, outpath, emotions, sets, file_format='mp3'):
    name, _, _, emotion, split = read_extended_excel_database(database_filepath)
    name_emotion_dict = {name[i]: emotion[i] for i in range(len(name))}
    split_dict = {name[i]: split[i] for i in range(len(name))}
    nb = len(os.listdir(filedir))
    [os.makedirs(os.path.join(outpath, i, emotion), exist_ok=True) for i in sets for emotion in emotions]   
    pbar = tqdm(total=nb, unit="file")

    for file in os.listdir(filedir):
        if (file.endswith(".mp3")):
            try:
                name_ = file[:-4]
                emotion = name_emotion_dict[name_]
                split = split_dict[name_]
                full_name = filedir + file
                x, sr = librosa.load(full_name, sr=44100)
                x = librosa.util.normalize(x)
                x = cut_musical_piece(x, sr, 30)
                if (file_format == 'mp3'):
                    melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=96, n_fft=2048, hop_length=512)
                    melgram = librosa.amplitude_to_db(melspectogram, ref=1)[np.newaxis, np.newaxis, :, :]
                    outfile = os.path.join(outpath, split, emotion, name_ + '.npy')
                    np.save(outfile, melgram)
                if (file_format == 'jpg'):
                    melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, n_fft=2048, hop_length=1024, power=2)
                    log_power = librosa.power_to_db(melspectogram, ref=np.max)
                    desired_size = 216
                    resized_spectrogram = librosa.util.fix_length(log_power, size=desired_size, axis=1)
                    resized_spectrogram = (resized_spectrogram - np.min(resized_spectrogram)) / (np.max(resized_spectrogram) - np.min(resized_spectrogram)) * 255
                    resized_spectrogram = resized_spectrogram.astype(np.uint32)
                    image = Image.fromarray(np.uint8(cm.jet(resized_spectrogram)*255))
                    rgb_im = image.convert('RGB').rotate(180)
                    output_image_path = os.path.join(outpath, split, emotion, name_ + ".jpg")
                    rgb_im.save(output_image_path)
            except KeyError:
                print(f"\nPassing {name_}")
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    emotions = ["happy", "sad", "angry", "relaxed"]
    sets = ['train', 'test', 'val']
    outpath = "../database/melgrams/melgrams_2048_nfft_1024_hop_128_mel_jpg_divided_resized/"
    filedir = "../database/songs/"
    database_filepath = "../database/MoodyLyrics4Q_cleaned_split.csv"
    save_melgram(filedir, database_filepath, outpath, emotions, sets, 'jpg')
