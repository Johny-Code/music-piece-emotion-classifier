import os
import sys
import librosa
import numpy as np
sys.path.append("../utils/")

from read_database import read_extended_excel_database
from cut_musical_piece import cut_musical_piece
from tqdm import tqdm
from matplotlib import cm
from PIL import Image


def save_melgram(filedir, database_filepath, outpath, emotions, sets, file_format='mp3', n_mels=96, 
                 n_fft=2048, hop_length=512, should_cut=False):
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
                x = cut_musical_piece(x, sr, 30, "beginning")
                if (file_format == 'mp3'):
                    melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
                    melgram = librosa.amplitude_to_db(melspectogram, ref=1)[np.newaxis, np.newaxis, :, :]
                    outfile = os.path.join(outpath, split, emotion, name_ + '.npy')
                    np.save(outfile, melgram)
                if (file_format == 'jpg'):
                    melspectogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2)
                    log_power = librosa.power_to_db(melspectogram, ref=np.max)
                    resized_spectrogram = (log_power - np.min(log_power)) / (np.max(log_power) - np.min(log_power)) * 255
                    resized_spectrogram = resized_spectrogram.astype(np.uint32)
                    
                    image = Image.fromarray(np.uint8(resized_spectrogram)*255)
                    gray_img = image.convert('L').rotate(180)
                   
                    # resize image, no improvement
                    if (should_cut):
                        original_width, original_height = gray_img.size
                        aspect_ratio = original_width / original_height
                        new_width = 1024
                        new_height = int(new_width / aspect_ratio)
                        resized_image = gray_img.resize((new_width, new_height), Image.BICUBIC)
                        gray_img = resized_image
                    
                    output_image_path = os.path.join(outpath, split, emotion, name_ + ".jpg")
                    gray_img.save(output_image_path)
            except KeyError:
                print(f"\nPassing {name_}")
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    emotions = ["happy", "sad", "angry", "relaxed"]
    sets = ['train', 'test', 'val']
    outpaths = [ "../database/melgrams/to_copy/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray_first30s/"]
    filedir = "../database/songs/"
    database_filepath = "../database/MoodyLyrics4Q_cleaned_split.csv"
    
    save_melgram(filedir, database_filepath, outpaths[0], emotions, sets, 'jpg', n_mels=128, 
                 n_fft=2048, hop_length=1024, should_cut=False)
