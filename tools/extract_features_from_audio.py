import pandas as pd
import librosa
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def cut_musical_piece(x, sr):
    middle = int(x.shape[0]/2)
    lower = int(middle - sr*time/2)
    upper = int(middle + sr*time/2)
    return x[lower:upper]


def min_max_scale(series, columns):
    scaler = MinMaxScaler()
    scaler.fit(series)
    scaled = scaler.fit_transform(series)
    scaled_df = pd.DataFrame(scaled, columns=columns)
    return scaled_df


def standard_scaler(series, columns):
    scaler = StandardScaler()
    scaler.fit(series)
    scaled = scaler.fit_transform(series)
    scaled_df = pd.DataFrame(scaled, columns=columns)
    return scaled_df


def flatten(l):
    return [item for sublist in l for item in sublist]


def generate_mean_std(data):
    mean=np.mean(data)
    std=np.std(data)
    return mean, std


def extract_tempo(x, sr):
    onset_env = librosa.onset.onset_strength(y=x, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    return pd.DataFrame(data=[tempo[0]], columns=["tempo"])


def extract_zero_crossing_rate(x, hop_length, frame_length):
    zrate=librosa.feature.zero_crossing_rate(x, hop_length=hop_length, frame_length=frame_length)
    zrate_mean=np.mean(zrate)
    zrate_std=np.std(zrate)
    return pd.DataFrame({'zcr_mean': [zrate_mean], 'zcr_std': [zrate_std]})


def extract_spectral_features(x, sr, hop_length, n_fft):
    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr, hop_length=hop_length, n_fft=n_fft)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, hop_length=hop_length, n_fft = n_fft)[0]
    spectral_flux = librosa.onset.onset_strength(y=x, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=x, hop_length=hop_length, n_fft=n_fft)[0]
    collist_centroids=['cent_mean','cent_std']
    collist_rolloff=['rolloff_mean','rolloff_std']
    collist_flux = ['flux_mean','flux_std']
    collist_flatness=['flatness_mean','flatness_std']
    all_spect_columns = collist_centroids+collist_rolloff+collist_flux+collist_flatness
    
    spectral_dict = {}
    features = [spectral_centroids, spectral_rolloff, spectral_flux, spectral_flatness]
    j=0
    for feature in features:
        mss = generate_mean_std(feature)
        for i in range(0,2):
            spectral_dict[all_spect_columns[i+j]] = [mss[i]]
        j += 2   
    return pd.DataFrame(data=spectral_dict)


def extract_MFCCs(x, sr, hop_length=int(512/2), mfcc_features_nb = 40, n_fft=512):
    mfccs = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length, n_mfcc=mfcc_features_nb, n_fft=n_fft)
    mfccs_mean=np.mean(mfccs,axis=1)
    mfccs_std=np.std(mfccs,axis=1)
    mfccs_df=pd.DataFrame()
    for i in range(0,mfcc_features_nb):
        mfccs_df['mfccs_mean_'+str(i)]=mfccs_mean[i]
    for i in range(0,mfcc_features_nb):
        mfccs_df['mfccs_std_'+str(i)]=mfccs_std[i]       
    mfccs_df.loc[0]=np.concatenate((mfccs_mean,mfccs_std),axis=0)  
    
    #1s length
    mfccs = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(sr/2), n_mfcc=mfcc_features_nb, n_fft=sr)
    mfccs_mean=np.mean(mfccs,axis=1)
    mfccs_std=np.std(mfccs,axis=1)
    mfccs_df_2=pd.DataFrame()
    for i in range(0,mfcc_features_nb):
        mfccs_df_2['mfccs_mean_1_'+str(i)]=mfccs_mean[i]
    for i in range(0,mfcc_features_nb):
        mfccs_df_2['mfccs_std_1_'+str(i)]=mfccs_std[i]       
    mfccs_df_2.loc[0]=np.concatenate((mfccs_mean,mfccs_std),axis=0) 
     
    #30s length
    mfccs = librosa.feature.mfcc(y=x, sr=sr, hop_length=sr*30, n_mfcc=mfcc_features_nb, n_fft=sr*30)
    mfccs_mean=np.mean(mfccs,axis=1)
    mfccs_std=np.std(mfccs,axis=1)
    mfccs_df_3=pd.DataFrame()
    for i in range(0,mfcc_features_nb):
        mfccs_df_3['mfccs_mean_30_'+str(i)]=mfccs_mean[i]
    for i in range(0,mfcc_features_nb):
        mfccs_df_3['mfccs_std_30_'+str(i)]=mfccs_std[i]       
    mfccs_df_3.loc[0]=np.concatenate((mfccs_mean,mfccs_std),axis=0) 
     
    return pd.concat([mfccs_df, mfccs_df_2, mfccs_df_3], axis=1)


def extract_OCS(x, sr, hop_length, n_fft):
    S = np.abs(librosa.stft(x))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, hop_length=hop_length, n_fft=n_fft)
    contrast_mean=np.mean(contrast,axis=1)
    contrast_std=np.std(contrast,axis=1)
    collist = []
    ocs_df=pd.DataFrame()

    for i in range(0,7):
        collist.append('contrast_mean_'+str(i))
    for i in range(0,7):
        collist.append('contrast_std_'+str(i))
    for c in collist:
        ocs_df[c]=0
        
    data=np.concatenate((contrast_mean,contrast_std),axis=0)
    ocs_df.loc[0]=data
    return ocs_df


def extract_chromagram(x, sr, hop_length, n_fft):
    chromagram = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length, n_fft=n_fft)
    chroma_mean=np.mean(chromagram,axis=1)
    chroma_std=np.std(chromagram,axis=1)

    chroma_df=pd.DataFrame()
    for i in range(0,12):
        chroma_df['chroma_mean_'+str(i)]=chroma_mean[i]
    for i in range(0,12):
        chroma_df['chroma_std_'+str(i)]=chroma_std[i]
        
    chroma_df.loc[0]=np.concatenate((chroma_mean,chroma_std),axis=0)
    return chroma_df


def extract_all_features(output_dfs, hop_length, n_fft):
    nb = len(os.listdir(filedir))
    pbar = tqdm(total=nb, unit="file")
    for file in os.listdir(filedir):
        if(file.endswith(".mp3")):
            x, sr = librosa.load(filedir+file, sr=44100)
            x = librosa.util.normalize(x)
            x = cut_musical_piece(x, sr)
            zero_crossing_rate = extract_zero_crossing_rate(x, hop_length, n_fft)
            tempo = extract_tempo(x, sr)
            spectral_df = extract_spectral_features(x, sr, hop_length, n_fft)
            mfccs_df = extract_MFCCs(x, sr, hop_length, 40, n_fft)
            ocs_df = extract_OCS(x, sr, hop_length, n_fft)
            chroma_df = extract_chromagram(x, sr, hop_length, n_fft)
            
            all_features = [zero_crossing_rate, tempo, spectral_df, mfccs_df, ocs_df, chroma_df]
            all_features_df=pd.concat(all_features, axis=1)
            output_dfs.append(pd.concat([all_features_df], ignore_index = True, axis=0))
        pbar.update()
    pbar.close()


def write_into_csv_file(filepath, dataframe):
    filedir, _ = os.path.split(filepath)
    os.makedirs(filedir, exist_ok=True)
    dataframe.to_csv(filepath, index=True)


def read_mood(filename):
    df = pd.read_csv(filename)
    return df['mood']


def join_emotion_with_features(database_filepath, csv_filepath, nb):
    mood = read_mood(database_filepath)
    df = pd.read_csv(csv_filepath, index_col=0)
    df['emotion'] = mood[:nb]
    return df


if __name__=="__main__":
    records_nb = 1902
    n_fft = 2048
    hop_length = int(n_fft/2)
    time = 30
    feature_path = "../database/features/1600_2048_nfft_norm_40_mfcc.csv"
    original_database_path = "../database/MoodyLyrics4Q.csv"
    filedir = '../database/songs/'
    output_dfs = []
    
    extract_all_features(output_dfs, hop_length, n_fft)
    final_df = pd.concat(output_dfs, ignore_index=True)
    
    normalized_df = final_df.copy()
    normalized_df = standard_scaler(normalized_df, final_df.columns)
    write_into_csv_file(feature_path, normalized_df)
    
    joined_df = join_emotion_with_features(original_database_path, feature_path, records_nb)
    write_into_csv_file(feature_path, joined_df)
    