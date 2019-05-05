import librosa
import pandas as pd
import numpy as np

'''
Music genre is extracted from the filename.
We should modify it for future use with other datasets.
'''
def read_gui(gui):
    datasets_folder = '/'.join(gui.split('/')[:-1])+'/'
    df = pd.read_csv(gui, names=['name','path','genre'])
    df['path'] = df['name'].apply(lambda x: datasets_folder + x)
    df['name'] = df['path'].apply(lambda x: x.split('/')[-1].split('.au')[0])
    df['genre'] = df['name'].apply(lambda x: x.split('.')[0])
    return df

def create_spectrogram(audiopath):
    y, sr = librosa.load(audiopath, mono=True)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def create_numfeats(audiopath):
    y, sr = librosa.load(audiopath, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    feats = f'{np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        feats += f' {np.mean(e)}'
    return feats.split() 

def extract_spectrograms(gui):
    df_all = read_gui(gui)

    genres_list = df_all['genre'].unique().tolist()
    genres_dict = {genres_list[i] : i for i in range(len(genres_list))}
    genres_revdict = {v: k for k, v in genres_dict.items()}

    y = []
    X_spect = np.empty((0, 640, 128))
    count = 0

    for index, row in df_all.iterrows():
        try:
            count += 1

            track_id = index
            audiopath = str(row['path'])
            genre = str(row['genre'])

            spect = create_spectrogram(audiopath)

            # Normalize for small shape differences
            spect = spect[:640, :]

            X_spect = np.append(X_spect, [spect], axis=0)
            y.append(genres_dict[genre])

            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(y)

    return X_spect, y_arr

def extract_numfeats(gui):
    df_all = read_gui(gui)

    genres_list = df_all['genre'].unique().tolist()
    genres_dict = {genres_list[i] : i for i in range(len(genres_list))}
    genres_revdict = {v: k for k, v in genres_dict.items()}

    y = []
    X_spect = np.empty((0, 25))
    count = 0

    for index, row in df_all.iterrows():
        try:
            count += 1

            track_id = index
            audiopath = str(row['path'])
            genre = str(row['genre'])

            feats = create_numfeats(audiopath)

            X_spect = np.append(X_spect, [feats], axis=0)
            y.append(genres_dict[genre])

            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(y)

    return X_spect, y_arr

