import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T


def get_melspectrogram_db(wav, sr, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300,
                          top_db=80):
    """
    Get the melspectrogram data for an audio file
    :param file_path: path to audio file
    :param sr: sampling rate
    :param length: crop the input audio to this length in seconds
    :param n_fft:
    :param hop_length:
    :param n_mels:
    :param fmin:
    :param fmax:
    :param top_db:
    :return:
    """
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def get_melspectrogram_db_from_file(file_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300,
                                    top_db=80):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin,
                                         fmax=fmax)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel


def get_melspectrogram_db_tensor(waveform, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300,
                                 top_db=80, mel_length=216):
    
    hop_length = len(waveform) // (mel_length - 1)

    new_audio_length =  mel_length * hop_length

    waveform = waveform[:new_audio_length]
    
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax
    )

    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to dB scale
    db_transform = T.AmplitudeToDB(top_db=top_db)
    mel_spectrogram_db = db_transform(mel_spectrogram)

    return mel_spectrogram_db

def get_melspectrogram_db_tensor_maestro(waveform, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300,
                                 top_db=80, mel_length=216):
    
    hop_length = len(waveform) // (mel_length - 1)

    new_audio_length =  mel_length * hop_length

    waveform = waveform[:new_audio_length]

    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to dB scale
    db_transform = T.AmplitudeToDB(top_db=top_db)
    mel_spectrogram_db = db_transform(mel_spectrogram)


    return mel_spectrogram

def get_melspectrogram_db_tensor_from_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)  # [(2, 88576), 44100]
    waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[0] == 1:
        waveform = waveform.squeeze()

    
    # waveform = waveform.mean(dim=0).unsqueeze(0)
    mel_spectrogram_db = get_melspectrogram_db_tensor(waveform, sample_rate)

    return mel_spectrogram_db


def split_audio_data(wav_file_path, hop_length_audio=5, window_size=5):
    waveform, sample_rate = torchaudio.load(wav_file_path, normalize=True)  # get audio parameters
    
    # take the mean of the two channels
    waveform = waveform.mean(dim=0, keepdim=True)

    # remove extra dimension indicating number of channels
    waveform = waveform.squeeze()
        
    split_wvs = list()
    for i in np.arange(0, len(waveform) + 1, hop_length_audio * sample_rate):
        if i + hop_length_audio * sample_rate > len(waveform):
            # make sure last sample is as long as the others
            split_wvs.append(waveform[-window_size * sample_rate:])
        else:
            split_wvs.append(waveform[i:i + window_size * sample_rate])
    return split_wvs