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
                                 top_db=80):
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

def get_melspectrogram_db_tensor_from_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)  # [(2, 88576), 44100]
    # waveform = waveform.mean(dim=0).unsqueeze(0)
    mel_spectrogram_db = get_melspectrogram_db_tensor(waveform, sample_rate)
    return mel_spectrogram_db