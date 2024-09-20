import os
import os.path as osp
import re
import json
import time
import h5py
from matplotlib.font_manager import json_dump
import numpy as np
import random
import librosa
from pytz import timezone
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
import pandas as pd
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class FSDCLIPS(Dataset):

    def __init__(self, root='./', phase='train', 
                 index_path=None, index=None, k=5, base_sess=None, data_type='audio'):
        self.root = os.path.expanduser(root)
        self.root = root
        self.data_type = data_type
        # self.make_extractor()
        self.phase = phase
        # self.train = train  # training set or test set
        self.all_train_df = pd.read_csv("data/FMC/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta/mini/Fsc89-mini-fsci_train.csv")
        self.all_val_df = pd.read_csv("data/FMC/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta/mini/Fsc89-mini-fsci_val.csv")
        self.all_test_df = pd.read_csv("data/FMC/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta/mini/Fsc89-mini-fsci_test.csv")

        if phase == 'train':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
            else:
                """
                if shuffle_dataset:
                    random.seed(time.time)
                else:
                    random.seed(0)
                """
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
        elif phase == 'val':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        elif phase =='test':
            self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        


    def SelectfromClasses(self, df, index, per_num=None):
        """
        select k samples from list_dict which label=index 
        Args:
            paths (list):
            labels (list):
            index (list):
            k (int): 
        
        Returns:
            data_tmp (list):
            target_tmp (list):
        """
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == df['label'])[0]
            # random.shuffle(ind_cl)

            k = 0
            # ind_cl is the index list whose label equals i, 
            # start_idx make sure 
            # there is no intersection between train and test set  
            for j in ind_cl:
                filename = df['FSD_MIX_SED_filename'][j].replace('.wav', '_' + str(int(df['start_time'][j] * 44100)) + '.wav')
                path = os.path.join(self.root, self.data_type, df['data_folder'][j], filename)
                data_tmp.append(path)
                targets_tmp.append(df['label'][j])
                k += 1
                if per_num is not None:
                    if k >= per_num:
                        break
              
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        # feature = self.wave_to_tfr(path)
        # return feature, targets
        # feature = self.wave_to_logmel(path)
        audio, sr = torchaudio.load(path)
        return audio.squeeze(0), targets

    def wave_to_tfr(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        # - 直接在这里进行重采样
        # transform = torchaudio.transforms.Resample(sr, 16000)
        # waveform = transform(waveform)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                window_type='hanning', num_mel_bins=128, dither=0.0,
                                                frame_shift=10)
        fbank = fbank.view(1, fbank.shape[0], fbank.shape[1]).repeat(3, 1, 1)
        return fbank

    def make_extractor(self):
        sample_rate=44100
        window_size=2048 
        hop_size=1024 
        mel_bins=128 
        fmin=0 
        fmax=22050
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        # if torch.cuda.is_available():
            # self.spectrogram_extractor.cuda()
            # self.logmel_extractor.cuda()

    def wave_to_logmel(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        # if torch.cuda.is_available():
            # waveform.cuda()
        # - 直接在这里进行重采样
        # transform = torchaudio.transforms.Resample(sr, 16000)
        # waveform = transform(waveform)
        waveform = waveform - waveform.mean()
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.view(1, x.shape[2], x.shape[3]).repeat(3, 1, 1)
        return x

