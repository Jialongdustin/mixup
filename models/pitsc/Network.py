import argparse
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union
from collections import OrderedDict
from models.resnet18_encoder import *
from models.resnet20_cifar import *
import numpy as np
from models.augment import Augments


class StochasticClassifier(nn.Module):

    def __init__(self, num_features, num_classes, temp, etf_vec=None):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))# each rotation have individual variance here
        self.temp = temp
    
    def forward(self, x, stochastic=False):
        mu = self.mu
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4) # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.linear(x, weight)
        score = score * self.temp

        return score

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
            #self.num_features = 100
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        if self.args.dataset == 'librispeech':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        else:
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     

        self.set_fea_extractor_for_s2s()
        self.fc = StochasticClassifier(num_features = self.num_features, num_classes = self.args.num_all, temp = self.args.network.temperature)        
        augments_cfg = [{'type': 'BatchMixupTwoLabel', 'alpha': self.args.pre_mixup_alpha, 'num_classes': -1, 'prob': self.args.pre_mixup_prob}, 
         {'type': 'BatchCutMixTwoLabel', 'alpha': self.args.pre_mixup_alpha, 'num_classes': -1, 'prob': self.args.pre_cutmix_prob}, 
         {'type': 'IdentityTwoLabel', 'num_classes': -1, 'prob': self.args.pre_idty_prob}]

        self.augments = Augments(augments_cfg)

    def forward_metric(self, x, label, stochastic = False, aug=False):
        x_f, x_f_a = self.encode(x, label, aug)
        if 'cos' in self.mode:
            x = self.fc(x_f, stochastic)
           

        elif 'dot' in self.mode:
            x = self.fc(x_f)

        return x, x_f, x_f_a

    def forward_proto(self, x, stochastic = False):
        x = x.unsqueeze(1)
        x = self.slf_attn(x, x, x)
        x = x.squeeze(1)
        x = self.fc(x, stochastic)
        return x

    def get_mel(self, x):
        if x.shape[1] == 44100 or x.shape[1] == 176400 or x.shape[1] == 132300:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        # if x.shape[1] == 44100 or x.shape[1] == 64000 or x.shape[1] == 32000:
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        return x

    def encode(self, x, label=None, aug=False):
        x = self.get_mel(x)
        if aug:
            assert label is not None
            x, lam, gt_label, gt_label_aux = self.augments(x, label)
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x_f_a = x.squeeze(-1).squeeze(-1)
        x = x_f_a.unsqueeze(1)
        # x = self.slf_attn(x, x, x)
        x = x.squeeze(1)
        if aug:
            return x, x_f_a, lam, gt_label, gt_label_aux
        else:
            return x, x_f_a
        
    
    def forward(self, input, label=None, stochastic=False, aug=False):
        """
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        """

        if self.mode != 'encoder':
            if aug:
                x_f, _, lam, gt_label, gt_label_aux = self.encode(input, label, aug)
                x = self.fc(x_f, stochastic)
                return x, x_f, lam, gt_label, gt_label_aux
            else:
                x_f, x_f_a = self.encode(input, label, aug)
                x = self.fc(x_f, stochastic)
                return x, x_f, x_f_a
        elif self.mode == 'encoder':
            input = self.encode(input, label=None, aug=aug)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        class_list = torch.from_numpy(class_list)

        support_data, support_label = None, None
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            if len(dataloader)==1:
                support_data, support_label = [_.cuda() for _ in batch]
            data, _=self.encode(data)
            data = data.detach()

        if self.args.strategy.data_init_new:
            print("Not updating new class with class means")
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            print("Updating new class with class means ")
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.network.new_mode:  # further finetune
            print("started finetuning######")
            self.update_fc_ft(new_fc,data,label,session)
        new_fc = self.update_fc_avg(data, label, class_list)
        return support_data, support_label, new_fc

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.mu.data[class_index]=proto
            #self.fc.mu.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.network.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.network.new_mode:
            return self.args.network.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs.epochs_new):
                old_fc = self.fc.mu.data[:self.args.num_base + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.mu.data[self.args.num_base + self.args.way * (session - 1):self.args.num_base + self.args.way * session, :].copy_(new_fc.data)
    
    def set_fea_extractor_for_s2s(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # mel_bin = self.args.extractor.mel_bins
        mel_bin =128
        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = mel_bin
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = mel_bin
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 180
        ls_mel_bins = mel_bin
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bin) 


