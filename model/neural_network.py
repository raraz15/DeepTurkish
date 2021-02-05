#!/usr/bin/env python
# coding: utf-8

# reference https://www.comet.ml/site/customer-case-study-building-an-end-to-end-speech-recognition-model-in-pytorch-with-assemblyai/

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from  utilities.utilities import load_checkpoint


def make_model(hparams, blank, steps_per_epoch, device):

    model = SpeechRecognitionModel(hparams['n_cnn_layers'], hparams['n_rnn_layers'],
                                   hparams['rnn_dim'], hparams['n_class'],
                                   hparams['N_fft'], hparams['stride'], 
                                   hparams['dropout']).to(device)

    criterion = nn.CTCLoss(blank=blank, zero_infinity=True).to(device)
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=hparams['epochs'],
                                        anneal_strategy='linear')

    if hparams['model_dir']: # If a model directory is specified, load it.
        
        model_state, optimizer_state, scheduler_state = load_checkpoint(hparams['model_dir'])
        
        model.load_state_dict(model_state)
        #optimizer.load_state_dict(optimizer_state) 
        #scheduler.load_state_dict(scheduler_state)

    return model, criterion, optimizer, scheduler


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats, dropout): #
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, batch_first, dropout): #
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, N_fft, stride=2, dropout=0.1): #
        super(SpeechRecognitionModel, self).__init__()
        n_feats = ((N_fft//2)+1)//2
        self.cnn = nn.Conv2d(1, 32, 5, stride=stride, padding=(1,1))  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, n_feats=n_feats, dropout=dropout) #
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, batch_first=i==0, dropout=dropout) #
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
