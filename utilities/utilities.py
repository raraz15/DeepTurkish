#!/usr/bin/env python
# coding: utf-8

import os 
from itertools import groupby

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def dataset_pointers(dataset_name, spectrogram_type='dB'):
    """
    Returns the path of the spectrograms, training and test dataframes and the alphabet.

        Parameters:
        ----------
            dataset_name (string) : METUbet or Mozilla(for example)
            specrogram_type (string) : type of spectrograms to train the model with. Default = dB.

        Returns:
        --------
            spectrogram_dir (string) : the path to the spectograms
            df_dir0, df_dir1 (DataFrame) : dataframe of the training and test datasets.
            alphabet (list) : the alphabet containing the blank symbol, read.
    """

    if dataset_name == "METUbet":

        dataset_dir = os.path.join("data","datasets",dataset_name,"data") # directory of the dataset

        spectrogram_dir = os.path.join(dataset_dir,"spectrograms",spectrogram_type) # training data

        df_dir0 = os.path.join(dataset_dir,"METUbet_encoded_train.pkl") # directory of the dataframes
        df_dir1 = os.path.join(dataset_dir,"METUbet_encoded_test.pkl")

        alphabet_dir = os.path.join(dataset_dir,'METUbet_alphabet.csv')
        alphabet = pd.read_csv(alphabet_dir,delimiter=",",header=None,encoding='utf8',skip_blank_lines=False)[0].tolist()
    
    else:

        dataset_name = "cv-corpus-5.1-2020-06-22"

        dataset_dir = os.path.join("data","datasets",dataset_name,"tr")

        spectrogram_dir = os.path.join(dataset_dir,"spectrograms",spectrogram_type) # training data

        df_dir0 = os.path.join(dataset_dir,'cv-corpus-5.1-2020-06-22_validated_simple_ordered_train.pkl')
        df_dir1 = os.path.join(dataset_dir,'cv-corpus-5.1-2020-06-22_validated_simple_ordered_test.pkl')

        alphabet_dir = os.path.join(dataset_dir,'cv-corpus-5.1-2020-06-22_validated_simple_alphabet.csv')
        alphabet = pd.read_csv(alphabet_dir,delimiter=",",header=None,encoding="utf8",skip_blank_lines=False)[0].tolist()


    print("Alphabet with length {}:,".format(len(alphabet)))
    print(alphabet)

    return spectrogram_dir, df_dir0, df_dir1, alphabet 


class IterMeter(object):
    """Keeps track of total iterations, epochs, project and model names"""

    def __init__(self,project_name,model_name,total_epochs=0):
        self.val = 0
        self.epoch = 0
        self.total_epochs = total_epochs
        self.project_name = project_name
        self.model_name = model_name

    def step(self):
        self.val += 1

    def get(self):
        return self.val
    
    def step_epoch(self):
        self.epoch += 1
        
    def get_epoch(self):
        return self.epoch
    
    def get_total_epochs(self):
        return self.total_epochs
    
    def get_project_name(self):
        return self.project_name
    
    def get_model_name(self):
        return self.model_name


def save_checkpoint(new_state, metric, iter_meter):
    """
    Compares the selected metric of the current model and the best model so far and keeps the information of the better.

        Parameters:
        -----------
            new_state (dict) : dictionary containing the model's information.
            metric (string) : the key to the metric in the state dict
            iter_meter (IterMeter) : IterMeter object of the experiment
    """ 
    
    model_name = iter_meter.get_model_name()

    folder_dir = os.path.join("data","models and losses",iter_meter.get_project_name(),model_name)  
    file_dir = os.path.join(folder_dir, model_name+'.pt') 

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    if iter_meter.get_epoch() > 1: # compare with best previous epoch

        old_state = torch.load(file_dir)
        #old_state = load_checkpoint(file_dir)

        if old_state[metric] > new_state[metric]:
            state = new_state
        else:
            state = old_state

    else: # if first epoch, do not compare
        state = new_state

    torch.save(state,file_dir)

    
def load_checkpoint(checkpoint_path):
    """
    Loads the model's, optimizer's and the scheduler's state dicts for continuing training or later testing.

        Parameters:
        -----------
            checkpoint_path (string) : path to the .pt file checkpoint

        Returns:
        --------
            corresponding state dictionaries
    """
    
    state = torch.load(checkpoint_path)
    
    model_state = state['model_dict']
    optimizer_state = state['optim_dict']
    scheduler_state = state['scheduler_dict']
       
    return model_state, optimizer_state, scheduler_state
    

def ctc_collapse_function(decoded, blank_idx): 
    """
    Collapses the repeated symbols and removes blank outputs for a single decoded output.

        Parameters:
        -----------
            decoded (list) : the decoded ctc output (alphabet correspondings)
            blank_idx (int) : idx of the blank symbol in the alphabet
    """

    return [i for i,_ in groupby(decoded) if i != blank_idx]


def code_to_str(code,alphabet):
    """
    Takes a code list/tensor and gets the corresponding sentence using the alphabet.
    Alphabet must have has blank(0) at index zero, each symbol is an idx.
    """
        
    return  "".join([alphabet[symbol] for symbol in code]) #sentence


def targets_to_sentences(targets,target_lengths,alphabet):
    """
    Takes 1D targets tensor containing the merged targets with the target_lengths tensors,
    separates each targets and decodes them with the given alphabet.
    Returns the sentences in a list.
    """

    target_lengths_list = target_lengths.tolist()
    
    batch_size = len(target_lengths_list)
    
    target_lengths_list.insert(0,0)

    start_idx = 0
    end_idx = 0

    sentences = []
    for i in range(batch_size):

        start_idx += target_lengths_list[i]
        end_idx += target_lengths_list[i+1]

        target = targets[start_idx:end_idx]

        sentence = code_to_str(target,alphabet)

        sentences.append(sentence)
        
    return sentences    