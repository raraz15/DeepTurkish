#!/usr/bin/env python
# coding: utf-8

import os

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import utilities as utils


def make_loaders(data_parameters, sortagrad=False):
    """
    Creates the training, validation and test loaders.
    Do NOt specify sortagrad for second and onwards epochs only specify for the first.
    You must provide a sorted dataset if sortagrad=True
    """
    
    df_train = pd.read_pickle(data_parameters['dataframe_dir_train']) # Read the Dataframes
    df_test = pd.read_pickle(data_parameters['dataframe_dir_test'])
  
    train_val_partition, train_val_labels = create_train_val_partition(df_train, data_parameters['split_ratio'], # partition the training set
                                                                             data_parameters['batch_size'])
    test_partition, test_labels = create_test_partition(df_test,data_parameters['batch_size']) 

    train_set = Dataset(train_val_partition['train'],train_val_labels,data_parameters['train_dir']) # Create a Dataset Object
    validation_set = Dataset(train_val_partition['validation'],train_val_labels,data_parameters['train_dir'])
    test_set = Dataset(test_partition['test'],test_labels,data_parameters['test_dir'])
    
    # Construct the data loaders with or without SortaGrad
    if sortagrad:
        
        # Set the shuffle false for the first epoch
        data_parameters_local = data_parameters['loader_parameters'].copy()
        data_parameters_local['shuffle'] = False
        
        train_loader = torch.utils.data.DataLoader(train_set,**data_parameters_local)
        validation_loader = torch.utils.data.DataLoader(validation_set,**data_parameters_local)
        test_loader = torch.utils.data.DataLoader(test_set,**data_parameters_local)
               
    else:
        
        # shuffle the training set
        train_loader = torch.utils.data.DataLoader(train_set,**data_parameters['loader_parameters'])
        
        # not the validation and test sets for better monitoring
        data_parameters_local = data_parameters['loader_parameters'].copy()
        data_parameters_local['shuffle'] = False
        
        validation_loader = torch.utils.data.DataLoader(validation_set,**data_parameters_local)
        test_loader = torch.utils.data.DataLoader(test_set,**data_parameters_local)
             
    return train_loader, validation_loader, test_loader


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, name_list, transcriptions, data_dir):
        """
        Initializes a Dataset Object. Stores all labels and the list of IDs we want to generate at each pass.
        
            Parameters
            ----------
                list_paths : list
                        List containing the file name for the audio recordings.
                transcriptions : dictionary
                        Dictionary of {path:transcription}
                data_dir : str
                        Directory pointing to the spectrograms.
        """ 
        
        self.transcriptions = transcriptions
        self.name_list = name_list
        self.data_dir = data_dir
        
    def __len__(self):
        """Reutns the total number of samples"""
        
        return len(self.name_list)
    
    def __getitem__(self, index):
        """
        Generates one sample of data.
        
        Parameters
        ----------    
            index : int
                index of desired sample
               
        Returns
        -------
            X : tensor of shape (N_frames,(1+N_fft/2)), dtype=torch.float32 
                Normalized spectrogram of the utterance. 
            N_frames : torch.long
                Number of frames the spectrogram has. N_frames is a function of hop size...
            target : 1D tensor,  long for native,dtype torch.int32(to use cudnn)
                Target sequence of the utterance               
            target_length : torch.long
                Length of the target sequence
        """
        
        # Read the spectrogram
        file_name = self.name_list[index] # Select sample
        
        ID = file_name.split('.')[0] # get rid of the .mp3 or .wav extension
                                      
        csv_path = os.path.join(self.data_dir,ID+'.csv') # Path of the spectrogram
                
        data = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32) # Assumes no header, no row index
        # Data in (f,t) format
        
        N_frames = torch.tensor((data.shape[1]-1)//2,dtype=torch.long) #get the number of frames        
                
        data = MinMaxScaler(copy=False).fit_transform(data) # Min max scale each feature independently
        
        X = torch.from_numpy(data) # Convert to tensor 
        X = torch.transpose(X,0,1).contiguous() # (t,f) for pad sequence 
                                                       
        target = torch.tensor(self.transcriptions[file_name],dtype=torch.long) # Get the corresponding transcription
        
        target_length = torch.tensor(target.shape[0],dtype=torch.long)
       
        return X, N_frames, target, target_length
    
    
def pad_collate_fn(data):
    """
    
        parameters:
        -----------
        
            data (tupple): features has shape (T,F)
    
        returns:
        --------
        
            packed_features (PackedSequence) : 
            frame_lengths (tensor) : dtype = torch.long
            targets (tensor) : 1D tensor
            target_lengths (tensor) : dtype = torch.long
    """
    
    X, N_frames, y, S = zip(*data) # collect all Xs in a single tuple....

    # Dtype is required by native pytorch
    frame_lengths = torch.tensor(N_frames,dtype=torch.long) #Frame lengths of each instance 
        
    features = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-80.0) 
    features = torch.transpose(features,1,2).contiguous().unsqueeze(dim=1) # (B,1,F,T)
        
    target_lengths = torch.tensor(S,dtype=torch.long) # corresponding target lengths
    
    targets = torch.cat(y)#.to(device) # target sequences 
          
    return features, frame_lengths, targets, target_lengths


def create_train_val_partition(df, split_ratio, N_batch):
    """
    Creates a partition of the dataset into training and validation and creates a label dictionary.
    
    Parameters
    ----------
        df : pandas dataframe
            Dataframe corresponding to the dataset.
        split_ratio : float
            Training sample percentage, must be between 0 and 1.
        N_batch : int
            Batch size
        shuffle : 
            Shuffle the dataframe before or not. Default True.
            
    Returns
    -------
        partition : Dict
            {'train' : List of IDs,
            'validation' : List of IDs}

        lables : Dict
            {'ID' : transcription (list) }
    """
              
    ID = df['path'].tolist()
    transcriptions = df['encoded'].tolist()    

    N_train = int(df.shape[0]*split_ratio)
    N_train = N_train-(N_train%N_batch) # Make N_train = int multiple of N_batch    
         
    # Split IDs into train and validation
    partition = dict()
    partition['train'] = ID[:N_train]
    partition['validation'] = ID[N_train:]

    labels = dict()    
    # Save all the transcriptions into labels dictionary
    for i,val in enumerate(ID): 
        labels[val] = transcriptions[i]   
        
    return partition, labels


def create_test_partition(df, N_batch):
    """
    Creates a partition of the dataset into a single test set.
    
    Parameters
    ----------
        df : pandas dataframe
            Dataframe corresponding to the dataset.
        split_ratio : float
            Training sample percentage, must be between 0 and 1.
        N_batch : int
            Batch size
        shuffle : 
            Shuffle the dataframe before or not. Default True.
            
    Returns
    -------
        partition : Dict
            {'test': list of IDs}
        lables : Dict
            {'ID' : transcription (list) }
    """
   
    ID = df['path'].tolist()
    transcriptions = df['encoded'].tolist()    

    partition = dict()
    partition['test'] = ID #

    labels = dict()    
    # Save all the transcriptions into labels dictionary
    for i,val in enumerate(ID): 
        labels[val] = transcriptions[i]   
        
    return partition, labels