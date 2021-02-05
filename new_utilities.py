#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


class Dataset_readcsv(torch.utils.data.Dataset):
    
    def __init__(self, name_list,transcriptions,data_dir):
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
        'Reutns the total number of samples'
        
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
            X : tensor of shape (N_frames,(1+N_fft/2)), dtype=torch.float32 (???)
                Normalized (?????) spectrogram of the utterance. 
            N_frames : torch.long
                Number of frames the spectrogram has. N_frames is a function of hop size...
            target : 1D tensor,  long for native,dtype torch.int32(to use cudnn)
                Target sequence of the utterance               
            target_length : torch.long
                Length of the target sequence
        """
        
        # Read spectrogram
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
        
            data (tupple??): features has shape T,F
    
        returns:
        --------
        
            packed_features (PackedSequence) : 
            frame_lengths (tensor) : dtype = torch.long
            targets (tensor) : 1D tensor
            target_lengths (tensor) : dtype = torch.long
    """
    
    X,N_frames,y,S = zip(*data) # collect all Xs in a single tuple....

    # CTC accepts cudnn requires int32 PyTorch Native long
    frame_lengths = torch.tensor(N_frames,dtype=torch.long) # ,device=device Frame lengths of each instance 
        
    features = torch.nn.utils.rnn.pad_sequence(X,batch_first=True,padding_value=-80.0)#.to(device) 
    features = torch.transpose(features,1,2).contiguous().unsqueeze(dim=1) # (B,1,F,T)
        
    target_lengths = torch.tensor(S,dtype=torch.long) # ,device=device corresponding target lengths
    
    targets = torch.cat(y)#.to(device) # target sequences  # how to create directly in device?
          
    return features, frame_lengths, targets, target_lengths

def create_train_val_partition(df,split_ratio,N_batch): #,shuffle=True
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
    
    
    """    
    if shuffle: # Shuffle the dataset if required
        df = df.sample(frac=1).reset_index(drop=True) """
        
    
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



def create_test_partition(df,N_batch): #,shuffle=True
    """
    Creates a partition of the dataset into only test.
    
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
    
    
    """    
    if shuffle: # Shuffle the dataset if required
        df = df.sample(frac=1).reset_index(drop=True) """
    
    
    ID = df['path'].tolist()
    transcriptions = df['encoded'].tolist()    

    partition = dict()
    partition['test'] = ID #

    labels = dict()    
    # Save all the transcriptions into labels dictionary
    for i,val in enumerate(ID): 
        labels[val] = transcriptions[i]   
        
    return partition, labels




def argmax_decode(batch):
    """
    Argmax decode the whole output batch.
    Takes the exponent to cancel log effect.
        
        Parameters:
        -----------
        
            batch (Tensor) : number_of_frames*batch_size*alphabet_size
        
        Returns:
        --------
            
            argmax_decoded_batch (list) : len = batch_size.
    """
    
    batch_size = batch.shape[1]
    
    batch = torch.exp(batch)    
        
    argmax_decoded_batch = [torch.argmax(batch[:,i,:],dim=-1).tolist()  for i in range(batch_size)]
    
    return argmax_decoded_batch  


def beam_search_decoder(batch,k):
    
    batch_size = batch.shape[1]
    
    decoded_batch = []
    for i in range(batch_size):        
        data = batch[:,i,:]
    
        sequences = [[list(), 0.0]]
        for row in data:
            all_candidates = list()

            for i in range(len(sequences)):
                seq, score = sequences[i]

                for j in range(len(row)):
                    candidate = [seq+[j],score+row[j]]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)

            sequences = ordered[:k]
        
            
        decoded_batch.append(sequences[0][0]) # Keep the best path in the end for each instance
        
    return decoded_batch  

def code_to_str(code,alphabet):
    """
    Takes a coded tensor and turns it to a sentence using the alphabet in a string.
    alphabet must have has blank(0) at index zero
    
    each symbol is an idx or value ??
    """
    sentence = ''
    for symbol in code:
        sentence += alphabet[symbol]
        
    return sentence


"""alphabet = ['0',' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
 'p', 'r', 's', 't', 'u', 'v', 'y', 'z', 'ç', 'ö', 'ü', 'ğ', 'ı', 'ş']"""

def remove_local_repetitions(decoded_utterances,alphabet,remove_blank=True):
    """
    Removes the local repetitions for each example in an output batch and decodes it using code_to_str.
    Blank remains until I see logical outputs.
        
        Parameters:
        -----------
            
            decoded_utterances (list) : decoded utterances containing repetitions. Length batch_size
        
        Returns:
        --------
            
            outputs (list) : cleaned version of decoded utterances. (Blank is not removed for now!!!!)
    """
    
    outputs = []
    for decoded in decoded_utterances:             
               
        # Find all indices of all appearing characters
        char_idx_dict = dict()
        for idx,char in enumerate(decoded):
            
            if char in char_idx_dict:
                char_idx_dict[char].append(idx)
            else:
                char_idx_dict[char] = [idx]

        #Find consecutive repetitions (except final occurance)
        remove_indices = []
        for char_indices in char_idx_dict.values():

            for char_idx in char_indices:

                if char_idx+1 in char_indices:
                    remove_indices.append(char_idx)
                    
        # Remove consecutive repetitions
        cleaned_code = [i for j, i in enumerate(decoded) if j not in remove_indices]
        
        clean_utterance = code_to_str(cleaned_code,alphabet)
        
        if remove_blank:
            clean_utterance = clean_utterance.replace('0','')
        
        outputs.append(clean_utterance)
        
    return outputs


def targets_to_sentences(targets,target_lengths,alphabet):
    """
    Takes targets(1D) and target_lengths tensors and decodes them using the alphabet.
    Returns the sentences.
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