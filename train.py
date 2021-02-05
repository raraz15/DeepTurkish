#!/usr/bin/env python
# coding: utf-8

import datetime as dt

import torch
import torch.nn.functional as F

from model.data_loader import make_loaders
from model.neural_network import make_model
from evaluation import test
import utilities.utilities as utils

import wandb
wandb.login()

torch.manual_seed(27) 
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(hyperparameters, data_parameters, decoder, project_name):
   
    with wandb.init(project=project_name, config=hyperparameters):
        
        model_name = dt.datetime.strftime(dt.datetime.now(),"%H_%M__%d_%m_%Y")
        print("Model Name: {}\n".format(model_name))       
        wandb.run.name = model_name
        
        config = wandb.config        
        iter_meter = utils.IterMeter(project_name, model_name, config['epochs'])
                        
        if hyperparameters['SortaGrad']:
            
            # Sorted, not shuffled loaders
            train_loader, validation_loader, _ = make_loaders(data_parameters, sortagrad=True)
            
            model, criterion, optimizer, scheduler = make_model(config, 0, int(len(train_loader)), device)
            print(model)
            print('Num Model Parameters\n', sum([param.nelement() for param in model.parameters()]))

            wandb.watch(model, criterion, log="all", log_freq=1)
            
            # First epoch is ordered and not shuffled
            single_epoch(hyperparameters, data_parameters, model, criterion, 
                 optimizer, scheduler, decoder, train_loader, validation_loader, iter_meter)
            
            # for the rest of the epochs, use shuffled dataset
            train_loader, validation_loader, test_loader = make_loaders(data_parameters, sortagrad=False)
                                    
            for _ in range(config['epochs']-1):
            
                single_epoch(hyperparameters, data_parameters, model, criterion, 
                     optimizer, scheduler, decoder, train_loader, validation_loader, iter_meter)
                       
        else:
            train_loader, validation_loader, test_loader = make_loaders(data_parameters, sortagrad=False)
            
            model, criterion, optimizer, scheduler = make_model(config, 0, int(len(train_loader)), device)
            print(model)
            print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

            wandb.watch(model, criterion, log="all", log_freq=10)
               
            for _ in range(config['epochs']):
                
                single_epoch(hyperparameters, data_parameters, model, criterion, 
                     optimizer, scheduler, decoder, train_loader, validation_loader, iter_meter)
        
        avg_test_loss, avg_cer, avg_wer = test(model, criterion, decoder, test_loader) # Test on the test set 

        wandb.log({'test_avg_loss':avg_test_loss,'test_avg_cer':avg_cer, 'test_avg_wer':avg_wer, 'epoch':iter_meter.get_epoch()})


def single_epoch(hyperparameters, data_parameters, model, criterion, 
             optimizer, scheduler, decoder, train_loader, validation_loader, iter_meter):
    
    iter_meter.step_epoch()

    train(model, criterion, optimizer, scheduler, train_loader, iter_meter)  

    avg_validation_loss, _, _ = test(model, criterion, decoder, validation_loader)

    state = {'hyperparameters': hyperparameters,
            'data_parameters': data_parameters,
            'epoch': iter_meter.get_epoch(),
            'iteration': iter_meter.get(),
            'model_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'avg_validation_loss':avg_validation_loss # metric to compare
    } 

    utils.save_checkpoint(state, 'avg_validation_loss', iter_meter)


def train(model, criterion, optimizer, scheduler, loader, iter_meter):
    
    model.train()    
    total_batches = len(loader) # number of batches in an epoch 

    for spectrograms,input_lengths,labels,label_lengths in loader:
        
        spectrograms,labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step() 
        scheduler.step()
        iter_meter.step()
        
        current_epoch = iter_meter.get_epoch()
        current_iteration = iter_meter.get() 
        batch_idx = current_iteration % total_batches
                
        if current_iteration % 10 == 0:
            train_log(loss, current_iteration*loader.batch_size, current_epoch)
        
        if current_iteration % 100 == 0 or batch_idx == 0 or batch_idx == 1:            
            print('[Training Epoch: {}/{}] Iteration: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
                current_epoch, iter_meter.get_total_epochs(),
                current_iteration, total_batches*iter_meter.get_total_epochs(),
                100. * current_iteration/(total_batches*iter_meter.get_total_epochs()), loss.item()))
                    

def train_log(loss, example_ct, epoch):
    loss = float(loss)

    wandb.log({"Training Loss": loss}, step=example_ct)


