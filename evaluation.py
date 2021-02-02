#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn.functional as F

import utilities.utilities as utils
import utilities.metrics as metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, criterion, decoder, test_loader):
    
	print('\nevaluating...')
	model.eval()

	with torch.no_grad():
		test_loss = 0
		test_cer, test_wer = [], []
		for spectrograms,input_lengths,labels,label_lengths in test_loader:

			spectrograms, labels = spectrograms.to(device), labels.to(device)

			output = model(spectrograms)  # (batch, time, n_class)
			output = F.log_softmax(output, dim=-1)       
			log_probs = torch.clone(output).cpu().numpy()           
			output = output.transpose(0, 1) # (time, batch, n_class)

			loss = criterion(output, labels, input_lengths, label_lengths)
			test_loss += loss.item()  

			sentences = utils.targets_to_sentences(labels, label_lengths, decoder.alphabet) # Map the target sequences to strings

			predicted_transcriptions = decoder.decode(log_probs)

			for j,pred in enumerate(predicted_transcriptions):
				test_cer.append(metrics.cer(sentences[j], pred))
				test_wer.append(metrics.wer(sentences[j], pred))

	
	print('---'*20)
	print('Target: {}'.format(sentences[0]))
	print('Predicted: {}\n'.format(predicted_transcriptions[0]))
	print('Target: {}'.format(sentences[2]))
	print('Predicted: {}\n'.format(predicted_transcriptions[2]))
	print('Target: {}'.format(sentences[4]))
	print('Predicted: {}'.format(predicted_transcriptions[4]))
	print('---'*20)
	

	avg_test_loss = test_loss/len(test_loader)
	avg_cer = sum(test_cer)/len(test_cer)
	avg_wer = sum(test_wer)/len(test_wer)


	print('Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(avg_test_loss, avg_cer, avg_wer)) 
	                                                          
	                                                                 
	return avg_test_loss, avg_cer, avg_wer




