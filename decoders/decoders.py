#!/usr/bin/env python
# coding: utf-8


import os

from numpy import argmax as numpy_argmax
from torch import Tensor

from utilities.utilities import code_to_str, ctc_collapse_function
import decoders.BeamSearch as BeamSearch
import decoders.LanguageModel as LanguageModel
import decoders.BKTree as BKTree
import decoders.LexiconSearch as LexiconSearch


class Argmax_decoder:
	"""
	Argmax decoder with ctc collapse function and conversion to string.
	"""

	def __init__(self, alphabet, blank_idx=0):
		self.alphabet = alphabet
		self.blank_idx = blank_idx


	def decode(self, batch):
		"""
		Argmax decode the complete output batch or a single output, apply the collapse function and convert to string.
			
			Parameters:
			-----------
			
				batch (ndarray) : (batch_size*)number_of_frames*alphabet_size
			
			Returns:
			--------
				
				argmax_decoded_batch (list) : len = batch_size.
		"""

		if len(batch.shape) == 3: # if it is a batch

			predicted_transcriptions = []
			for single in batch:

				decoded = numpy_argmax(single, axis=-1).tolist()

				collapsed_code = ctc_collapse_function(decoded, self.blank_idx)

				predicted_transcriptions.append(code_to_str(collapsed_code, self.alphabet))


			return predicted_transcriptions	

		else: # or a single output

			decoded = numpy_argmax(batch, axis=-1).tolist()

			collapsed_code = ctc_collapse_function(decoded, self.blank_idx)

			return code_to_str(collapsed_code, self.alphabet)


class BeamSearch_decoder:
	"""
	Class for beam Search decoder with possible character level language model integration.
	"""

	def __init__(self, alphabet, blank_idx, BW, prune, LM_text_name=""):
		"""
		Initializes the BeamSearch decoder object..
			
			Parameters:
			-----------
			
				alphabet (list): The alphabet that is used for coding the utterances. Blank symbol must be included.
				blank_idx (int): index of the blank symbol in the alphabet.
				BW (int) : Beam Width of the search.
				prune (float) : pruning threshold for increasing speed.
				LM_text_name (string) : the name of the text file that contains the corpus. (just the name no file type or path)
		"""


		self.alphabet = alphabet
		self.blank_idx = blank_idx
		self.BW = BW

		assert prune < 0, "Utilizing log probability, threshold must be negative."

		self.prune =prune
		self.LM = self.char_level_LM(LM_text_name) if LM_text_name else None


	def char_level_LM(self, LM_text_name):
		"""
		Initializes the LM by assuming ngram dicts and the corpus share the same name and uploads uploads.
		For more information, check the LanguageModel.py
			
			Parameters:
			-----------
			
				LM_text_name (string) : the name of the text file that contains the corpus. (just the name no file type or path)
			
		"""		

		# Provide uni,bi and trigram dictionaries if possible, otherwise they will be exported once created
		unigram_path = os.path.join("data","language model data",LM_text_name+"-unigram_dict.json")
		bigram_path = os.path.join("data","language model data",LM_text_name+"-bigram_dict.json")
		trigram_path = os.path.join("data","language model data",LM_text_name+"-trigram_dict.json")


		classes = self.alphabet.copy() # there is no blank character in real texts
		del classes[self.blank_idx]

		lmFactor = 0.7 # can be adjusted anytime

		LM_text_path = os.path.join("data","language model data",LM_text_name+".txt")
		LM = LanguageModel.LanguageModel(LM_text_path, classes, lmFactor, unigram_path, bigram_path, trigram_path)

		return LM


	def decode(self,log_probs): 
		"""
		Decodes the log probability batch or a single output with the beam search algorithm.
			
			Parameters:
			-----------
			
				log_probs (Tensor): Log probabilities tensor of shape (batch_size*number_of_frames*alphabet_size)
		"""
		
		if len(log_probs.shape) == 3: # if a batch
			return [BeamSearch.ctcBeamSearch(log_prob, self.alphabet, self.blank_idx, self.LM, self.BW, self.prune) for log_prob in log_probs]
		else:  # if a single output
			return BeamSearch.ctcBeamSearch(log_probs, self.alphabet, self.blank_idx, self.LM, self.BW, self.prune)

	def set_BW(BW):
		"""
		Sets the beam width of the search.
		"""
		self.BW = BW


	def set_prune(prune):
		"""
		Sets the pruning threshold of the search.
		"""
		self.prune = prune


class LexiconSearch_decoder:
	"""
	Class for Lexicon Search decoding algorithm object.
	"""

	def __init__(self, alphabet, tolerance, LM_text_name, approximator_properties=('argmax')):
		"""
		Initializes the Lexicon Search decoder object.
			
			Parameters:
			-----------
			
				alphabet (list): The alphabet that is used for coding the utterances. Blank symbol must be included.
				tolerance (int): Positive integer for searching words through the BKTree that are close.
				LM_text_name (string) : the name of the text file that contains the corpus. (just the name no file type or path)
		"""

		self.alphabet = alphabet
		self.init_lexicon(LM_text_name)
		self.bktree = BKTree.BKTree(self.words) # construct the BKTree from a chosen words list

		assert tolerance > 0, "Tolerance must be a positive integer."
		self.tolerance = tolerance
		self.approximator = self.construct_approximator(approximator_properties)


	def init_lexicon(self, LM_text_name):
		"""
		Reads the corpus and extracts the words list for Lexicon Search.
		"""
		path = os.path.join("data","language model data",LM_text_name+".txt")
		with open(path,encoding="utf-8") as f:
			words = f.read().split() # read each line and seperate from spaces

		self.words = words


	def construct_approximator(self, approximator_properties):
		"""
		You can work with other approximators but you have to construct them. 
		"""

		if approximator_properties[0] == "BeamSearch+LM":

			approx_obj = BeamSearch_decoder(self.alphabet, *approximator_properties[1:])
		else:
			approx_obj = Argmax_decoder(self.alphabet, 0)

		approximator = (lambda data: approx_obj.decode(data)) 

		return approximator


	def decode(self, log_probs):
		"""
		Performs the Lexicon Searcg for the log_probs with given decoder approximator.
			Parameters:
			-----------

				log_probs (Tensor): Log probabilities tensor of shape (batch_size*number_of_frames*alphabet_size)
				approximator : a decoder approximator for initially decoding the log_probs. For more information go to LexiconSearch.py

		"""

		return [LexiconSearch.ctcLexiconSearch(log_probs[j,:,:], self.alphabet, self.approximator,self.bktree,self.tolerance) for j in range(log_probs.shape[0])]


	def set_tolerance(self, tolerance):
		"""
		Sets the tolerance level for the search.
		"""

		self.tolerance = tolerance