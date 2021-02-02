from __future__ import division
from __future__ import print_function
import codecs
import re
import json
import numpy as np
from nltk import ngrams
import os

# idea: https://github.com/githubharald/CTCDecoder, completion raraz15

def read_txt(file_path,encoding='utf-8'):
    
	lst = []
	with open(file_path,'r',encoding=encoding) as fp:
		line = fp.readline()
		lst.append(line.strip("\n"))

		while line:

			line = fp.readline()
			lst.append(line.strip("\n"))

	return lst


class LanguageModel:
	"simple language model: word list for token passing, char bigrams for beam search"
	def __init__(self, fn, classes, lmFactor=0.1, unigram_path="", bigram_path="", trigram_path=""):
		"read text from file to generate language model"
		# classes should not include blank!!!
		self.initWordList(fn, classes)

		self.set_lmFactor(lmFactor)

		self.initCharUnigrams(fn, classes, unigram_path)
		self.init_unigram_occurance_matrix()
		self.init_unigram_prob_matrix()

		self.initCharBigrams(fn, classes, bigram_path)
		self.init_bigram_occurance_matrix()
		self.init_bigram_prob_matrix()

		self.initCharTrigrams(fn, classes, trigram_path)
		self.init_trigram_occurance_matrix()
		self.init_trigram_prob_matrix()


	def set_lmFactor(self,lmFactor):
		assert lmFactor > 0 and lmFactor <= 1, "Language Model Factor must be in (0,1]" 
		self.lmFactor = lmFactor


	def initCharUnigrams(self,fn,classes,unigram_path):
		"internal init of character unigrams"

		if os.path.exists(unigram_path):
			print("Reading unigram dict...")
			with open(unigram_path,'r', encoding='utf-8') as infile:
				self.unigram = json.load(infile)
		else:
			print("Forming unigram dict...")
			self.unigram = {c: 0 for c in classes}

			sentence_list = read_txt(fn,encoding='utf-8')

			for txt in sentence_list: # for each sentence in the text
				for c, in list(ngrams(txt,1)): # get all unigrams

					if c not in classes:
						continue
					else:
						self.unigram[c] += 1

			self.export_unigram(unigram_path)
			print("Unigram dict exported.")


	def init_unigram_occurance_matrix(self):

		self.unigram_occurance_matrix = self.unigram.values()


	def init_unigram_prob_matrix(self):

		total_characters = sum(self.unigram_occurance_matrix)

		self.unigram_prob_matrix = [c/total_characters for c in self.unigram_occurance_matrix]


	def getUnigramProb(self,c):

		return self.unigram_prob_matrix[c]


	def export_unigram(self,unigram_path):

		with open(unigram_path,'w', encoding='utf-8') as outfile:
			json.dump(self.unigram, outfile, ensure_ascii=False, indent=4)


	def initCharBigrams(self, fn, classes, bigram_path):
		"internal init of character bigrams"

		if os.path.exists(bigram_path):
			print("Reading bigram dict...")
			with open(bigram_path,'r', encoding='utf-8') as infile:
				self.bigram = json.load(infile)
		else:
			print("Forming bigram dict...")
			self.bigram = {c: {d: 0 for d in classes} for c in classes}

			sentence_list = read_txt(fn,encoding='utf-8')

			for txt in sentence_list:
				for c0,c1 in list(ngrams(txt,2)):

					if (c0 not in classes) or (c1 not in classes):
						continue
					else:
						self.bigram[c0][c1] += 1

			self.export_bigram(bigram_path)
			print("Bigram dict exported.")


	def init_bigram_occurance_matrix(self):

		bigram_occurance_matrix = []

		for key in self.bigram.keys():
			bigram_occurance_matrix.append(list(self.bigram[key].values()))

		self.bigram_occurance_matrix = bigram_occurance_matrix


	def init_bigram_prob_matrix(self):
		
		bigram_prob_matrix = []

		for row in self.bigram_occurance_matrix:

			numBigrams = sum(row)

			if numBigrams == 0:
				bigram_prob_matrix.append(row)
			else:
				bigram_prob_matrix.append([r/numBigrams for r in row])

		self.bigram_prob_matrix = bigram_prob_matrix 


	def getBigramProb(self, first, second):

		return self.bigram_prob_matrix[first][second]


	def export_bigram(self,bigram_path):

		with open(bigram_path,'w', encoding='utf-8') as outfile:
			json.dump(self.bigram, outfile, ensure_ascii=False, indent=4)


	def initCharTrigrams(self, fn, classes, trigram_path):
		"internal init of character trgirams"

		if os.path.exists(trigram_path):
			print("Reading trigram dict...")
			with open(trigram_path,'r', encoding='utf-8') as infile:
				self.trigram = json.load(infile)
		else:
			print("Forming trigram dict...")
			# init trigrams with 0 values
			self.trigram = {c: {d: {e: 0 for e in classes} for d in classes} for c in classes}

			sentence_list = read_txt(fn,encoding='utf-8')

			for txt in sentence_list:

				for c0,c1,c2 in list(ngrams(txt,3)):

					if (c0 not in classes) or (c1 not in classes) or (c2 not in classes):
						continue
					else:
						self.trigram[c0][c1][c2] += 1

			self.export_trigram(trigram_path)
			print("Unigram dict exported.")


	def init_trigram_occurance_matrix(self):

		trigram_occurance_matrix = []
		for c0 in self.trigram.keys():
			c1_list = [] 

			for c1 in self.trigram[c0].keys():		        
				c1_list.append(list(self.trigram[c0][c1].values())) # c2 values for that c1 for that c0

			trigram_occurance_matrix.append(c1_list)

		self.trigram_occurance_matrix = trigram_occurance_matrix


	def init_trigram_prob_matrix(self):

		trigram_prob_matrix = []
		for c0_c1c2 in self.trigram_occurance_matrix: # for all c0

			P1_list = []

			for c1_c2 in c0_c1c2: # for all c1 given c0

				numTrigrams = sum(c1_c2) # number of trigrams for c0c1

				if numTrigrams == 0:
					P1_list.append(c1_c2) # all zeros
				else:
					P1_list.append([c2/numTrigrams for c2 in c1_c2]) # c2 for c0c1

			trigram_prob_matrix.append(P1_list)

		self.trigram_probability_matrix = trigram_prob_matrix


	def getTrigramProb(self, first, second, third):

		return self.trigram_probability_matrix[first][second][third]


	def export_trigram(self,trigram_path):

		with open(trigram_path,'w', encoding='utf-8') as outfile:
			json.dump(self.trigram, outfile, ensure_ascii=False, indent=4)


	def initWordList(self, fn,classes):
		"internal init of word list"
		#with open(fn,'r', encoding='utf-8') as infile:
		#	txt = infile.read()
		sentence_list = read_txt(fn,encoding='utf-8')

		words_list = []
		for txt in sentence_list:
			words = re.findall(r'\w+', txt)
			for word in filter(lambda x: x.isalpha(), words):
				words_list.append(word)

		self.words = list(set(words_list))

	def getWordList(self):
		"get list of unique words"
		return self.words
