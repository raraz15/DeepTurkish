from __future__ import division
from __future__ import print_function
import numpy as np
import math

# reference: https://github.com/githubharald/CTCDecoder


def multiply_probs(ln_pa, ln_pb):
	"""
	Returns the log probability of multiplied probabilities.
	Pc = Pa * Pb => ln(Pc) = ln(Pa)+ln(Pb)
	"""
	return ln_pa+ln_pb


def sum_probs(ln_pa,ln_pb):
	"""
	Returns the log prob. of two added probs.
	Pc = Pa + Pb => ln(Pc) = ln(Pa) + ln(1+exp(ln(Pb)-ln(Pa))), by Graves dissertation eq 7.18
	"""
	#print(ln_pa,ln_pb,ln_pa + math.log(1+math.exp(ln_pb-ln_pa)))
	return ln_pa + math.log(1+math.exp(ln_pb-ln_pa))


class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal = -100#-math.inf # blank and non-blank
		self.prNonBlank = -100#-math.inf # non-blank
		self.prBlank = -100#-math.inf # blank
		self.prText = 0 # LM score
		self.lmApplied = False # flag if LM was already applied to this beam
		self.labeling = () # beam-labeling


class BeamState:
	"information about the beams at specific time-step"
	def __init__(self):
		self.entries = {}

	def norm(self):
		"length-normalise LM score"
		for (k, _) in self.entries.items():
			labelingLen = len(self.entries[k].labeling)
			#self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))
			self.entries[k].prText *= (1.0 / (labelingLen if labelingLen else 1.0)) 

	def sort(self):
		"return beam-labelings, sorted by probability"
		beams = [v for (_, v) in self.entries.items()]

		sortedBeams = sorted(beams, reverse=True, key=lambda x: multiply_probs(x.prTotal, x.prText)) 

		return [x.labeling for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
	"calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
	if lm and not childBeam.lmApplied:

		parentBeam_len = len(parentBeam.labeling) # check the length of parent labelling to decide which model to use

		if  parentBeam_len == 0: # if there is no lebeling yet, use unigram
			c1 = childBeam.labeling[-1]-1

			unigramProb = lm.getUnigramProb(c1)

			if unigramProb:

				unigramProb_weighted = lm.lmFactor * math.log(unigramProb)
				childBeam.prText = multiply_probs(parentBeam.prText, unigramProb_weighted)

			else:

				childBeam.prText = -100#-math.inf

			childBeam.lmApplied = True

		elif parentBeam_len == 1: # if there is just a single charater, use bigram 

			c1 = parentBeam.labeling[-1]-1
			c2 = childBeam.labeling[-1]-1 # second char comes from childbeam

			bigramProb = lm.getBigramProb(c1, c2) # probability

			if bigramProb:

				bigramProb_weighted = lm.lmFactor * math.log(bigramProb)
				childBeam.prText = multiply_probs(parentBeam.prText, bigramProb_weighted)

			else:
				# no smoothing on the LM, if bigram pron 0, logProb = -inf
				childBeam.prText = -100#-math.inf

			childBeam.lmApplied = True # only apply LM once per beam entry

		else: #parentBeam_len == 2: # if moe than two characters are transcribed, use trigram.

			c0, c1 = parentBeam.labeling[-2]-1, parentBeam.labeling[-1]-1
			c2 = childBeam.labeling[-1]-1

			trigramProb = lm.getTrigramProb(c0, c1, c2)

			if trigramProb:

				trigramProb_weighted = lm.lmFactor * math.log(trigramProb)
				childBeam.prText = multiply_probs(parentBeam.prText, trigramProb_weighted)

			else:
				childBeam.prText = -100#-math.inf

			childBeam.lmApplied = True


def addBeam(beamState, labeling):
	"add beam if it does not yet exist"
	if labeling not in beamState.entries:
		beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, blankIdx, lm, beamWidth=25, prune=math.log(0.001)):
	""" 
	Everything has been converted to log probabilities
	beam search as described by the paper of Hwang et al. and the paper of Graves et al. 
	"""

	maxT, maxC = mat.shape

	# indices of non-blank characters in the matrix
	non_blank_indices= list(range(maxC))
	non_blank_indices.remove(blankIdx)

	# initialise beam state
	last = BeamState()
	labeling = ()
	last.entries[labeling] = BeamEntry()
	last.entries[labeling].prBlank = 0
	last.entries[labeling].prTotal = 0

	# go over all time-steps
	for t in range(maxT):
		curr = BeamState()

		# get beam-labelings of best beams
		bestLabelings = last.sort()[0:beamWidth]

		# go over best beams
		for labeling in bestLabelings:

			# probability of paths ending with a non-blank
			prNonBlank = -100 #-math.inf
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				prNonBlank = multiply_probs(last.entries[labeling].prNonBlank, mat[t, labeling[-1]])

			# probability of paths ending with a blank
			prBlank = multiply_probs(last.entries[labeling].prTotal, mat[t, blankIdx])

			# add beam at current time-step if needed
			addBeam(curr, labeling)

			# fill in data
			curr.entries[labeling].labeling = labeling
			curr.entries[labeling].prNonBlank = sum_probs(prNonBlank,curr.entries[labeling].prNonBlank)
			curr.entries[labeling].prBlank = sum_probs(prBlank, curr.entries[labeling].prBlank)
			curr.entries[labeling].prTotal = sum_probs(sum_probs(prBlank, prNonBlank), curr.entries[labeling].prTotal)
			curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
			curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

			# Only check the non-blank probabilities are checked here 
			# prune low probability characters
			pruned_c = np.where(mat[t][1:] >= prune)[0]+1 # add 1 to match the indices of the ctc matrix

			# extend current beam-labeling
			for c in pruned_c: # c in [1,31]
			
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# if new labeling contains duplicate char at the end, only consider paths ending with a blank
				if labeling and labeling[-1] == c:
					prNonBlank = multiply_probs(mat[t, c], last.entries[labeling].prBlank)
				else:
					prNonBlank = multiply_probs(mat[t, c], last.entries[labeling].prTotal)

				# add beam at current time-step if needed
				addBeam(curr, newLabeling)
				
				# fill in data
				curr.entries[newLabeling].labeling = newLabeling
				curr.entries[newLabeling].prNonBlank = sum_probs(prNonBlank, curr.entries[newLabeling].prNonBlank)
				curr.entries[newLabeling].prTotal = sum_probs(prNonBlank, curr.entries[newLabeling].prTotal)
				
				# apply LM
				applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

		# set new beam state
		last = curr

	# normalise LM scores according to beam-labeling-length
	last.norm()

	 # sort by probability
	bestLabeling = last.sort()[0] # get most probable labeling

	# map labels to chars
	prev = ''
	res = ''
	for l in bestLabeling:
	
		if l != blankIdx: # remove blanks

			if l == 2 and prev == 2: # because we don't do collapsing we deal with the "."s here. ('.'=2)
				break
			else:
				res += classes[l]

			prev = l

	return res