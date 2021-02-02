import decoders.Loss

# reference https://github.com/githubharald/CTCDecoder, advancement : raraz15

def ctcLexiconSearch(mat, classes, approximator, bkTree, tolerance):
	"compute approximation with best path decoding, search most similar words in dictionary, calculate score for each of them, return best scoring one. See Shi, Bai and Yao."

	approx = approximator(mat)

	approx_words = approx.split(" ") # extract each word

	query_words = [] #list containing query words
	for approx_word in approx_words:

		# get similar words from dictionary within given tolerance
		words = bkTree.query(approx_word, tolerance)

		# if there are no matches or the word itself is returned, do not search
		if not words or approx_word in words:
			query_words.append(approx_word)

		else:
			# else compute probabilities of all similar words and return best scoring one
			wordProbs = [(w, Loss.ctcLabelingProb(mat, w, classes,0)) for w in words]

			wordProbs.sort(key=lambda x: x[1], reverse=False) 

			query_words.append(wordProbs[0][0])

	return " ".join(query_words)
