import gensim
import numpy as np
import cPickle as pickle
import scipy.io
from scipy.sparse import csc_matrix


def id2word_func(fname = 'data/KOS/vocabkos.txt'):
	result = {}
	for lineNo, line in enumerate(open(fname)):
		result[int(lineNo)] = line.strip()
	return result

def save_corpus(filename = 'data/KOS/docwordkos.txt', max_docs = 4000):
	f = open(filename)
	n_docs = int(f.readline())
	voc_size = int(f.readline())
	n_words = int(f.readline()) #total words
	doc_nr = 0	
	line_nr = 0
	docs = list()
	docs.append(list())
	for line in f:
		ws = line.split()
		if int(doc_nr)+1 == int(ws[0]) and ws[0] != max_docs:
			floats = map(float, ws[1:3])
			docs[doc_nr].append(tuple(map(float, ws[1:3])))
		elif int(doc_nr) == int(ws[0]) and ws[0] != max_docs:
			doc_nr = int(ws[0])-1
			docs.append(list())
			docs[doc_nr].append(tuple(map(float, ws[1:3])))
		elif ws[0] == max_docs:
			break
		else:
			print "something wrong with if else in save_corpus"

		line_nr += 1 

	np.save('KOS_for_gensim.npy', docs)

