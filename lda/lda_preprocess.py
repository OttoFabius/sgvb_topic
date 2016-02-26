import gensim
import numpy as np
import cPickle as pickle
import scipy.io
from scipy.sparse import csc_matrix
import ConfigParser
import sys


def id2word_func(fname = '/home/otto/Documents/thesis/sgvb_topic/data/KOS/vocabkos.txt'):
	result = {}
	for lineNo, line in enumerate(open(fname)):
		result[int(lineNo)] = line.strip()
	return result

def save_corpus(filename = '/home/otto/Documents/thesis/sgvb_topic/data/KOS/docwordKOS.txt', train_size=3000, test_size=430, dataset='KOS'):
	f = open(filename)
	n_docs = int(f.readline())
	voc_size = int(f.readline())
	n_words = int(f.readline()) #total words
	doc_nr = 0	
	line_nr = 0
	n = train_size+test_size
	docs = list()
	docs.append(list())
	for line in f:
		wds = line.split()

		if int(doc_nr)+1 == int(wds[0]):
			docs[doc_nr].append(tuple(map(float, wds[1:3])))
		elif int(doc_nr)+2 == int(wds[0]): #next document
			doc_nr = int(wds[0])-1
			docs.append(list())
			docs[doc_nr].append(tuple(map(float, wds[1:3])))
		else:
			print "something wrong with if else in save_corpus"
			# raw_input()

		line_nr += 1 
	np.random.shuffle(docs)


	gensim.corpora.MmCorpus.serialize('/home/otto/Documents/thesis/sgvb_topic/data/'+dataset+'/corpus_train_'+ str(train_size) + '.mm', docs[:train_size])
	gensim.corpora.MmCorpus.serialize('/home/otto/Documents/thesis/sgvb_topic/data/'+dataset+'/corpus_test_' + str(test_size)  + '.mm', docs[train_size:train_size+test_size])
	print 'done preprocessing'

