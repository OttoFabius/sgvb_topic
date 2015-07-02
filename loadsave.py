import gzip
import cPickle as pickle
import numpy as np
import scipy.io
from scipy.sparse import csc_matrix, csr_matrix, vstack, lil_matrix
from sklearn.utils import resample


def load_pickle_list(filename = "_train_list.pklz"):
	"""assumes files for words (x) and documents)d) start with x and d and are identical further"""

	print "loading X...."
	f = gzip.open('x' + filename,'rb')
	x = pickle.load(f)
	f.close()
	print "done, now d..."

	f = gzip.open('d' + filename,'rb')
	d = pickle.load(f)
	f.close()
	print "done"

	return x, d

def load_pickle_matrix(filename = "_train_matrix.pklz"):
	"""assumes files for words (x) and documents)d) start with x and d and are identical further"""

	print "loading X...."
	f = gzip.open('x' + filename,'rb')
	x = pickle.load(f)
	f.close()
	print "done, now d..."

	f = gzip.open('d' + filename,'rb')
	d = pickle.load(f)
	f.close()
	print "done"

	return x, d

def create_pickle_matrix(filename = 'data/KOS/docwordkos_matrix.npy'):
	"""
		converts term frequency representation of documents with dimensions (V, n_docs) into matrix of 
		shuffled word representations and matching document representations
		It takes around 15-30 mins all in all and needs 8gb ram (or more) for KOS
	"""
	print 'pickling'
	docs = np.load(filename)

	words_in_doc = (np.sum(docs, axis = 1))
	[ndocs, V] = docs.shape

	X = 	lil_matrix((int(np.sum(words_in_doc)), V))
	d = 	lil_matrix((int(np.sum(words_in_doc)), V))

	total_words = 0

	for doc in xrange(ndocs):
		X_doc = np.zeros((words_in_doc[doc],V))
		counter = 0

		d_doc = np.tile(docs[doc,:], (words_in_doc[doc],1))

		for word in xrange(V):
			n_of_word = int(docs[doc,word])
			for k in xrange(n_of_word):
				X_doc[counter, word] = 1
				counter += 1


		X[int(total_words):int(total_words+words_in_doc[doc]), :] = X_doc
		d[int(total_words):int(total_words+words_in_doc[doc]), :] = d_doc
		total_words += words_in_doc[doc]
		print 'next doc: doc', doc
		print 'done'

	# shuffle
	print "converting to csr, first X:"

	X = X.tocsr()
	print " now d"
	d = d.tocsr()

	print "shuffling"
	X, d = resample(X, d, random_state=0, replace=False)
	# order = np.arange(X.shape[0])
	# np.random.shuffle(order)
	# X_shuf = X[order,:]
	# d_shuf = dorder,:]

	print "converting to csc"

	X = csc_matrix(X)
	d = csc_matrix(d)


		# X.append(csc_matrix(X_doc))
		# d.append(np.expand_dims(docs[doc,:], axis=1))
		# print 'next doc: doc ', doc

			
	print"pickling X...."
	f = gzip.open('x_train_matrix.pklz','wb')
	pickle.dump(X, f)
	f.close()
	print "done, now d"

	f = gzip.open('d_train_matrix.pklz','wb')
	pickle.dump(d, f)
	f.close()

def create_pickle_list(filename = 'data/KOS/docwordkos_matrix.npy', dtype = 'npy', lemmatize=True, stem=False, sw_removal=True):
	"""Creates a list of arrays of docs and matching sparse one-of-k word representations for an sgvb topic model.
		works for .mat and.npy"""
	if dtype == 'npy':
		docs = np.load(filename)
	elif dtype == 'mat':
		mat = scipy.io.loadmat(filename)
		docs = mat['inputs']

	words_in_doc = (np.sum(docs, axis = 1))

	X = list()
	d = list()

	[ndocs, V] = docs.shape
	for doc in xrange(ndocs):
		X_doc = np.zeros((words_in_doc[doc],V))
		counter = 0
		for word in xrange(V):

			n_of_word = int(docs[doc,word])
			for k in xrange(n_of_word):
				X_doc[counter, word] = 1
				counter += 1

		X.append(csc_matrix(X_doc))
		d.append(np.expand_dims(docs[doc,:], axis=1))
			

	print"pickling X...."
	f = gzip.open('x_train_list.pklz','wb')
	pickle.dump(X, f)
	f.close()
	print "done, now d"
	f = gzip.open('d_train_list.pklz','wb')
	pickle.dump(d, f)
	f.close()
	print "done"

def convert_to_matrix(filename = 'data/KOS/docwordkos.txt'):
	"""converts text file to numpy matrix for function create_pickle_list.
	Created for KOS dataset.
	text file must only contain '.' for the extension and must be structured as follows:
	first line contains the number of documents
	second line the vocabulary size
	third line the total number of words (unused currently)

	each line thereafter contains {doc_id word_id word_freq}"""

	f = open(filename)
	n_docs = int(f.readline())
	voc_size = int(f.readline())
	f.readline() #total words

	docs = np.zeros([n_docs, voc_size])
	for line in f:
		ws = line.split()
		docs[int(ws[0])-1, int(ws[1])-1] = int(ws[2])
	np.save(str(filename).rsplit('.')[0] + '_matrix.npy', docs) 