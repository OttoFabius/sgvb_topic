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

def load_pickle_matrix(filename = "_matrix.pklz"):
	"""assumes files for words (x) and documents)d) start with x and d and are identical further"""

	print "loading X...."
	f = gzip.open('x' + filename,'rb')
	x = pickle.load(f)
	f.close()
	print "done, now d..."

	f = gzip.open('d' + filename,'rb')
	d = pickle.load(f)
	f.close()
	print "done, now d_num"

	f = gzip.open('d_nrs' + filename,'rb')
	d_num = pickle.load(f)
	f.close()
	print "done"

	return x, d, d_num

def create_pickle_matrix(filename = 'data/KOS/docwordkos_matrix.npy', traintest=True, n_train=2000, n_test=1430):
	"""
		converts term frequency representation of documents with dimensions (V, n_docs) into matrix of 
		shuffled word representations and matching document representations
	"""
	docs = np.load(filename)
	[n_docs, V] = docs.shape

	if traintest:
		n_docs = n_train
	words_in_doc = (np.sum(docs[:n_docs], axis = 1))

	total_words = 0

	X 		= lil_matrix((int(np.sum(words_in_doc)), V))
	d 		= lil_matrix((int(n_docs)			   , V))
	doc_nrs = np.zeros 	((int(np.sum(words_in_doc)), 1))

	for doc in xrange(n_docs):
		counter = 0
		X_doc = np.zeros((words_in_doc[doc],V))
		doc_nrs_doc = np.ones((words_in_doc[doc],1))*(doc+1)

		d_doc = docs[doc,:]

		for word in xrange(V):
			n_of_word = int(docs[doc,word])

			for k in xrange(n_of_word):
				X_doc[counter, word] = 1
				counter += 1

		X 		[int(total_words):int(total_words+words_in_doc[doc]), :	] = X_doc
		d 		[int(total_words):int(total_words+words_in_doc[doc]), : ] = d_doc
		doc_nrs [int(total_words):int(total_words+words_in_doc[doc])	] = doc_nrs_doc
		total_words += words_in_doc[doc]
		print 'next doc: doc', doc
		print 'done'

	# shuffle
	print "shuffling"

	order = np.arange(X.shape[0])
	np.random.shuffle(order)
	X 		= X 	 [order,:]
	doc_nrs = doc_nrs[order  ]

	print "converting X to csr:"

	X = X.tocsr()
	print " now d"
	d = d.tocsr()

	print"pickling X...."
	if traintest:
		f = gzip.open('x_train_matrix.pklz','wb')
	else:
		f = gzip.open('x_matrix.pklz','wb')
	pickle.dump(X, f)
	f.close()
	print "done, now d"
	if traintest:
		f = gzip.open('d_train_matrix.pklz','wb')
	else:
		f = gzip.open('d_matrix.pklz','wb')
	pickle.dump(d, f)
	f.close()

	print "and the doc nrs"
	if traintest:
		f = gzip.open('d_nrs_train_matrix.pklz','wb')
	else:
		f = gzip.open('d_nrs_matrix.pklz','wb')
	pickle.dump(doc_nrs, f)
	f.close()
	print "done"

	################################		 create test set 		################################

	if traintest:
		print "now test set"
		total_words = 0
		words_in_doc = (np.sum(docs[n_train:n_train+n_test], axis = 1))	

		X 		= lil_matrix((int(np.sum(words_in_doc)), V))
		d 		= lil_matrix((int(n_test)			   , V))
		doc_nrs = np.zeros 	((int(np.sum(words_in_doc)), 1))	

		for doc in xrange(n_train, n_test):
			counter = 0

			X_doc = np.zeros((words_in_doc[doc-n_train],V))
			doc_nrs_doc = np.ones((words_in_doc[doc-n_train],1))*(doc+1)

			d_doc = docs[doc,:]

			for word in xrange(V):
				n_of_word = int(docs[doc,word])

				for k in xrange(n_of_word):
					X_doc[counter, word] = 1
					counter += 1

			X 		[int(total_words):int(total_words+words_in_doc[doc-n_train]), :	] = X_doc
			d 		[int(total_words):int(total_words+words_in_doc[doc-n_train]), : ] = d_doc
			doc_nrs [int(total_words):int(total_words+words_in_doc[doc-n_train])	] = doc_nrs_doc
			total_words += words_in_doc[doc]
			print 'next doc: doc', doc-n_train
			print 'done'

		# shuffle
		print "shuffling"
		order = np.arange(X.shape[0])
		np.random.shuffle(order)
		X 		= X 	 [order,:]
		doc_nrs = doc_nrs[order  ]

		print "converting X to csr:"
		X = X.tocsr()
		print " now d"
		d = d.tocsr()
				
		print"pickling X...."
		f = gzip.open('x_test_matrix.pklz','wb')
		pickle.dump(X, f)
		f.close()

		print "done, now d"
		f = gzip.open('d_test_matrix.pklz','wb')
		pickle.dump(d, f)
		f.close()

		print "and the doc nrs"
		f = gzip.open('d_nrs_test_matrix.pklz','wb')
		pickle.dump(doc_nrs, f)
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

	[n_docs, V] = docs.shape
	for doc in xrange(n_docs):
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

	f.readline() #total words

	docs = np.zeros([n_docs, voc_size])
	for line in f:
		ws = line.split()
		docs[int(ws[0])-1, int(ws[1])-1] = int(ws[2])
	np.save(str(filename).rsplit('.')[0] + '_matrix.npy', docs) 

def convert_to_sparse(filename = 'data/NY/docwordny.txt', verbose=False):
	"""converts text file to scipy sparse matrix
	Created for NY Times dataset.
	text file must only contain '.' for the extension and must be structured as follows:
	first line contains the number of documents
	second line the vocabulary size
	third line the total number of words (unused currently)

	each line thereafter contains {doc_id word_id word_freq}"""

	f = open(filename)
	n_docs = int(f.readline())
	voc_size = int(f.readline())

	f.readline() #total words

	docs 	= lil_matrix((n_docs, voc_size))
	i = 0
	for line in f:

		ws = line.split()
		docs[int(ws[0])-1, int(ws[1])-1] = int(ws[2])
		if verbose==True:

			if i % 1000 == 0:
				print i
			i+=1
			
	f = gzip.open('data/NY/docwordny_matrix.pklz','wb')
	pickle.dump(doc_nrs, f)
	f.close()