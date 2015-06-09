import numpy as np
import scipy.io

import gzip
import cPickle as pickle

"""
	converts term frequency representation of documents with dimensions (V, n_docs) into matrix of 
	word representations and matching document representations with dimensions (voc, 1)
"""


mat = scipy.io.loadmat('data/dbworld/dbworld_bodies_stemmed.mat')

docs = mat['inputs']
words_in_doc = (np.sum(docs, axis = 1))

X = list()
d = list()

[ndocs, V] = docs.shape
for doc in xrange(ndocs):
	X_doc = np.zeros((words_in_doc[doc],V))
	counter = 0
	for word in xrange(V):
		n_of_word = docs[doc,word]
		for k in xrange(n_of_word):
			X_doc[counter+k, word] = 1
			counter += 1

	X.append(X_doc)
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