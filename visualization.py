import numpy as np
import theano as th
from loadsave import load_pickle_list, load_pickle_matrix, create_pickle_matrix, create_pickle_list # , load_pickle
from model_list import topic_model
from model_matrix import topic_model_matrix
import sys
import ConfigParser
import time
from scipy.sparse import csr_matrix
from train import parse_config
import matplotlib.pyplot as plt
from scipy.stats import mode

def get_indices(model, x, z_list, words_per_z = 50):

	y = model.generate_samples_2D(z_list)

	word_freqs = csr_matrix.sum(x, axis=0)
	rel_word_freqs = word_freqs/np.sum(word_freqs)


	vectors = []
	indices 	= np.zeros((len(z_list), words_per_z))
	indices_min = np.zeros((len(z_list), words_per_z))

	for i in xrange(len(z_list)):
		freq_increase = y[i]/rel_word_freqs
		vect = np.array(freq_increase)

		vectors.append(vect)

		for j in xrange(words_per_z):
			index     = np.argmax(vect)
			index_min = np.argmin(vect)
	
			vect[0,index] 	  = 1
			vect[0,index_min] = 1

			indices    [i,j] = index
			indices_min[i,j] = index_min

	return indices, indices_min

def lookup_words(indices, n_z, indices_min=None):

	words     = []
	words_min = []

	f = open("data/KOS/vocabkos.txt")
	lines=f.readlines()
	for i in xrange(len(z_list)):
		words_z     = []
		words_z_min = []

		j = 0
		for index in indices[i,:]:
			words_z.append(lines[int(index)])
			j += 1

		words.append(words_z)

		if indices_min:
			for index in indices_min[i,:]:
				words_z_min.append(lines[int(index)])
				j += 1
			words_min.append(words_z_min)
		else:
			words_min = "not computed"

	return words, words_min


if __name__ == "__main__":

	latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, learning_rate, sigmaInit, doc_per_doc, batch_size, only_trainset = parse_config()
	x,d,d_nrs = load_pickle_matrix(filename='_matrix.pklz') #no argument uses KOS dataset
	d_nrs = np.ndarray.flatten(d_nrs)
	voc_size = d.shape[1]    
	model = topic_model_matrix(voc_size, latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, learning_rate, sigmaInit, batch_size, only_trainset)
	model.load_parameters('results/' + sys.argv[1])

	z_list = [ 	np.array([3,3]), np.array([3,-3]), \
				np.array([-3,3]), np.array([-3,-3])]


	indices, indices_min = get_indices(model, x, z_list)
	words, words_min = lookup_words(indices, len(z_list))

	for i in xrange(len(z_list)):
		print z_list[i], words[i]
