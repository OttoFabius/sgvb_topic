import gzip
import cPickle as pickle
import numpy as np
import scipy.io
from scipy.sparse import csc_matrix, csr_matrix, vstack, lil_matrix
import time
import ConfigParser
import matplotlib.pyplot as plt
from random import shuffle
from scipy.linalg import orth
from vae_1l_rp import topic_model_1layer
from vae_2l_rp import topic_model_2layer
from vae_21l import topic_model_21layer
from vae_20l import topic_model_20layer
from vae_lin import topic_model_linear

def parse_config(fname):
    config = ConfigParser.ConfigParser()
    config.read("results/vae_own/" + fname + "/parameters.cfg")

    argdict = {}

    argdict['dimZ'] = config.getint('parameters','latent_variables')
    argdict['HUe1'] = config.getint('parameters','HUe1')
    argdict['HUe2'] = config.getint('parameters','HUe2')
    argdict['HUd1'] = config.getint('parameters','HUd1')
    argdict['HUd2'] = config.getint('parameters','HUd2')
    argdict['learning_rate'] = config.getfloat('parameters','learning_rate')
    argdict['sigmaInit'] = config.getfloat('parameters','sigmaInit')
    argdict['batch_size'] = config.getint('parameters','batch_size')
    argdict['trainset_size'] = config.getint('parameters','trainset_size')
    argdict['testset_size'] = config.getint('parameters','testset_size')
    argdict['dataset_num'] = config.getint('parameters','dataset')
    argdict['KLD_free'] = config.getfloat('parameters','KLD_free')
    argdict['KLD_burnin'] = config.getfloat('parameters','KLD_burnin')
    argdict['save_every'] = config.getint('parameters', 'save_every')
    argdict['samples'] = config.getint('parameters', 'samples_perplex')
    argdict['max_epochs'] = config.getint('parameters', 'max_epochs')
    argdict['rp'] = config.getint('parameters','rp')

    if argdict['dataset_num'] == 0:
        argdict['dataset']='kos'
    elif argdict['dataset_num'] == 1:
        argdict['dataset']='ny'
    argdict['minfreq'] = config.getint('parameters','minfreq')
    argdict['entselect'] = config.getint('parameters','entselect')

    return argdict

def load_stats(fname):

    lowerbound = np.load(fname + '/lowerbound.npy')
    testlowerbound = np.load(fname + '/lowerbound_test.npy')
    KLD =  np.load(fname + '/kld.npy')
    KLD_used = np.load(fname + '/kld_used.npy')
    recon_train = np.load(fname + '/recon_train.npy')
    recon_test = np.load(fname + '/recon_test.npy')
    perplexity = np.load(fname + '/perplexity.npy')
    perp_sem = np.load(fname + '/perp_sem.npy')
    epoch = lowerbound.shape[0]

    return lowerbound, testlowerbound, KLD, KLD_used, recon_train, recon_test, perplexity, perp_sem, epoch

def save_stats(fname, lowerbound, testlowerbound, KLD, KLD_used, recon_train, recon_test, perplexity, perp_sem):
    np.save(fname + '/lowerbound.npy', lowerbound)
    np.save(fname + '/lowerbound_test.npy', testlowerbound)
    np.save(fname + '/kld.npy', KLD)
    np.save(fname + '/kld_used.npy', KLD_used)
    np.save(fname + '/recon_train.npy', recon_train)
    np.save(fname + '/recon_test.npy', recon_test)
    np.save(fname + '/perplexity.npy', perplexity)
    np.save(fname + '/perp_sem.npy', perp_sem)

def perplexity_during_train(model, data, argdict, rest=None, selected_features=None):

    samples = argdict['samples']


    docnrs = np.arange(1, argdict['testset_size'], 1)

    log_perplexity_list = []
    for i in xrange(samples):
        log_perplexity = 0
        n_words=0
        for docnr in docnrs:
            doc = data[docnr,:]
            if type(rest)==np.ndarray:
                rest_doc = rest[docnr, :, np.newaxis]
            else:
                rest_doc=None
            log_perplexity_doc, n_words_doc = model.calculate_perplexity(doc.T, rest=rest_doc, selected_features=selected_features)
            log_perplexity += log_perplexity_doc
            n_words += n_words_doc

    	log_perplexity_list.append(-log_perplexity/n_words)
    perplexity = np.exp(np.array(log_perplexity_list))
    perp_mean = np.mean(perplexity)
    perp_sem = np.std(perplexity)/np.sqrt(samples)

    return perp_mean, perp_sem

def perplexity_rest(data_rest):

    means = csc_matrix.mean(data_rest, axis=0)

    mult_params = np.array(means/np.sum(means))[0,:]
    perp=0
    for doc in xrange(data_rest.shape[0]):

        perp_doc = np.sum(data_rest[doc, :]*np.log(mult_params))
        perp+=perp_doc
    return perp


def load_dataset(argdict):
	dataset = argdict['dataset']
	if argdict['dataset']=='kos': 
		if argdict['minfreq']>0:
			print "loading kos dataset with minimum", argdict['minfreq'], 'word frequency'
			f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['minfreq'])+'.pklz','rb')
		elif argdict['entselect']>0:
			f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['entselect'])+'_ent.pklz','rb')
			print "loading kos dataset with", argdict['entselect'], 'features selected on entropy'
		else:
			print 'loading kos dataset full vocabulary'
			f = gzip.open('data/'+dataset+'/docword_matrix.pklz','rb')

	elif argdict['dataset']=='ny':
		if argdict['minfreq']>0:
			print "loading NY dataset with minimum", argdict['minfreq'], 'word frequency'
			f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['minfreq'])+'.pklz','rb')
		elif argdict['entselect']>0:
			f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['entselect'])+'_ent.pklz','rb')
			print "loading NY dataset with", argdict['entselect'], 'features selected on entropy'
		else:
			print 'loading NY dataset full vocabulary'
			f = gzip.open('data/ny/docwordny_matrix.pklz','rb')

	x = pickle.load(f)
	f.close()
	print "done"

	return x

def load_used_features(argdict):
	
    dataset = argdict['dataset']
    if argdict['minfreq']>0:
        print "loading used features with minimum", argdict['minfreq'], 'word frequency'
        f = gzip.open('data/'+dataset+'/docword_' + str(argdict['minfreq']) + '_indices.pklz','rb')
        used_features = pickle.load(f)
        f.close()
    elif argdict['entselect']>0:
        print "loading used entropy features with", argdict['entselect'], 'features selected on entropy'
        f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['entselect'])+'_entselect_indices.pklz','rb')	
        used_features = pickle.load(f)
        f.close()
    else:
        print 'all features used'
        used_features = None
	
	print "done"
	return used_features


def convert_to_matrix(dataset='kos', n_docs_max=3000):

	"""converts text file to numpy matrix for function create_pickle_list.
	Created for KOS dataset.
	text file must only contain '.' for the extension and must be structured as follows:
	first line contains the number of documents
	second line the vocabulary size
	third line the total number of words (unused currently)

	each line thereafter contains {doc_id word_id word_freq}"""

	filename = 'data/'+dataset+'/docword.txt'
	f = open(filename)
	n_docs = int(f.readline())
	voc_size = int(f.readline())
	f.readline() #total words

	docs = np.zeros([n_docs_max, voc_size])

	for line in f:
		ws = line.split()

		if int(ws[0]) == n_docs_max-1: 
			print "max docs reached"
			break
		docs[int(ws[0])-1, int(ws[1])-1] = int(ws[2])

	np.save(str(filename).rsplit('.')[0] + '_matrix.npy', docs) 

def convert_to_sparse(dataset='kos', n_docs_max=3430, min_per_doc=20):
	"""converts text file to scipy sparse matrix
	Created for NY Times dataset.
	text file must only contain '.' for the extension and must be structured as follows:
	first line contains the number of documents
	second line the vocabulary size
	third line the total number of words (unused currently)

	each line thereafter contains {doc_id word_id word_freq}"""
	filename = 'data/'+dataset+'/docword.txt'
	f = open(filename)
	n_docs = int(f.readline())
	voc_size = int(f.readline())

	f.readline() #total words

	docs 	= lil_matrix((n_docs_max, voc_size))
	for i, line in enumerate(f):

		ws = line.split()	
		if int(ws[0])% 100 == 0:
			print 'doc nr', int(ws[0])

		if int(ws[0])==n_docs_max:
			break
		docs[int(ws[0])-1, int(ws[1])-1] = int(ws[2])

	print 'removing too small docs and shuffling the rest'
	row_indices = np.ndarray.flatten(np.array(np.nonzero(docs.sum(1)>min_per_doc)[0]))
	shuffle(row_indices)
	docs_pruned = csc_matrix(docs[row_indices,:])
	print 'done'

	docs_pruned_lil = lil_matrix(docs_pruned)
	
	print 'saving as ' + filename.strip('.txt')+'_matrix.pklz'
	f = gzip.open(filename.strip('.txt')+'_matrix.pklz','wb')
	pickle.dump(docs_pruned_lil, f)
	f.close()
	print 'done'

def initialize_model(argdict):
    print "initializing model + graph..."
    if argdict['HUe2']==0:
        model = topic_model_1layer(argdict)
    elif argdict['HUd2']!=0:
        model = topic_model_2layer(argdict)
    elif argdict['HUd1']!=0:
        model = topic_model_21layer(argdict)    
    elif argdict['HUd1']==0:
        model = topic_model_20layer(argdict)

    else:
        print 'no model selected :('
    return model

def select_features(mincount=0, dataset='kos'):
    start = time.time()
    print"loading pickled data"
    if dataset=='ny':
    	print "NY dataset"
    	f = gzip.open('data/'+dataset+'/docword_matrix.pklz','rb')
    elif dataset=='kos':
    	print "kos dataset"
    	f = gzip.open('data/'+dataset+'/docword_matrix.pklz','rb')
    data_orig = pickle.load(f)
    f.close()
    print "done"
    print 'csc'
    data_orig = csc_matrix(data_orig)
    print "getting indices"
    row_indices = np.ndarray.flatten(np.array(np.nonzero(data_orig.sum(0)>mincount)[1]))
    rest_indices = np.ndarray.flatten(np.array(np.nonzero(data_orig.sum(0)<=mincount)[1]))
    print "dp"
    data_pruned = lil_matrix(data_orig[:,row_indices])
    print "rest"
    data_rest = lil_matrix(data_orig[:,rest_indices])

    print 'saving, first used'
    f = gzip.open('data/'+dataset+'/docword_matrix_' + str(mincount) + '.pklz','wb')
    pickle.dump(data_pruned, f)
    f.close()
    print "done, now rest"
    f = gzip.open('data/'+dataset+'/docword_rest_matrix_' + str(mincount) + '.pklz','wb')
    pickle.dump(data_rest, f)
    f.close()
    print "done, now indices"
    f = gzip.open('data/'+dataset+'/docword_' + str(mincount) + '_used_indices.pklz','wb')
    pickle.dump(row_indices, f)
    f.close()

    f = gzip.open('data/'+dataset+'/docword_' + str(mincount) + '_rest_indices.pklz','wb')
    pickle.dump(rest_indices, f)
    f.close()


    print "done, new shape of used data = ", data_pruned.shape

def select_features_ent(n_features=1000, dataset='kos'):
	
	print"loading pickled data"
	if dataset=='ny':
		print "ny dataset"
		f = gzip.open('data/'+dataset+'/docword_matrix.pklz','rb')
	elif dataset=='kos':
		print "kos dataset"
		f = gzip.open('data/'+dataset+'/docword_matrix.pklz','rb')
	data = pickle.load(f)
	f.close()

	data_csc = csc_matrix(data)

	sum_per_word = data_csc.sum(0)
	n_total = data_csc.sum()

	p = sum_per_word/n_total
	doc_sizes = data_csc.sum(1)
	q = data_csc/doc_sizes

	neg_entropy = np.sum(np.multiply(q, np.log(p)), axis=0)
	indices = neg_entropy.argsort()[:,:n_features]
	indices = np.array(indices).squeeze()

	data_selected = data_csc[:,np.squeeze(indices)]

	if dataset=='ny':
		f = gzip.open('data/'+dataset+'/docword_' + str(n_features) + 'entselect.pklz','wb')
		g = gzip.open('data/'+dataset+'/docword_' + str(n_features) + 'entselect_indices.pklz','wb')
	elif dataset == 'kos':
		f = gzip.open('data/'+dataset+'/docword_' + str(n_features) + '_entselect.pklz','wb')
		g = gzip.open('data/'+dataset+'/docword_' + str(n_features) + '_entselect_indices.pklz','wb')

	pickle.dump(data_selected, f)
	pickle.dump(indices, g)
	f.close()
	g.close()

def save_parameters(model, path):
    """Saves parameters"""
    pickle.dump([name for name in model.params.keys()], open(path + "/names.pkl", "wb"))
    pickle.dump([p.get_value() for p in model.params.values()], open(path + "/params.pkl", "wb"))
    pickle.dump([m.get_value() for m in model.m.values()], open(path + "/m.pkl", "wb"))
    pickle.dump([v.get_value() for v in model.v.values()], open(path + "/v.pkl", "wb"))

def load_parameters(model, path):
    """Loads parameters, restarting training is possible afterwards"""
    names = pickle.load(open(path + "/names.pkl", "rb"))
    params = pickle.load(open(path + "/params.pkl", "rb"))

    for name,param in zip(names,params): 
        model.params[name].set_value(param)

    m_list = pickle.load(open(path + "/m.pkl", "rb"))
    v_list = pickle.load(open(path + "/v.pkl", "rb"))

    for name,m in zip(names,m_list): 
        model.m[name].set_value(m)
    for name,v in zip(names,v_list): 
        model.v[name].set_value(v)

def create_rp(K=100, dataset = 'kos', mincount=50):

    f = gzip.open('data/'+dataset+'/docword_rest_matrix_' + str(mincount) + '.pklz','rb')
    data_rest = pickle.load(f)
    f.close()
    
    data = csc_matrix(data_rest)

    N, D = data.shape
    print N, 'datapoints', 'and', D, 'dimensions'
    print 'creating R'
    R = np.random.normal(0, 1/np.sqrt(D), [D, K])
    print 'orthogonalize'
    R_o = orth(R)
    print 'check orth, max is ', np.max(np.dot(R_o.T, R_o))
    print 'projecting data'
    data_proj = data.dot(R)

    print "saving data"
    f = gzip.open('data/'+dataset+'/R_' + str(mincount) + '.pklz','wb')
    pickle.dump(R_o, f)
    f.close()

    f = gzip.open('data/'+dataset+'/data_proj_' + str(mincount) + '.pklz','wb')
    pickle.dump(data_proj, f)
    f.close()





