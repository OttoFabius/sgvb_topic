import gzip
import cPickle as pickle
import numpy as np
import scipy.io
from scipy.sparse import csc_matrix, csr_matrix, vstack, lil_matrix, vstack
import time
import ConfigParser
import matplotlib.pyplot as plt
from random import shuffle
from scipy.linalg import orth
import itertools

def parse_config(fname):
    config = ConfigParser.ConfigParser()
    config.read("results/vae_own/" + fname + "/parameters.cfg")

    argdict = {}

    argdict['dimZ'] = config.getint('parameters','latent_variables')
    argdict['HUe1'] = config.getint('parameters','HUe1')
    argdict['HUe2'] = config.getint('parameters','HUe2')
    argdict['HUe3'] = config.getint('parameters','HUe3')
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
    argdict['full_vocab'] = config.getint('parameters', 'use_full_vocab')
    argdict['stickbreak'] = config.getint('parameters', 'stickbreak')
    argdict['normalize_input'] = config.getint('parameters', 'normalize_input')
    argdict['kld_weight'] = config.getfloat('parameters', 'kld_weight')
    argdict['ignore_logvar'] = config.getint('parameters', 'ignore_logvar')

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


def perplexity_rest(data_train, indices_used, data_test):

    means = csc_matrix.mean(csc_matrix(data_train), axis=0)

    mult_params = np.array(means/np.sum(means))[0,:]

    # mult_params = np.zeros_like(mult_params)+1./data_test.shape[1]
    perp=0
    data_test = csc_matrix(data_test)


    data_test[:,indices_used]=0

    for doc in xrange(data_test.shape[0]):
        perp_doc = np.sum(data_test[doc, :]*np.log(mult_params))
        perp+=perp_doc


    n_rest = csc_matrix.sum(data_test)

    return perp/csc_matrix.sum(data_test)

def select_half(data_sparse, seen_words=0.5):
    # check dimension looped over

    data = data_sparse.todense() 
    data_seen = np.zeros(data.shape)
    j=0
    for row in data:
        a = np.squeeze(np.array(row))
        b = list(itertools.chain.from_iterable([i]*e for i,e in enumerate(a.astype(int))))

        c = np.zeros(len(a), dtype = int)
        ind = np.random.choice(b,(len(b)+1)/2,False)
        a_new = np.zeros(len(a))
        
        for i in ind:
            a_new[i] +=1
        data_seen[j,:] = a_new
        j+=1
    data_unseen = data - data_seen

    return csc_matrix(data_seen), csc_matrix(data_unseen)

def load_dataset(argdict):
    dataset = argdict['dataset']
 
    print "loading dataset with minimum", argdict['minfreq'], 'word frequency'
    if argdict['trainset_size']>0:
        print 'trainset size', argdict['trainset_size']
        f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['minfreq'])+'_'+argdict['trainset_size']+'traindocs.pklz','rb')
    elif argdict['trainset_size']==0:
        print 'using all available train docs'
	f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['minfreq'])+'.pklz','rb')



	x = pickle.load(f)
	f.close()
	print "done"

	return x

def get_unused_sum(argdict):
    if argdict['full_vocab']==0:
        unused_sum = 0.
    elif argdict['full_vocab']==1:
        f = gzip.open('data/'+argdict['dataset']+'/docword_means.pklz','rb')
        means = pickle.load(f)
        f.close()
        f = gzip.open('data/'+argdict['dataset']+'/docword_' + str(argdict['minfreq']) + '_indices.pklz','rb')
        indices = pickle.load(f)
        unused_sum = 1 - np.sum(means[:,indices])/np.sum(means)

    return unused_sum
        

def load_used_features(argdict):
	
    dataset = argdict['dataset']
    if argdict['minfreq']>0:
        f = gzip.open('data/'+dataset+'/docword_' + str(argdict['minfreq']) + '_indices.pklz','rb')
        used_features = pickle.load(f)
        f.close()
    elif argdict['entselect']>0:
        f = gzip.open('data/'+dataset+'/docword_matrix_'+str(argdict['entselect'])+'_entselect_indices.pklz','rb')	
        used_features = pickle.load(f)
        f.close()
    else:
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

def convert_to_sparse(dataset='kos', n_docs_max=3430, min_per_doc=10):
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
    print 'filling lil matrix'
    docs 	= lil_matrix((n_docs_max, voc_size))
    for i, line in enumerate(f):
        ws = line.split()	
    	if int(ws[0])% 100 == 0:
    		print 'doc nr', int(ws[0])

    	if int(ws[0])==n_docs_max:
    		break
    	docs[int(ws[0])-1, int(ws[1])-1] = int(ws[2])

    print 'getting indices'
    row_indices = np.ndarray.flatten(np.array(np.nonzero(docs.sum(1)>min_per_doc)[0]))
    shuffle(row_indices)
    print 'converting to csr'
    docs = csr_matrix(docs)
    print "taking out selected rows"
    docs_pruned = docs[row_indices,:]
    print 'converting back to lil'

    docs_pruned_lil = lil_matrix(docs_pruned)
	
    print 'saving as ' + filename.strip('.txt')+'_matrix.pklz'
    f = gzip.open(filename.strip('.txt')+'_matrix.pklz','wb')
    pickle.dump(docs_pruned_lil, f)
    f.close()
    print 'done'

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

def select_subset(n_train, n_test=1000, dataset='ny', mincount=3000):
    start = time.time()
    print "loading pickled data"
    f = gzip.open('data/'+dataset+'/docword_matrix_' + str(mincount) + '.pklz','rb')
    data_orig = pickle.load(f)
    f.close()

    print "done"
    print "csr"
    data_orig = csr_matrix(data_orig)

    # data_orig = csc_matrix.transpose(data_orig)
    print "selecting docs"
    data_train = data_orig[:n_train,:]
    data_test = data_orig[data_orig.shape[0]-n_test:,:]
    data = lil_matrix(concatenate_csr_matrices_by_rows(data_train, data_test))

    print data_train.shape, data_test.shape

    print 'saving'
    f = gzip.open('data/'+dataset+'/docword_matrix_' + str(mincount) + '_' + str(n_train) + 'traindocs.pklz','wb')
    pickle.dump(data, f)
    f.close()

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

def create_rp(K=100, dataset = 'kos', mincount=50, orth=False):
    #memory error for orth = true on server
    #using something K = 1000 for ny as a baseline
    f = gzip.open('data/'+dataset+'/docword_rest_matrix_' + str(mincount) + '.pklz','rb')
    data_rest = pickle.load(f)
    f.close()
    
    data = csc_matrix(data_rest)

    N, D = data.shape
    print N, 'datapoints', 'and', D, 'dimensions'
    print 'creating R'
    R = np.random.normal(0, 1/np.sqrt(D), [D, K])
    print 'R shape is', R.shape
    if orth==True:
        print 'orthogonalize'
        R = orth(R)
        print 'check orth, max is ', np.max(np.dot(R.T, R))
    print 'projecting data'
    data_proj = data.dot(R)
    print 'data proj shape is', data_proj.shape, 'size in memory is', data_proj.nbytes/(1024*1024), 'MB'

    print "saving data"
    np.save('data/'+dataset+'/R_' + str(mincount)+'.npy', R)
    np.save('data/'+dataset+'/data_proj_' + str(mincount)+'.npy', data_proj)



    # f = gzip.open('data/'+dataset+'/R_' + str(mincount) + '.pklz','wb')
    # pickle.dump(R, f)
    # f.close()

    # f = gzip.open('data/'+dataset+'/data_proj_' + str(mincount) + '.pklz','wb')
    # pickle.dump(data_proj, f)
    # f.close()


def concatenate_csr_matrices_by_rows(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csr_matrix((new_data, new_indices, new_ind_ptr))