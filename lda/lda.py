import logging, gensim, bz2
from gensim.corpora.dictionary import Dictionary
from lda_preprocess import id2word_func, parse_config
import logging	
import sys	

def donothing():
	print "a"

if __name__ == "__main__":


	dataset, trainset_size, testset_size, passes, n_topics, eval_every = parse_config(sys.argv[1])

	if dataset == 0:
		dataset = 'KOS'
		print dataset
	# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	mm_train = gensim.corpora.MmCorpus('/home/otto/Documents/thesis/sgvb_topic/data/'+dataset+'/corpus_train_'+str(trainset_size)+'.mm') #data/KOS/corpus
	mm_test =  gensim.corpora.MmCorpus('/home/otto/Documents/thesis/sgvb_topic/data/'+dataset+'/corpus_test_'+str(testset_size)+'.mm')

	id2word = id2word_func(fname='/home/otto/Documents/thesis/sgvb_topic/data/'+dataset+'/vocab'+dataset+'.txt')

	print 'training model with', passes, ' passes and ', n_topics, ' topics...'
	lda = gensim.models.ldamodel.LdaModel(corpus=mm_train, id2word=id2word, num_topics=n_topics, passes=0)
	lda.save('/home/otto/Documents/thesis/sgvb_topic/results/lda/'+sys.argv[1]+'/saved_model')
	print "saved successfully"



	log_perplex_train = []
	log_perplex_test = []

	log_perplex = lda.log_perplexity(mm_test, total_docs=None)

	passes_done = 0
	while passes_done <= passes:
		lda.update(mm_train, passes=eval_every, eval_every=None)

		print "calculating perplexity on train set after", passes_done, 'passes:'
		log_perplex = lda.log_perplexity(mm_train, total_docs=None)
		print log_perplex
		log_perplex_train.append(log_perplex)
		
		print "calculating perplexity on test set:"
		log_perplex = lda.log_perplexity(mm_test, total_docs=None)
		print log_perplex
		log_perplex_test.append(log_perplex)

		gam = lda.inference(mm_test, collect_sstats=False)
		print gam
		raw_input()
		passes_done+=eval_every
		lda.save('/home/otto/Documents/thesis/sgvb_topic/results/lda/'+sys.argv[1]+'/saved_model')
		print "saved successfully"


	print "done. Now calc log perplex, gensim way..."
	# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	log_perplex = lda.log_perplexity(mm_test, total_docs=None)
	print "log perplex = ", log_perplex


		