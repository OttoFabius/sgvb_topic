import logging, gensim, bz2
from gensim.corpora.dictionary import Dictionary
from lda_preprocess import id2word_func

mm = gensim.corpora.MmCorpus('data/KOS/corpus.mm')
id2word = id2word_func()
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=5, update_every=0, passes=5)
print "now calc log perplex"
log_perplex = lda.log_perplexity(mm, total_docs=None)
print log_perplex
print "done"
