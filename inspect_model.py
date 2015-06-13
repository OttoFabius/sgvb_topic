from model_list import model	
from loadsave import load_pickle_list
import numpy as np
import matplotlib.pyplot as plt

_ , d = load_pickle_list()
voc_size = d[1].size

topic_model = model(voc_size)
print 'loading parameters'
model.load_parameters(topic_model, 'results')

for key, value in topic_model.params.items():
	print key, ':'
	print 'mean', np.mean(value.get_value())
	print 'sd', np.std(value.get_value())
	plt.hist(np.ndarray.flatten(value.get_value()),10)
	plt.show()
	raw_input()
	plt.close()

