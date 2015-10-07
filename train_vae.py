import numpy as np
import theano as th
from loadsave import load_pickle_list, load_pickle_matrix, create_pickle_matrix, create_pickle_list # , load_pickle
import sys
import ConfigParser
import time
import scipy.sparse as sp
import VAE, optimizer, generate_params, blocks

if __name__=="__main__":

	def parse_config(foldername):
	    config = ConfigParser.ConfigParser()
	    config.read(foldername + "/parameters.cfg")

	    dim_h_en_z = [config.getint('parameters','dim_h_en_z_1')]
	    dim_h_de_x = [config.getint('parameters','dim_h_de_x_1')]
	    dim_z = config.getint('parameters','dim_z')
	    L = config.getint('parameters','L')   
	    iterations = config.getint('parameters','iterations')
	    learningRate = config.getfloat('parameters','learningRate')
	    polyak = config.getboolean('parameters','polyak')
	    batch_size = config.getint('parameters','batch_size')  

	    dim_h_en_z_2 = config.getint('parameters','dim_h_en_z_2')
	    if dim_h_en_z_2!=0:
	    	dim_h_en_z.append(dim_h_en_z_2)
	    dim_h_de_x_2 = config.getint('parameters','dim_h_de_x_2')
	    if dim_h_de_x_2!=0:
	    	dim_h_de_x.append(dim_h_de_x_2) 

	   	dim_h_en_z_3 = config.getint('parameters','dim_h_en_z_3')
	    if dim_h_en_z_3!=0:
	    	dim_h_en_z.append(dim_h_en_z_3)
	    dim_h_de_x_3 = config.getint('parameters','dim_h_de_x_3')
	    if dim_h_de_x_3!=0:
	    	dim_h_de_x.append(dim_h_de_x_3) 

	    return dim_h_en_z, dim_h_de_x, dim_z, L, iterations, learningRate,  polyak, batch_size

  #-------------------       		 parse config file       		--------------------

	foldername = "results/vae/" + sys.argv[1]
	dim_h_en_z, dim_h_de_x, dim_z , L, iterations , learningRate, polyak, batch_size = parse_config(foldername)
	normalization = 'l2'
	nonlinearity ='relu'
	type_rec = 'poisson'
	type_latent = 'gaussian'

	#-------------------      		 load dataset		       		--------------------

	x_all = np.load('data/NY/docwordny_matrix.npy')
	x = x_all[:3000,:]
	n, v = x.shape
	x_valid = x_all[3000:,:]

	name_log = foldername + '/log.txt'
	model = VAE.VAE(n, v, dim_h_en_z=dim_h_en_z, dim_h_de_x=dim_h_de_x, dim_z=dim_z, batch_size=batch_size,
                nonlinearity=nonlinearity, normalization=normalization, L=L,
                type_rec=type_rec, type_latent=type_latent, iterations=iterations, learningRate=learningRate, 
                polyak=polyak, name_log=name_log, seed=12345)
	model.fit(x, x_valid)

