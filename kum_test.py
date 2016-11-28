import numpy as np
import theano.tensor as T
import theano
from hteano import printing, function

# z and v = latent vars x batch size. First try latent vars x 1
# approach: calculate z(i) = v(i)*(1-z(i)) with z(1) = v(1)

v_th =T.fmatrix("v")
z1_th=T.fmatrix("z1")
updates_th=T.fvector("updates")

def stickbreak(v, z_prev, sl_used):
	z = v*(1-sl_used)
	new_sl_used = sl_used+z
	return z, new_sl_used


z_th, updates = theano.scan(fn=stickbreak,
                                  outputs_info=[z1_th, updates_th],
                                  sequences=[v_th]
                                  )

# Compile a function

calculate_z = theano.function(inputs=[v_th, z1_th, updates_th], outputs=z_th)

break_fractions = np.asarray([[0.5, 0.3, 1], [0.2, 0.5, 1]], dtype=np.float32).T
initial_stick_length = np.asarray([0,0], dtype=np.float32)
print break_fractions.shape, initial_stick_length.shape

z_th, updates = calculate_z(break_fractions, initial_stick_length, initial_stick_length)



print type(z_th)
print z_th.shape
print theano.shared(z_th).get_value()

# # Test

# # z1 = np.expand_dims(z1,0)


# sl_used = 0
# z = np.array([0., 0., 0.])
# for i in xrange(len(v)):
# 	z[i], sl_used = stickbreak(sl_used, v[i])
# print v,z