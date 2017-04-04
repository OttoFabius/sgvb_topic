from helpfuncs import *
import matplotlib.pyplot as plt

lb = np.array([-7.740858032, -7.581120946,-7.473220351, -7.37937281, -7.307962388, -7.27])
# lb_best = np.array([-7.601326273, -7.573787193,-7.471527762, -7.37937281, -7.303669425, -7.27])
size = np.array([5,10,20,50,100,300])*1000
plt.plot(size, lb, linewidth=2)
plt.plot(size, lb, 'bo', markersize=8)
# plt.plot(size, lb_best,'r')
# plt.plot(size, lb_best, 'r*')
plt.gca().set_xscale("log")
# plt.ylim((-9, -6))
plt.xlabel('Number of Documents')
plt.ylabel('Test Lowerbound')
plt.title('Lower Bound of Log Likelihood')
# plt.show()
# plt.savefig(title)
plt.close()



def plot_lb(name):

    argdict = parse_config(name)
    lb, lb_test, KLD, KLDtrain, recon_train, recon_test, perplexity, perp_sem, epoch = load_stats('results/vae_own/'+ name)
    plt.plot(lb)
    # name = 'kos/batchnorm/without'
    # argdict = parse_config(name)
    # lb, lb_test, KLD, KLDtrain, recon_train, recon_test, perplexity, perp_sem, epoch = load_stats('results/vae_own/'+ name)
    # plt.plot(lb)
    
plot_lb('kos/batchnorm/with')
plot_lb('kos/batchnorm/without')
plt.gca().set_xscale("log")
plt.legend(['With BN', 'Without BN'], loc=2)
plt.show()
plt.close()

plot_lb('ny/batch_norm/small')
plot_lb('ny/ho/400-200e-0d')
plt.gca().set_xscale("log")
plt.legend(['With BN', 'Without BN'], loc=2)
plt.show()



