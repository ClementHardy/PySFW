# import recupDescriptionDonnees
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.dictionary import expo,derivExpo,cauchy,derivCauchy
from sliding_frank_wolfe.SFW_algorithm import SFW
from sliding_frank_wolfe.regularization_parameter import  cross_validation_lambda
from sliding_frank_wolfe.regularization_parameter import  find_lambda
# import pip
# pip.main(['install','numba'])

import numpy as np
import matplotlib.pyplot as plt


###############################RANDOM DATA#####################################

#####GENERATE RANDOM SIGNALS###################################################
nb_pts = 1000 #number of points in the grid
nb_spl = 100 #number of signals
k = 10 #number of parametric functions
noise_level = 0.01
normalized = True
times = np.linspace(0,20,nb_pts) #grid over which the signals are discretized
# c = np.random.uniform(0.1,1,k) #scaling parameter
# b = np.linspace(times[0]+5,times[-1]-5,k) #location parameter
# np.random.shuffle(b)
# np.random.shuffle(c)
# parameters = np.zeros((k,2))  #array containing the parameters of the parametric functions
# parameters[:,0] = b
# parameters[:,1] = c
# A = np.abs(np.random.uniform(0,10,size=(nb_spl,k))) #linear coefficients
# F = build_Phi(times, parameters, k, 0, normalized,expo) #build a matrix containing all the parametric functions
# data_noiseless = np.dot(F,np.transpose(A)) #signals without noise
# data = data_noiseless + np.random.normal(0,noise_level,size = data_noiseless.shape) #noisy signals
# np.save('parameters',parameters)
# np.save('data',data)
# np.save('data_noiseless', data_noiseless)
parameters = np.load('parameters.npy')
data = np.load('data.npy')
data_noiseless = np.load('data_noiseless.npy')


######PLOT RANDOM SIGNALS##########################################################
plt.figure()
plt.plot(times,data)
plt.show()


#####PARAMETERS OF THE SFW ALGORITHM###############################################
low_b = np.minimum(times[0],times[-1])
up_b = np.maximum(times[0],times[-1])
low_c = 0.01
up_c = 10
upper_bounds = [up_b,up_c] #lower bounds on the parameter
lower_bounds = [low_b,low_c] #upper bounds on the parameter
size_grids  = [10,10]
func = expo
deriv_func = derivExpo
reg = noise_level *  1. / (nb_spl * np.sqrt(nb_pts ))
print('reg:', reg)


# res_optim = SFW(data, times, reg, lower_bounds, upper_bounds, func, deriv_func, threshold=threshold ,
#                  merging_threshold= merging_threshold, rank="full", size_grids = None, normalized= True,
#                  epsilon = 1e-4, max_iter = max_iter,step_mesh = 1e-1)
res_optim = SFW(data, times, reg, lower_bounds, upper_bounds, func, deriv_func)
Aopt = res_optim.linear_coefficients
parameters_opt = res_optim.dictionary_parameters
coeff  = res_optim.history_norms_linear_parameters
ite = res_optim.iterations
print('the matrix of linear coefficients if os full rank:', res_optim.is_full_rank )
print('the rank is:',res_optim.rank)
print('the sparsity is:', res_optim.sparsity )

kopt = parameters_opt.shape[0]
Fopt = build_Phi(times, parameters_opt, kopt, 0, normalized = True,func=func)
dataopt = np.dot(Fopt,np.transpose(Aopt))
#print('norm : ' , np.linalg.norm(Fopt,axis = 0, ord = 2))

plt.figure(3)
plt.plot(times,dataopt,color='b')
plt.show()

plt.figure(4)
plt.plot(times,data_noiseless,color='r')
plt.plot(times,dataopt,color='b')
plt.show()

plt.figure(5)
plt.scatter(parameters_opt[:,0],parameters_opt[:,1])
plt.show()
#iterations
plt.figure(6)
plt.step(np.arange(1,len(ite) +  1),ite)
plt.show()

size_partition = nb_spl // 2
range_lbda =  np.logspace(-6, -2, 3)
res_lbda = find_lambda(data, times, range_lbda, lower_bounds, upper_bounds, func, deriv_func)[1]

plt.plot(range_lbda, res_lbda)
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.show()
