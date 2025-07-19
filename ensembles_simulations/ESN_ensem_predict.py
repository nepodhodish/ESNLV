"""

Testing Echo State Network (ESN) 
on Lotka-Voltera (LV) equation

@author: Anton Erofeev
"""




import time
start = time.time()

import numpy as np
import pandas as pd
import pickle as pk

import scipy.sparse as sparse
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

import copy
import warnings
warnings.filterwarnings('ignore')




"""
we make a file "result.txt" to keep track on what stage our document is
"""
main_out = open('result.txt', 'w')
main_out.writelines(f'start\n')
main_out.flush()


"""
mean - point from attractor, mean for uniform distribution
r - variables' range 
v - variance for uniform distribution
"""
mean = np.array([0.30, 0.45, 0.13, 0.36])
r = np.array([0.43, 0.53, 0.36, 0.39])
v = r * 0.3


"""
open test data
"""
with open(f'test_traj_LV.pkl', 'rb') as file:
    test_traj = pk.load(file)


"""
open trained weights 
"""
with open('../model_LV.pkl', 'rb') as file:
    weights = pk.load(file)






"""
ESN model class
"""
class ESN:
    def __init__(self, parameters):
        self.N = parameters['N']
        self.degree = parameters['degree']
        self.radius = parameters['radius']
        self.dim_input = parameters['dim_input']
        self.w = parameters['w']
        self.a = parameters['a']
        
        self.Win = self.generate_win()
        self.A = self.generate_A()
        self.Wout = np.zeros((self.N,self.dim_input))
    

    def generate_win(self):
        """
        Win matrix is generated from uniform distribution (-w, w) of size dim x N
        """
        Win = np.random.uniform(-self.w, self.w, (self.dim_input, self.N))
        return Win


    def generate_A(self):
        """
        generate matrix A, which acts inside reservoir of size N x N
        """
        A = sparse.rand(self.N,self.N,density=self.degree/float(self.N)).todense()
        eigvals = np.linalg.eigvals(A)
        max_abs_eigval = np.max(np.abs(eigvals))
        A = (A/max_abs_eigval) * self.radius

        """
        sometimes eigenvalues are very small, so infinity values appear
        in this case we need to regenerate matrix
        """
        if (A == np.inf).sum() > 0:
            A = self.generate_A()

        return np.array(A)


    def warming_states(self, data):
        """
        when we generate data, we need to skip first values, until trajectory gets to attractor
        """
        size = data.shape[0]

        state = np.tanh(data[0] @ self.Win)
        for i in range(1, size):
            state = np.tanh(state @ self.A + data[i] @ self.Win)

        self.warm_state = state 
    

    def training(self, data):
        """
        training Wout matrix by excplicit formula
        """
        train_size = data.shape[0] - 1
        train_data = data[:-1]
        
        states = np.zeros((train_size, self.N))

        states[0] = np.tanh(train_data[0] @ self.Win)
        for i in range(1,train_size):
            states[i] = np.tanh(states[i-1] @ self.A + train_data[i] @ self.Win)

        idenmat = (self.a) * np.identity(self.N)
        Uinv = np.linalg.inv(states.T @ states + idenmat)
        self.Wout = Uinv @ states.T  @ data[1:]

    def prediction(self, predict_size, mean, v):
        """
        esn generates new trajectory using trained Wout matrix
        """
        states = np.zeros((predict_size, self.N))
        predict_data = np.zeros((predict_size, self.dim_input))

        high = mean + v
        low = mean - v
        low[low < 0.05] = 0.05

        predict_data[0] = np.random.uniform(low, high, self.dim_input) 

        states[0] = np.tanh(predict_data[0] @ self.Win)
        predict_data[1] = states[0] @ self.Wout

        for i in range(2, predict_size):

            states[i-1] = np.tanh(states[i-2] @ self.A + predict_data[i-1] @ self.Win)
            predict_data[i] = states[i-1] @ self.Wout 

        return predict_data









"""
Model parameters
"""
parameters = {'N': 200,   #reservoir dimension
            'dim_input': 4,   #total dim of the data
            'degree': 180,   #num connections per neuron
            'radius': 0.5,  #specral radius
            'w': 5,  #range for Win = unif[-w,w]
            'a': 0.001, #regularization
            'm1': np.array([0.30130301, 0.4586546 , 0.13076546, 0.35574161]), #m1,2,3,4 LV statistics
            'm2': np.array([0.00613064, 0.01567845, 0.00802094, 0.00660596]),
            'm3': np.array([ 5.97923646e-04, -8.20942794e-05,  4.08964845e-04, -1.78523141e-05]),
            'm4': np.array([0.00018159, 0.00064422, 0.00017017, 0.0001261 ]),
            }


"""
Upload trained model weights
"""
np.random.seed(2)


"""
Predicting
"""

main_out.writelines(f'predicting {time.time() - start}\n')
main_out.flush()  



"""
load weights into model
"""
model = ESN(parameters)
model.Win, model.A, model.Wout = weights


"""
prepare array for predicted trajectories
and perform prediction
"""
predict_traj = np.zeros((test_traj.shape[0], test_traj.shape[1], test_traj.shape[2]))

for traj in range(test_traj.shape[0]):

    predict_traj[traj] = model.prediction(predict_traj.shape[1], mean, v)



"""
take max from every trajectory in ensemble
"""
predict_max = np.max(predict_traj, axis=1),
test_max = np.max(test_traj, axis=1),



main_out.writelines(f'prediction done {time.time() - start}\n')
main_out.flush()   




"""
save data
"""
with open(f'predict_traj_LV.pkl', 'wb') as file:
    pk.dump(predict_traj, file)

with open(f'predict_max.pkl', 'wb') as file:
    pk.dump(predict_max, file)

with open(f'test_max.pkl', 'wb') as file:
    pk.dump(test_max, file)







"""
plot LV and ESN max histograms + estimate GEV coefficient
"""

pdf = PdfPages("LV_ESN_max.pdf")


lv = test_max.T
esn = predict_max.T
    
fig = plt.figure(figsize=(10, 12), layout='constrained')
gs = GridSpec(4, 2, figure=fig)
axx = []
axx.append(fig.add_subplot(gs[0,0]))
axx.append(fig.add_subplot(gs[1,0]))
axx.append(fig.add_subplot(gs[2,0]))
axx.append(fig.add_subplot(gs[3,0]))
axx.append(fig.add_subplot(gs[0,1]))
axx.append(fig.add_subplot(gs[1,1]))
axx.append(fig.add_subplot(gs[2,1]))
axx.append(fig.add_subplot(gs[3,1]))


cntr = 0

"""
fit LV values
"""
for ii in range(parameters['dim_input']):
    
    ax = axx[cntr]
    
    vari = lv[ii]

    c, loc, scale = stats.genextreme.fit(vari)
    c *= -1
    
    x = np.linspace(np.min(vari), np.max(vari), 200)
    
    ax.hist(vari, bins=x, density=True)
    ax.set_title(f"LV c={np.round(c, 3)}")
    ax.set_xlabel(f"X{ii}")
    ax.set_ylabel("Density")
    ax.grid(True)
    
    cntr += 1


"""
fit ESN values
"""
for ii in range(parameters['dim_input']):
    
    ax = axx[cntr]
    
    vari = esn[ii]
      
    c, loc, scale = stats.genextreme.fit(vari)
    c *= -1
    
    x = np.linspace(np.min(vari), np.max(vari), 200)
    
    ax.hist(vari, bins=x, density=True)
    ax.set_title(f"ESN c={np.round(c, 3)}")
    ax.set_xlabel(f"X{ii}")
    ax.set_ylabel("Density")
    ax.grid(True)

    cntr += 1


pdf.savefig(fig)



pdf.close()





"""
finish
"""
main_out.writelines(f'end {time.time() - start}\n')
main_out.flush()                
main_out.close()
    
