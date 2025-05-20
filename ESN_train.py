"""

Training Echo State Network (ESN) 
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

import warnings
warnings.filterwarnings('ignore')




#printing out updates
main_out = open('result.txt', 'w')
main_out.writelines(f'start\n')
main_out.flush()






#open train data
with open(f'../trajectory/train_trace_LV.pkl', 'rb') as file:
    train_data = pk.load(file)









#ESN model class
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
        Win = np.random.uniform(-self.w, self.w, (self.dim_input, self.N))
        return Win


    def generate_A(self):

        A = sparse.rand(self.N,self.N,density=self.degree/float(self.N)).todense()
        eigvals = np.linalg.eigvals(A)
        max_abs_eigval = np.max(np.abs(eigvals))
        A = (A/max_abs_eigval) * self.radius

        if (A == np.inf).sum() > 0:
            A = self.generate_A()

        return np.array(A)


    def warming_states(self, data):

        size = data.shape[0]

        state = np.tanh(data[0] @ self.Win)
        for i in range(1, size):
            state = np.tanh(state @ self.A + data[i] @ self.Win)

        self.warm_state = state 
    

    def training(self, data):

        train_size = data.shape[0] - 1
        train_data = data[:-1]
        
        states = np.zeros((train_size, self.N))

        states[0] = np.tanh(train_data[0] @ self.Win)
        for i in range(1,train_size):
            states[i] = np.tanh(states[i-1] @ self.A + train_data[i] @ self.Win)

        idenmat = (self.a) * np.identity(self.N)
        Uinv = np.linalg.inv(states.T @ states + idenmat)
        self.Wout = Uinv @ states.T  @ data[1:]


    def prediction(self, start_sample, predict_size):

        states = np.zeros((predict_size, self.N))
        test_predict = np.zeros((predict_size, self.dim_input))

        states[0] = np.tanh(self.warm_state @ self.A + start_sample @ self.Win)
        test_predict[0] = states[0] @ self.Wout

        for i in range(1, predict_size):

            states[i] = np.tanh(states[i-1] @ self.A + test_predict[i-1] @ self.Win)
            test_predict[i] = states[i] @ self.Wout 

        return test_predict









################Model parameters###################

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



np.random.seed(2)
model = ESN(parameters)

main_out.writelines(f"N {parameters['N']}, d {parameters['degree']}, r {parameters['radius']}, w {parameters['w']}, a {parameters['a']}\n")
main_out.flush()




###################Training###########################

training_size = 1000000
main_out.writelines(f'training {time.time() - start}\n')
main_out.flush()   

model.training(train_data[:training_size+1])





#save trained weights
with open(f'model_LV.pkl', 'wb') as file:
    pk.dump([model.Win, model.A, model.Wout], file)




#finish
main_out.writelines(f'end {time.time() - start}\n')
main_out.flush()                
main_out.close()
    
