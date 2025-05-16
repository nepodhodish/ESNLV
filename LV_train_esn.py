"""

Training and evaluating Echo State Network (ESN) 
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






#open training data
with open(f'../trajectory/trace_LV_2_e7.pkl', 'rb') as file:
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










#training and predicting parameters
warm = 100
training_size = 1000000
predict_size = 10000000









pdf = PdfPages("Esn_LV.pdf")

for N in [200]:
    for degree in [180, 90, 40]:
    


        #Model parameters###################

        parameters = {'N': N,   #reservoir dimension
                    'dim_input': 4,   #total dim of the data
                    'degree': degree,   #num connections per neuron
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

        main_out.writelines(f'N {parameters['N']}, d {parameters['degree']}, r {parameters['radius']}, w {parameters['w']}, a {parameters['a']}\n')
        main_out.flush()

        ###################Training###########################

        main_out.writelines(f'training {time.time() - start}\n')
        main_out.flush()   
        history = []
        model.training(train_data[:training_size+1])
        

        ######################Predicting########################
        pred_batch = 400

        main_out.writelines(f'predicting {time.time() - start}\n')
        main_out.flush()  


        
        model.warming_states(train_data[-warm-pred_batch-1:-pred_batch-1])
        predict_data = model.prediction(train_data[-pred_batch-1], predict_size)


        '''
        with open(f'model_LV.pkl', 'wb') as file:
            pk.dump([model.Win, model.A, model.Wout], file)
        with open(f'predict_data.pkl', 'wb') as file:
            pk.dump(predict_data, file)
        '''

        

        
        
        ####################Results printing####################

        
        
        #statistics of predicted histograms
        tm1 = np.round(stats.moment(predict_data, moment=1, center=0),5)
        tm2 = np.round(stats.moment(predict_data, moment=2),5)
        tm3 = np.round(stats.moment(predict_data, moment=3),5)
        tm4 = np.round(stats.moment(predict_data, moment=4),5)

        description = f'''Train: X Y Z W
        m1: {np.round(parameters['m1'],5)}
        m2: {np.round(parameters['m2'],5)}
        m3: {np.round(parameters['m3'],5)}
        m4: {np.round(parameters['m4'],5)}

        Predicted: X Y Z W
        m1: {tm1}
        m2: {tm2}
        m3: {tm3}
        m4: {tm4}

        Model parameters:
        N: {parameters['N']}
        degree: {parameters['degree']}
        radius: {parameters['radius']}
        w: {parameters['w']}
        a: {parameters['a']}

        Training parameters:
        warm: {warm}
        train_size: {training_size}
        predict_size: {predict_size}
        '''
                

        
        






        #####Check, whether predicted trajectory is similar to original in the beginning#####
        #####And check, whether predicted trajectory blow up in the end#####
        #####for all dimensions#####

        fig = plt.figure(figsize=(12, 16), layout='constrained')
        fig.legend(title=description, bbox_to_anchor=(0.4, 0.45))
        gs = GridSpec(8, 2, figure=fig)
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[2,1])
        ax5 = fig.add_subplot(gs[3,1])
        ax6 = fig.add_subplot(gs[4,1])
        ax7 = fig.add_subplot(gs[5,1])
        ax8 = fig.add_subplot(gs[6,1])
        ax9 = fig.add_subplot(gs[7,1])

        ts = np.arange(pred_batch)

        ax2.plot(ts, predict_data[:pred_batch, 0], lw=0.5)
        ax2.plot(ts, train_data[-pred_batch:, 0], lw=0.5)
        ax2.set_xlabel(f'First {pred_batch} steps')
        ax2.set_ylabel('X')
        ax2.grid(True)

        ax3.plot(ts, predict_data[:pred_batch, 1], lw=0.5)
        ax3.plot(ts, train_data[-pred_batch:, 1], lw=0.5)
        ax3.set_xlabel(f'First {pred_batch} steps')
        ax3.set_ylabel('Y')
        ax3.grid(True)

        ax4.plot(ts, predict_data[:pred_batch, 2], lw=0.5)
        ax4.plot(ts, train_data[-pred_batch:, 2], lw=0.5)
        ax4.set_xlabel(f'First {pred_batch} steps')
        ax4.set_ylabel('Z')
        ax4.grid(True)

        ax5.plot(ts, predict_data[:pred_batch, 3], lw=0.5)
        ax5.plot(ts, train_data[-pred_batch:, 3], lw=0.5)
        ax5.set_xlabel(f'First {pred_batch} steps')
        ax5.set_ylabel('W')
        ax5.grid(True)

        ax6.plot(ts, predict_data[-pred_batch:, 0], lw=0.5)
        ax6.set_xlabel(f'Last {pred_batch} steps')
        ax6.set_ylabel('X')
        ax6.grid(True)

        ax7.plot(ts, predict_data[-pred_batch:, 1], lw=0.5)
        ax7.set_xlabel(f'Last {pred_batch} steps')
        ax7.set_ylabel('Y')
        ax7.grid(True)

        ax8.plot(ts, predict_data[-pred_batch:, 2], lw=0.5)
        ax8.set_xlabel(f'Last {pred_batch} steps')
        ax8.set_ylabel('Z')
        ax8.grid(True)

        ax9.plot(ts, predict_data[-pred_batch:, 3], lw=0.5)
        ax9.set_xlabel(f'Last {pred_batch} steps')
        ax9.set_ylabel('W')
        ax9.grid(True)

        pdf.savefig(fig)






        ########Plotting histograms of predicted and original trajectories################
        ########for all dimensions


        fig = plt.figure(figsize=(16, 16), layout='constrained')
        gs = GridSpec(4, 4, figure=fig)
        y_lim = [[6, 0.6], [1.3, 1.2], [60, 2.5], [1.3, 1.3]]
        x_dmain = [0.01, 0.015, 0.005, 0.005]
        x_dtail = [[-0.005, 0.02], [-0.01, 0.01], [-0.0007, 0.01], [-0.01, 0.01]]
        consts = {
                    'left_q': 0.02,
                    'right_q': 0.98,
                }

        for i in range(train_data.shape[1]):
            ax1 = fig.add_subplot(gs[i,0])
            ax2 = fig.add_subplot(gs[i,1])
            ax3 = fig.add_subplot(gs[i,2])
            ax4 = fig.add_subplot(gs[i,3])

            var_p = predict_data[:,i]
            var_t = train_data[:,i]
            bins = np.linspace(var_t.min() - x_dmain[i], var_t.max() + x_dmain[i], 200)
            ax1.hist(var_p, bins=bins, alpha=0.8, density=True, label='Esn')
            ax1.hist(var_t, bins=bins, alpha=0.5, density=True, label='LV')
            ax1.set_xlabel(f'Variable {i}')
            ax1.set_ylabel('pdf')
            ax1.legend()

            ax2.hist(var_p, bins=bins, alpha=0.8, density=True, label='Esn', log=True)
            ax2.hist(var_t, bins=bins, alpha=0.5, density=True, label='LV', log=True)
            ax2.set_xlabel(f'Variable {i}')
            ax2.set_ylabel('pdf')
            ax2.legend()


            t_min = var_t.min() + x_dtail[i][0]
            t_max = var_t.max() + x_dtail[i][1]


            quant = np.quantile(var_t,consts['left_q'])
            l_t = f'LV'
            l_p = f'Esn'
            ax3.hist(var_p, bins=np.linspace(t_min,t_max,2000), alpha=0.8, density=True, log=True, label=l_p)
            ax3.hist(var_t, bins=np.linspace(t_min,t_max,2000), alpha=0.5, density=True, log=True, label=l_t)
            ax3.set_xlim(t_min,quant)
            ax3.set_ylim(0, y_lim[i][0])
            ax3.set_xlabel(f'Left var {i} tail')
            ax3.legend()

            quant = np.quantile(var_t,consts['right_q'])
            l_t = f'LV'
            l_p = f'Esn'
            ax4.hist(var_p, bins=np.linspace(t_min,t_max,2000), alpha=0.8, density=True, log=True, label=l_p)
            ax4.hist(var_t, bins=np.linspace(t_min,t_max,2000), alpha=0.5, density=True, log=True, label=l_t)
            ax4.set_xlim(quant,t_max)
            ax4.set_ylim(0, y_lim[i][1])
            ax4.set_xlabel(f'Right var {i} tail')
            ax4.legend()

        pdf.savefig(fig)





            


#finish
pdf.close()


main_out.writelines(f'end {time.time() - start}\n')
main_out.flush()                
main_out.close()
    
