"""

Solve Lotka-Volterra (LV) Competitive equations
https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations

@author: Ilya Timofeyev
"""

import time
start = time.time()

import numpy as np
import matplotlib.pyplot as plt 
from time import perf_counter
import pickle as pk
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


"""
we make a file "result.txt" to keep track on what stage our document is
"""
main_out = open('result.txt', 'w')
main_out.writelines('start\n')
main_out.flush()






"""
trajectory parameters
"""
params = {
    "dim": 4, # LV dimension
    "dt": 0.01, # time step of integration
    "num_lg_steps" : 150, # number of time steps Dt for output
    "num_lg_skip" : 0, # skip this many large time-steps; start output after that
    "num_sm_steps": 200, # number of small time-steps in a big time-step
    "INT": "RK4", # Integrator; RK2, RK4, RK5
    "num_traj": 5000 # how many trjectories to generate
    }






"""
some necessary functions
"""
def init_condition(dim):
    """
    returns initial condition
    """
    
    """
    mean - point from attractor, mean for uniform distribution
    r - variables' range 
    v - variance for uniform distribution
    """
    mean = np.array([0.30, 0.45, 0.13, 0.36])
    r = np.array([0.43, 0.53, 0.36, 0.39])
    v = r * 0.3

    high = mean + v
    low = mean - v
    """
    we limit the variables from below, because integrator does not work with negative values
    """
    low[low < 0.05] = 0.05

    uinit = np.random.uniform(low, high, dim) 
    
    return uinit


def compute_rhs(u):
    """
    compute rhs-hand side of the equations
    r and alf are global
    d/dt u = r * u * (1 - alf *u)
    
    or
    another model 
    d/dt u = r * u + u * alf * u
    
    """
    
    tmp1 = r * u
    tmp2 = iden - alf.dot(u)
    rhs = tmp1 * tmp2
    
    #tmp1 = r * u
    #tmp2 = alf.dot(u)
    #rhs = tmp1 + u * tmp2
    
    
    return rhs



def make_one_step_rk2(u, dim, dt):
    """
    RK2 Integrator
    
    u = u(t)
    k1 = f(u)
    k2 = f(u + 1/2 dt k1)
    u(t+dt) = u(t) + dt k2
    """
    
    k1 = compute_rhs(u)
    k2 = compute_rhs(u + 0.5 * dt * k1)
    
    return u + dt*k2

def make_one_step_rk4(u, dim, dt):
    """
    RK4 Integrator
    
    u = u(t)
    k1 = f(x)
    k2 = f(x + 1/2 dt k1)
    k3 = f(x + 1/2 dt k2)
    k4 = f(x + dt k3)
    x(t+dt) = x(t) + dt (k1 + 2k2 + 2k3 + k4) / 6
    """
    
    k1 = compute_rhs(u)
    k2 = compute_rhs(u + 0.5 * dt * k1)
    k3 = compute_rhs(u + 0.5 * dt * k2)
    k4 = compute_rhs(u + dt * k3)
    
    return u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.


def make_one_step_rk5(u, dim, dt):
    """
    RK5 Integrator
    """
    
    k1 = compute_rhs(u) * dt
    k2 = compute_rhs(u + 1/3*k1) * dt
    k3 = compute_rhs(u + 4/25*k1 + 6/25*k2) * dt
    k4 = compute_rhs(u + 1/4*k1 - 3*k2 + 15/4*k3) * dt
    k5 = compute_rhs(u + 2/27*k1 + 10/9*k2 - 50/81*k3 + 8/81*k4) * dt
    k6 = compute_rhs(u + 2/25*k1 + 12/25*k2 + 2/15*k3 + 8/75*k4) * dt
    
    return u + (23*k1 + 125*k3 - 81*k5 + 125*k6) / 192.



"""
main steps in generating the trajectory
"""
if __name__ == '__main__':

    dim = params["dim"]
    num_traj = params["num_traj"]

    dt = params["dt"]
    num_lg_steps = params["num_lg_steps"]
    num_sm_steps = params["num_sm_steps"]
    num_lg_skip = params["num_lg_skip"]
    Dt = dt * num_sm_steps
    
    
    np.random.seed(0)
    
    """
    base matrices from wiki 
    """
    r = np.array([1., 0.72, 1.53, 1.27])
    alf = np.array([[1., 1.09, 1.52, 0], [0, 1., 0.44, 1.36], [2.33, 0, 1., 0.47], [1.21, 0.51, 0.35, 1.]])
    iden = np.array([1., 1., 1., 1.])

    #alf = 1e-2 * np.array([[-0.743, 323., -72.1, -0.169], [-0.0992, -0.914, -14.3, 0.42], [-0.113, -46., -0.548, -50.9], [-0.211, -1.99, -6.27, -0.0708]])
    #r = 0.005 * np.array([6.89, 6.65, 7.69, 6.17])


    """
    make the choice of integrator
    """
    if params["INT"] == "RK2":
        make_step = make_one_step_rk2
    elif params["INT"] == "RK4":
        make_step = make_one_step_rk4
    elif params["INT"] == "RK5":
        make_step = make_one_step_rk5
    else:
        make_step = make_one_step_rk4
    


    """
    array to save all trajectories
    """
    sampled_traj = np.zeros((num_traj, num_lg_steps, dim))



    """
    array to save individual trajectories
    """
    usave = np.zeros((num_lg_steps, dim))
    



    """
    array of initial points
    """
    init_points = np.zeros((num_traj, dim))
    for traj in range (num_traj):
        init_points[traj] = init_condition(dim)





    for traj in range(num_traj):

        """
        Initial condition
        """
        u = init_points[traj]

        """
        skip num_lg_skip  large time-steps
        """
        for ii in range(num_lg_skip):
            for jj in range(num_sm_steps):
                u = make_step(u, dim, dt)
            
            
        """    
        save initial condition
        """
        usave[0] = u
    
        if traj % 20 == 0:
            main_out.writelines(f'generating MC {traj} {time.time() - start}\n')
            main_out.flush()



        """
        Main Time-Stepping Loop
        """
        for ii in range(2, num_lg_steps+1):
            for jj in range(num_sm_steps):
    
                u = make_step(u, dim, dt)
    
            usave[ii-1] = u
    

    
        sampled_traj[traj] = usave


    with open('train_traj_LV.pkl', 'wb') as file:
        pk.dump(sampled_traj, file)

    """
    with open('test_traj_LV.pkl', 'wb') as file:
        pk.dump(sampled_traj, file)
    """
    

    """
    finish
    """
    main_out.writelines('finish\n')
    main_out.flush()
    main_out.close()



