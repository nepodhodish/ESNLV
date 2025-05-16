"""

Solve Lotka-Volterra (LV) Competitive equations
https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations

@author: Ilya Timofeyev
"""



import numpy as np
import matplotlib.pyplot as plt 
from time import perf_counter
import pickle as pk


#printing out updates
main_out = open('result.txt', 'w')
main_out.writelines(f'start\n')
main_out.flush()







#trajectory parameters
params = {
    "dim": 4, #LV dimension
    "dt": 0.01, # time step of integration
    "num_lg_steps" : 5000000, # number of time steps Dt for output
    "num_lg_skip" : 100000, # skip this many large time-steps; start output after that
    "num_sm_steps": 200, # number of small time-steps in a big time-step
    "INT": "RK4", #Integrator; RK2, RK4, RK5
    }








def init_condition(dim):
    """
    returns initial condition
    """
    
    uinit = np.zeros(dim)

    #adding small random pertrubations for generating new trajectories
    uinit[0] = 0.2 + np.random.uniform(-0.05, 0.05)
    uinit[1] = 0.2 + np.random.uniform(-0.05, 0.05)
    uinit[2] = 0.2 + np.random.uniform(-0.05, 0.05)
    uinit[3] = 0.2 + np.random.uniform(-0.05, 0.05)
    
    return uinit


def compute_rhs(u):
    """
    compute rhs-hand side of the equations
    r and alf are global
    d/dt u = r * u * (1 - alf *u)
    
    """
    
    tmp1 = r * u
    tmp2 = iden - alf.dot(u)
    rhs = tmp1 * tmp2
    
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




if __name__ == '__main__':

    dim = params["dim"]

    dt = params["dt"]
    num_lg_steps = params["num_lg_steps"]
    num_sm_steps = params["num_sm_steps"]
    num_lg_skip = params["num_lg_skip"]
    Dt = dt * num_sm_steps
    
    
    """
    base matrices from wiki 
    """
    r = np.array([1., 0.72, 1.53, 1.27])
    alf = np.array([[1., 1.09, 1.52, 0], [0, 1., 0.44, 1.36], [2.33, 0, 1., 0.47], [1.21, 0.51, 0.35, 1.]])
    iden = np.array([1., 1., 1., 1.])



    if params["INT"] == "RK2":
        make_step = make_one_step_rk2
        print("RK2 Integrator")
    elif params["INT"] == "RK4":
        make_step = make_one_step_rk4
        print("RK4 Integrator")
    elif params["INT"] == "RK5":
        make_step = make_one_step_rk5
        print("RK4 Integrator")
    else:
        make_step = make_one_step_rk4
        print("Unknown Integrator ", params["INT"])
        print("Using RK4 Integrator")
    
    

    # initial values    

    u = init_condition(dim)
    

    # arrays for saving the solution
    usave = np.zeros((dim, num_lg_steps))
    
    print("Total time = ", Dt * (num_lg_steps + num_lg_skip))
    print("Skip time = ", Dt * num_lg_skip)
    print("Output time = ", Dt * num_lg_skip, Dt * (num_lg_steps + num_lg_skip))
    print("Dt output = ", Dt)
    print("initail" , u)
    print("-----------------------------")
    
    
    

    main_out.writelines('skipping\n')
    main_out.flush()
    print(f'skip')

    #------------------
    # skip num_lg_skip  large time-steps
    #------------------
    for ii in range(num_lg_skip):
        for jj in range(num_sm_steps):
    
            u = make_step(u, dim, dt)
    

    
    # save initial condition
    usave[:, 0] = u
    
    
    main_out.writelines('generating\n')
    main_out.flush()
    print(f'generating')

    #------------------
    # Main Time-Stepping Loop
    #------------------


    for ii in range(2, num_lg_steps+1):
        for jj in range(num_sm_steps):
    
            u = make_step(u, dim, dt)
    
        usave[:, ii-1] = u

        if ii % 10000 == 0:
            main_out.writelines(f'steps done {ii}\n')
            main_out.flush()
            print(f'steps already {ii}')
    

    
    with open('trace_LV_2_e7.pkl', 'wb') as file:
        pk.dump(usave.T, file)



    
    #finish
    main_out.writelines('finish\n')
    main_out.flush()
    main_out.close()



