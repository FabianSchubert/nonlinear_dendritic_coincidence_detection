#! /usr/bin/env python3

import numpy as np
from stdParams import *
from src.neuronmodel import *
from tqdm import tqdm
import os

from datetime import datetime

from multiprocessing import Pool

N_processes = 6 #number of parallel processes

N_sweep_distraction_scaling = 20
N_sweep_distraction_dimension = 9
N_samples = 1
distract_scaling = np.linspace(0.,10.,N_sweep_distraction_scaling)

N_p = 100
N_out = 2

distract_dimension = np.linspace(1.,N_p-1,N_sweep_distraction_dimension).astype("int")

T = int(2e5)
T_test = int(1e4)
T_sample = int(1e3)
t_ax = np.arange(T)

stdMainDir = .25
distMainDir = 2.

modes = ["comp","point"]
#modes = ["comp_bcm","point_bcm"]
#modes = ["point"]

perf = np.ndarray((N_sweep_distraction_scaling,
                    N_sweep_distraction_dimension,
                    N_samples,len(modes)))

I_p_samples = np.ndarray((N_sweep_distraction_scaling,
                    N_sweep_distraction_dimension,
                    N_samples,len(modes),T_sample,N_out))
                    
labels_samples = np.ndarray((N_sweep_distraction_scaling,
                    N_sweep_distraction_dimension,
                    N_samples,len(modes),T_sample))

params = []

for mode in modes:
    for s in range(N_sweep_distraction_scaling):
        for n in range(N_sweep_distraction_dimension):
            for i in range(N_samples):
                params.append([distract_dimension[n],distract_scaling[s],mode])



def run_sim(arglist):
    
    N_distr = arglist[0]
    s = arglist[1]
    mode = arglist[2]
    
    patterns = np.random.normal(0.,1.,(T,N_p))
    patterns[:,0] *= stdMainDir
    patterns[:,1:N_distr+1] *= s
    patterns[:,N_distr+1:] *= 0.
    patterns[:,0] += (1.*(np.random.rand(T) <= 0.5) - 0.5) * distMainDir

    
    w_p = np.ones((T,N_out,N_p))
    
    n_d = np.ones((T,N_out))
    n_p = np.ones((T,N_out))
    b_d = np.zeros((T,N_out))
    b_p = np.zeros((T,N_out))
    
    x_d = np.ndarray((T,2))
    x_d[:,0] = 1.*(patterns[:,0] < 0.)
    x_d[:,1] = 1.*(patterns[:,0] > 0.)
    
    #generate random orthonormal basis via qr-decomposition
    Q,R = np.linalg.qr(np.random.normal(0.,1.,(N_p,N_p)))
    #Transform x_p with orthogonal basis.
    x_p = (Q @ patterns.T).T
    
    I_p = np.ndarray((T,N_out))
    I_d = np.ndarray((T,N_out))
    
    x_p_av = np.ndarray((T,N_p))
    
    I_p_av = np.ndarray((T,N_out))
    I_d_av = np.ndarray((T,N_out))
    
    y = np.ndarray((T,N_out))
    
    y_squ_av = np.ndarray((T,N_out))
    y_av = np.ndarray((T,N_out))
    
    #### Init values
                    
    w_p[0] = (w_p[0].T / np.linalg.norm(w_p[0],axis=1)).T
                    
    I_p[0] = n_p[0] * (w_p[0] @ x_p[0]) - b_p[0]
    I_d[0] = n_d[0] * x_d[0] - b_d[0]
    
    x_p_av[0] = 0.
    
    I_p_av[0] = 0.
    I_d_av[0] = 0.
    
    if(mode == "comp"):
        y[0] = psi(I_p[0],I_d[0])
    else:
        y[0] = phi(I_p[0] + I_d[0])
        
    y_squ_av[0] = y[0]**2.
    y_av[0] = y[0]

    for t in tqdm(range(1,T),disable=True,leave=False):
        
        #if(mode=="comp"):
        #    w_p[t] = w_p[t-1] + mu_w * np.outer(y[t-1]*(y[t-1]-thetay),x_p[t-1] - x_p_av[t-1])
        #if(mode=="point"):
        w_p[t] = (w_p[t-1]
            + mu_w * (np.outer(y[t-1]*(y[t-1]-y_squ_av[t-1]),x_p[t-1] - x_p_av[t-1]) - eps_dec*w_p[t-1]))
            
        #if(mode=="comp"):
        #    w_p[t] = (w_p[t].T / np.linalg.norm(w_p[t],axis=1)).T
        
        b_p[t] = b_p[t-1] + mu_b * (I_p[t-1] - I_pt)
        b_d[t] = b_d[t-1] + mu_b * (I_d[t-1] - I_dt)
        
        if(mode=="comp"):
            n_p[t] = n_p[t-1] + mu_n * (VI_pt - (I_p[t-1] - I_p_av[t-1])**2.)
        
        n_d[t] = n_d[t-1] + mu_n * (VI_dt - (I_d[t-1] - I_d_av[t-1])**2.)
        
        I_p[t] = n_p[t] * (w_p[t] @ x_p[t]) - b_p[t]
        I_d[t] = n_d[t] * x_d[t] - b_d[t]
        
        x_p_av[t] = (1.-mu_av)*x_p_av[t-1] + mu_av*x_p[t]
        
        I_p_av[t] = (1.-mu_av)*I_p_av[t-1] + mu_av*I_p[t]
        I_d_av[t] = (1.-mu_av)*I_d_av[t-1] + mu_av*I_d[t]
        
        
        if(mode == "comp"):
            y[t] = psi(I_p[t],I_d[t])
        if(mode == "point"):
            y[t] = phi(I_p[t] + I_d[t])
            
        y_squ_av[t] = (1.-mu_av)*y_squ_av[t-1] + mu_av * y[t]**2.
        y_av[t] = (1.-mu_av)*y_av[t-1] + mu_av * y[t]
                    
    patterns_test = np.random.normal(0.,1.,(T_test,N_p))
    patterns_test[:,0] *= stdMainDir
    patterns_test[:,1:N_distr+1] *= s
    patterns_test[:,N_distr+1:] *= 0.
    patterns_test[:,0] += (1.*(np.random.rand(T_test) <= 0.5) - 0.5) * distMainDir
    
    lab_test = 1.*(patterns_test[:,0] > 0.)
    
    x_p_test = (Q @ patterns_test.T).T
                    
    I_p_test = n_p[-1] * (w_p[-1] @ x_p_test.T).T - b_p[-1]
    
    pred = np.argmax(I_p_test,axis=1)
    return (1.*(pred == lab_test)).mean(), I_p_test[:T_sample], lab_test[:T_sample]

pool = Pool(N_processes)
output_list = list(tqdm(pool.imap_unordered(run_sim, params), total=len(params)))

#with Pool() as p:
#    perf_list = p.map(run_sim,params)

k = 0
for mode in modes:
    for s in range(N_sweep_distraction_scaling):
        for n in range(N_sweep_distraction_dimension):
            for i in range(N_samples):
                perf[s,n,i,modes.index(mode)] = output_list[k][0]
                I_p_samples[s,n,i,modes.index(mode),:,:] = output_list[k][1]
                labels_samples[s,n,i,modes.index(mode),:] = output_list[k][2]
                k += 1

savefold = os.path.join(DATA_DIR,"classification_dimension_scaling_bcm/")
if not os.path.exists(savefold):
    os.makedirs(savefold)

np.savez(os.path.join(savefold
        +"classification_dimension_scaling_bcm_"
        +datetime.now().strftime("%d-%m-%y-%H:%M:%S")
        +".npz"),
         perf = perf,
         I_p_samples = I_p_samples,
         labels_samples = labels_samples,
         distract_scaling = distract_scaling,
         distract_dimension = distract_dimension)
