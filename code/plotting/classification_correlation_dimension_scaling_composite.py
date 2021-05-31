#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from stdParams import *
from src.plottools import gen_mesh_ax, colorbar
plt.style.use('mpl_style.mplstyle')
import os
import sys

plt.rc('text.latex', preamble=r'\renewcommand{\familydefault}{\sfdefault}')

folders = ["classification_dimension_scaling"
,"classification_dimension_scaling_bcm"]

fig, ax = plt.subplots(2,3,gridspec_kw={'width_ratios': [1,1,1]}, figsize=(FIG_WIDTH,FIG_WIDTH*0.65))

for i in range(2):

    files = os.listdir(os.path.join(DATA_DIR,folders[i]))

    data = []

    for file in files:
        data.append(np.load(os.path.join(DATA_DIR,folders[i]+"/"+file)))

    perf = []
    Ip = []
    labels = []
    dim = []
    s = []

    for dat in data:
        perf.append(dat["perf"])
        Ip.append(dat["I_p_samples"])
        labels.append(dat["labels_samples"])
        dim.append(dat["distract_dimension"])
        s.append(dat["distract_scaling"])

    for k in range(1,len(perf)):
        if(not(np.array_equal(Ip[k].shape,Ip[k-1].shape))):
            print("Ip arrays do not match!")
            sys.exit()
        if(not(np.array_equal(labels[k].shape,labels[k-1].shape))):
            print("labels arrays do not match!")
            sys.exit()
        if(not(np.array_equal(dim[k],dim[k-1]))):
            print("dim arrays do not match!")
            sys.exit()
        if(not(np.array_equal(s[k],s[k-1]))):
            print("scaling arrays do not match!")
            sys.exit()

    perf = np.array(perf)
    Ip = np.array(Ip)
    labels = np.array(labels)
    dim = dim[0]
    s = s[0]

    n_total_samples = Ip.shape[0]*Ip.shape[3]

    Ip_flatten = np.ndarray((Ip.shape[1],Ip.shape[2],Ip.shape[4],Ip.shape[5],n_total_samples))
    labels_flatten = np.ndarray((labels.shape[1],labels.shape[2],labels.shape[4],labels.shape[5],n_total_samples))
    rho = np.ndarray((Ip.shape[1],Ip.shape[2],Ip.shape[4],n_total_samples))

    for k in range(Ip.shape[0]):
        for l in range(Ip.shape[3]):
            Ip_flatten[:,:,:,:,k*perf.shape[3]+l] = Ip[k,:,:,l,:,:,1]
            labels_flatten[:,:,:,:,k*perf.shape[3]+l] = labels[k,:,:,l,:,:]

    Ip = Ip_flatten
    labels = labels_flatten

    for k in range(Ip.shape[0]):
        for l in range(Ip.shape[1]):
            for m in range(Ip.shape[2]):
                for n in range(n_total_samples):
                    rho[k,l,m,n] = np.corrcoef(Ip[k,l,m,:,n],labels[k,l,m,:,n])[1,0]

    rho_mean = rho.mean(axis=3)

    dim_ax = gen_mesh_ax(dim)
    s_ax = gen_mesh_ax(s)

    pc0 = ax[i,0].pcolormesh(s_ax,dim_ax,rho_mean[:,:,0].T,rasterized=True,vmin=0.,vmax=1.)
    pc1 = ax[i,1].pcolormesh(s_ax,dim_ax,rho_mean[:,:,1].T,rasterized=True,vmin=0.,vmax=1.)
    bars0 = ax[i,2].barh(dim-2.5,rho_mean[:,:,0].sum(axis=0),height=5,label="Comp.")
    bars1 = ax[i,2].barh(dim+2.5,rho_mean[:,:,1].sum(axis=0),height=5,label="Point")

    ax[i,2].legend()

    ax[i,0].set_xlabel(r'$s$')
    ax[i,1].set_xlabel(r'$s$')

    ax[i,0].set_ylabel(r'$N_{\rm dist}$')
    ax[i,1].set_ylabel(r'$N_{\rm dist}$')
    ax[i,2].set_ylabel(r'$N_{\rm dist}$')
    ax[i,2].set_xlabel(r'$\Sigma_{\rho}$')

    #ax[0].set_xlim(right=5.)
    #ax[1].set_xlim(right=5.)
    
    colorbar(pc1)


for j in range(3):
    
    fig.tight_layout(h_pad=0.1,w_pad=0.1,pad=0.1)
    
    for i in range(2):
        title_idx_comp = ["A","B","C","D","E","F"][i*3]
        title_idx_point = ["A","B","C","D","E","F"][i*3+1]
        title_idx_aggregate = ["A","B","C","D","E","F"][i*3+2]
        
        title_comp = ('\\makebox['+str(ax[i,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ '
        +'{\\bf '+title_idx_comp+'} \\hfill '+"Compartment Model"+'}')
        title_point = ('\\makebox['+str(ax[i,1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ '
        +'{\\bf '+title_idx_point+'} \\hfill '+"Point Model"+'}')
        title_aggregate = ('\\makebox['+str(ax[i,1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ '
        +'{\\bf '+title_idx_aggregate+'} \\hfill '+"Aggregate Corr."+'}')
            
        ax[i,0].set_title(title_comp,usetex=True)
        ax[i,1].set_title(title_point,usetex=True)
        ax[i,2].set_title(title_aggregate,usetex=True)
        
    fig.tight_layout(h_pad=0.1,w_pad=0.1,pad=0.1)


fig.savefig(os.path.join(PLOT_DIR,"classification_correlation_dimension_scaling_composite.pdf"))
fig.savefig(os.path.join(PLOT_DIR,"classification_correlation_dimension_scaling_composite.png"),dpi=600)

plt.show()
