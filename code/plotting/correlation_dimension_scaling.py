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

files = os.listdir(os.path.join(DATA_DIR,"correlation_dimension_scaling"))

data = []

for file in files:
    data.append(np.load(os.path.join(DATA_DIR,"correlation_dimension_scaling/"+file)))

rho = []
dim = []
s = []

for dat in data:
    rho.append(dat["align_corrcoef"])
    dim.append(dat["distract_dimension"])
    s.append(dat["distract_scaling"])

for k in range(1,len(rho)):
    if(not(np.array_equal(rho[k].shape,rho[k-1].shape))):
        print("corr. arrays do not match!")
        sys.exit()
    if(not(np.array_equal(dim[k],dim[k-1]))):
        print("dim arrays do not match!")
        sys.exit()
    if(not(np.array_equal(s[k],s[k-1]))):
        print("scaling arrays do not match!")
        sys.exit()

rho = np.array(rho)
dim = dim[0]
s = s[0]

n_total_samples = rho.shape[0]*rho.shape[3]

rho_flatten = np.ndarray((rho.shape[1],rho.shape[2],rho.shape[4],n_total_samples))

for k in range(rho.shape[0]):
    for l in range(rho.shape[3]):
        rho_flatten[:,:,:,k*rho.shape[3]+l] = rho[k,:,:,l,:]

rho = rho_flatten

dim_ax = gen_mesh_ax(dim)
s_ax = gen_mesh_ax(s)

n_sweep = rho.shape[2]

rho_mean = rho.mean(axis=3)

fig, ax = plt.subplots(1,3,gridspec_kw={'width_ratios': [1,1,1]}, figsize=(FIG_WIDTH,FIG_WIDTH*0.35))

pc0 = ax[0].pcolormesh(s_ax,dim_ax,rho_mean[:,:,0].T,rasterized=True,vmin=0.,vmax=1.)
pc1 = ax[1].pcolormesh(s_ax,dim_ax,rho_mean[:,:,1].T,rasterized=True,vmin=0.,vmax=1.)
bars0 = ax[2].barh(dim-2.5,rho_mean[:,:,0].sum(axis=0),height=5,label="Comp.")
bars1 = ax[2].barh(dim+2.5,rho_mean[:,:,1].sum(axis=0),height=5,label="Point")

ax[2].legend()

ax[0].set_xlabel(r'$s$')
ax[1].set_xlabel(r'$s$')

ax[0].set_ylabel(r'$N_{\rm dist}$')
ax[1].set_ylabel(r'$N_{\rm dist}$')
ax[2].set_ylabel(r'$N_{\rm dist}$')
ax[2].set_xlabel(r'$\Sigma_{\rho}$')

ax[0].set_xlim(right=4.)
ax[1].set_xlim(right=4.)

ax[0].set_title("Compartment Model",loc="right")
ax[1].set_title("Point Model",loc="right")

colorbar(pc1)

fig.tight_layout(h_pad=0.,w_pad=0.,pad=0.)

fig.savefig(os.path.join(PLOT_DIR,"corr_dimension_scaling.pdf"))
fig.savefig(os.path.join(PLOT_DIR,"corr_dimension_scaling.png"),dpi=600)

plt.show()
