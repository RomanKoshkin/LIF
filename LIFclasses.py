import numpy as np
import matplotlib.pyplot as plt
# from NeuroTools import stgen
# from utils import axvlines as pltX
from scipy.signal import find_peaks
from scipy.stats import binom
from sklearn import preprocessing
import time
import numba

def plt_elig_wts(e, w):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Eligibility')
    plt.imshow(e)
    plt.ylabel('signaling neurons (synapses \nlistening to respective neurons)')
    plt.xlabel('receiving neurons')
    plt.subplot(1,2,2)
    plt.imshow(w)
    plt.title('Weights')
    plt.ylabel('signaling neurons (synapses \nlistening to respective neurons)')
    plt.xlabel('receiving neurons')

def plot_sim_res4class(what, net, neur = [0], syn=[]):
    
    N = net.N
#     POSTSYN = net.POSTSYN
    PRESYN = net.PRESYN
#     ELIGIBILITY = net.ELIGIBILITY
    VV = net.VV
    I_EE = net.I_EE
    I_II = net.I_II
    AMPA = net.AMPA
    NMDA = net.NMDA
    GABA = net.GABA
    mask = net.neur_type_mask
    
    if what=='V':
        leg = []
        plt.figure(figsize=(15,5))
        plt.subplot(2,1,1)
        plt.title('Membrane voltage')
        for n in range(len(neur)):
            plt.plot(net.t, VV[neur[n],:])
            leg.append('{} exc'.format(neur[n]) if net.neur_type_mask[neur[n]]==1 else '{} inh'.format(neur[n]))
        plt.legend(leg, title='Neuron')
        plt.tight_layout()
    elif what=='I':
        plt.figure(figsize=(15,5))
        c = 0
        for n in neur:
            c += 1
            plt.subplot(len(neur),1,c)
            plt.plot(net.t, I_EE[n,:], label='I_E, neur. {}'.format(n))
            plt.plot(net.t, I_II[n,:], linestyle='--', label='I_I, neur. {}'.format(n))
            plt.legend(title='Currents')
        plt.title('I_E and I_I')
        plt.tight_layout()
    elif what=='G':
        plt.figure(figsize=(15, 2*N))
        for n in range(len(neur)):
            plt.subplot(N,1,n+1)
            for s in syn:
                plt.plot(net.t, AMPA[s, neur[n],:], label='AMPA, syn.{}'.format(s))
                plt.plot(net.t, NMDA[s, neur[n],:], linestyle='--', label='NMDA, syn.{}'.format(s))
                plt.plot(net.t, GABA[s, neur[n],:], linestyle=':', label='GABA, syn.{}'.format(s) )
                plt.title('AMPA, NMDA and GABA conductances in receiving neuron: {}'.format(neur[n]))
                plt.legend()
        plt.tight_layout()
    
    elif what=='E':
        plt.figure(figsize=(15,7))
        ax = []
        for i in range(3):
            ax.append(plt.subplot(3,1,i+1))
        for i in neur:
            ax[0].plot(net.t, POSTSYN[i,:], label='Neuron {}'.format(i))
        for i in neur:
            for j in range(N):
                ax[1].plot(net.t, PRESYN[j,i,:], label='Neuron {}, Syn. {}'.format(i,j))
                ax[2].plot(net.t, ELIGIBILITY[j,i,:], label='Neuron {}, Syn. {}'.format(i,j))
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[1].set_title('Presynaptic activity')
        ax[0].set_title('Postsynaptic activity')
        ax[2].set_title('Eligibility')
        plt.tight_layout()
    else:
        pass
    

    
def plot_sim_res(what, neur_type_mask, t, AMPA, NMDA, GABA, VV, neur = [0], syn=[]):
    
    N = 50
    
    if what=='V':
        leg = []
        plt.figure(figsize=(15,5))
        plt.subplot(2,1,1)
        plt.title('Membrane voltage')
        for n in range(len(neur)):
            plt.plot(VV[neur[n],:])
            leg.append('{} exc'.format(neur[n]) if neur_type_mask[neur[n]]==1 else '{} inh'.format(neur[n]))
        plt.legend(leg, title='Neuron')
        plt.tight_layout()
    elif what=='I':
        plt.figure(figsize=(15,5))
        c = 0
        for n in neur:
            c += 1
            plt.subplot(len(neur),1,c)
            plt.plot(net.t, I_EE[n,:], label='I_E, neur. {}'.format(n))
            plt.plot(net.t, I_II[n,:], linestyle='--', label='I_I, neur. {}'.format(n))
            plt.legend(title='Currents')
        plt.title('I_E and I_I')
        plt.tight_layout()
    elif what=='G':
        plt.figure(figsize=(15, 2*N))
        for n in range(len(neur)):
            plt.subplot(N,1,n+1)
            for s in syn:
                plt.plot(t, AMPA[s, neur[n],:], label='AMPA, syn.{}'.format(s))
                plt.plot(t, NMDA[s, neur[n],:], linestyle='--', label='NMDA, syn.{}'.format(s))
                plt.plot(t, GABA[s, neur[n],:], linestyle=':', label='GABA, syn.{}'.format(s) )
                plt.title('AMPA, NMDA and GABA conductances in receiving neuron: {}'.format(neur[n]))
                plt.legend()
        plt.tight_layout()
    
    elif what=='E':
        plt.figure(figsize=(15,7))
        ax = []
        for i in range(3):
            ax.append(plt.subplot(3,1,i+1))
        for i in neur:
            ax[0].plot(net.t, POSTSYN[i,:], label='Neuron {}'.format(i))
        for i in neur:
            for j in range(N):
                ax[1].plot(net.t, PRESYN[j,i,:], label='Neuron {}, Syn. {}'.format(i,j))
                ax[2].plot(net.t, ELIGIBILITY[j,i,:], label='Neuron {}, Syn. {}'.format(i,j))
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[1].set_title('Presynaptic activity')
        ax[0].set_title('Postsynaptic activity')
        ax[2].set_title('Eligibility')
        plt.tight_layout()
    else:
        pass