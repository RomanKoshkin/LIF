import numpy as np
import matplotlib.pyplot as plt
# from NeuroTools import stgen
# from utils import axvlines as pltX
from scipy.signal import find_peaks
import time as TIME
from scipy.stats import binom
from sklearn import preprocessing

   
   
def run():
    N = 50
    T = 100
    dt = 0.01
    t = np.arange(0, T, dt)
    seed = 20

    stim_type = 1
    np.random.seed(seed)
    refractory_period = 1
    AP = np.zeros((N, 1))

    syn_timer = -np.ones((N, N))
    AP_delayed = np.zeros((N, N))

    # equilibrium potentials:
    V_E = 0
    V_I = -80  # equilibrium potential for the inhibitory synapse
    EL = -65  # leakage potential, mV

    # critical voltages:
    Vth = -55  # threshold after which an AP is fired,   mV
    Vr = -70  # reset voltage (after an AP is fired), mV
    Vspike = 10

    # define neuron types in the network:
    neur_type_mask = np.random.choice([0, 1], N, p=[0.5, 0.5]).reshape(*AP.shape).astype('float64')
    neur_type_mask[0] = 1
    neur_type_mask[1] = 1
    neur_type_mask[-1] = 1
    neur_type_mask[-2] = 1
#         print('NEUR_TYPE_MASK: {}'.format(neur_type_mask))

    # taus
    tau = np.zeros((1, N))
    tau[0, np.where(neur_type_mask == 0)[0]] = 10
    tau[0, np.where(neur_type_mask == 1)[0]] = 20

    tau_ampa = 8
    tau_nmda = 100
    tau_gaba = 8
#         postsyn_tau = 0.1
    # presyn_tau = 5
#         tau_eligibility = 40

    V = np.ones((1, N)) * EL


    # define weights:
    w = np.ones((N, N)).astype('float') * 0.8
    w = w + np.random.rand(*w.shape) * 0.4

    for i in range(N):
        w[i, i] = 0
    w[:, :2] = 0  # input neurons don't receive synapses
    w[:2, -2:] = 0  # output neurons don't listen to input neurons
    w[-2:, -2:] = 0  # output neurons don't listen to themselves
    w[-2:,:] = 0 # output neurons don't feed back to reservoir
    #         w[-2, -1] = 1
    #         w[-1, -2] = 1

    w_mask = np.ones_like(w)
    w_mask[:,:2] = 0
    w_mask[-2:,:] = 0
    w_mask[:,:-2] = 0
#         w_mask[2,2] = 0
#         w_mask[3,3] = 0
    w_mask[:2,-2:] = 0
#         w_mask = np.zeros_like(w)  # fix all weights except readout connections
#         w_mask[2:-2, -2:] = 1
   




    i = 0
    t = np.arange(0, T, dt)
    V = V * 0 + EL
    
    AP = AP * 0
    
    ampa = np.zeros((N, N))
    nmda = np.zeros((N, N))
    gaba = np.zeros((N, N))
#         eligibility = np.zeros((N, N))
#         postsyn_act = np.zeros((1, N))
    presyn_act = np.zeros((N, N))
    in_refractory = np.zeros((1, N))
    VV = np.zeros((N, len(t)))
    I_EE = np.zeros((N, len(t)))
    I_II = np.zeros((N, len(t)))
    AMPA = np.zeros((N, N, len(t)))
    NMDA = np.zeros((N, N, len(t)))
    GABA = np.zeros((N, N, len(t)))
#         POSTSYN = np.zeros((N, len(t)))
    PRESYN = np.zeros((N, N, len(t)))
#         ELIGIBILITY = np.zeros((N, N, len(t)))
    Ie = np.zeros((N, len(t)))
    stim_t = range(int(15 / dt), int(40 / dt))
    if stim_type == 0:
        Ie[0, stim_t] = 12
        Ie[1, stim_t] = 0
    if stim_type == 1:
        Ie[0, stim_t] = 0
        Ie[1, stim_t] = 12

   
    for i in range(len(t)):

        preAP = AP.dot(np.logical_not(AP.T).astype(int))  # turn postsynaptic APs to presynaptic
        inc, out = np.where(preAP == 1)
        syn_timer[inc, out] = np.round(1 + np.random.rand() * 2, 2)  # randomized synaptic delay timer
        inc, out = np.where(np.round(syn_timer, 4) == 0)
        AP_delayed[inc, out] = 1

        ampa += (-ampa / tau_ampa + neur_type_mask * AP_delayed * w) * dt
        nmda += (-nmda / tau_nmda + neur_type_mask * AP_delayed * w) * dt
        gaba += (-gaba / tau_gaba + (1.0 - neur_type_mask) * AP_delayed * w) * dt
#         postsyn_act += (-postsyn_act / postsyn_tau + AP.T) * dt

        AP = AP * 0
        AP_delayed = AP_delayed * 0
        where_reset = np.where(V.flatten() >= Vspike)[0]
        V[:, where_reset] = Vr
        in_refractory[:, where_reset] = refractory_period
        where_refractory = np.where(in_refractory.flatten() > 0)[0]

        I_E = np.sum(-ampa * (V - V_E) - 0.1 * nmda * (V - V_E), 0)
        I_I = np.sum(-gaba * (V - V_I), 0)

        dV = (-(V - EL) / tau + I_E + I_I + Ie[:, i]) * dt
        dV[0, where_refractory] = 0
        V += dV

        where_is_AP = np.where(V.flatten() > Vth)[0]

        V[:, where_is_AP] = Vspike
        AP[where_is_AP, :] = 1

        I_EE[:, i] = I_E
        I_II[:, i] = I_I
        VV[:, i] = V
        AMPA[:, :, i] = ampa
        NMDA[:, :, i] = nmda
        GABA[:, :, i] = gaba
#         POSTSYN[:, i] = postsyn_act.flatten()
        PRESYN[:, :, i] = presyn_act
#         ELIGIBILITY[:, :, i] = eligibility
        in_refractory -= dt

        syn_timer -= dt
        AP_delayed *= 0

    return AMPA, NMDA, GABA, VV
        
AMPA, NMDA, GABA, VV = run()