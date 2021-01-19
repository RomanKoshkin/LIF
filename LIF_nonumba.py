# same as above, but jitted with numba (gives a 2x performance boost)

import time
import numpy as np
from LIFclasses import *
from numba import jit

seed = 20
np.random.seed(seed)

def run():
    N = 300
    T = 30
    dt = 0.01
    t = np.arange(0, T, dt)
    
    refractory_period = 1
    AP = np.zeros((N, 1))
    
    syn_timer = -np.ones((N, N))
    AP_delayed = np.zeros((N, N))
    preAP = np.zeros((N,N))

    # equilibrium potentials:
    V_E = 0.0
    V_I = -80.0  # equilibrium potential for the inhibitory synapse
    EL = -65.0  # leakage potential, mV

    # critical voltages:
    Vth = -55.0  # threshold after which an AP is fired,   mV
    Vr = -70.0  # reset voltage (after an AP is fired), mV
    Vspike = 10.0

    # define neuron types in the network:
    neur_type_mask = np.zeros_like(AP)
    neur_type_mask[:10] = 0
    neur_type_mask[10:] = 1  

    # initialize spikes
    exc_id = np.where(neur_type_mask == 1)[0]
    ons = np.random.choice(exc_id, int(0.4*len(exc_id)), replace=False)
    for a in ons:
        AP[a, 0] = 1

    # taus
    tau = np.zeros((1, N))
    idx = np.where(neur_type_mask == 0)[0]
    for j in idx:
        tau[0, j] = 10
    idx = np.where(neur_type_mask == 1)[0]
    for j in idx:
        tau[0, j] = 20
    

    tau_ampa = 8
    tau_nmda = 100
    tau_gaba = 8
    
    V = np.ones((1, N)) * EL


    # define weights:
    w = np.zeros((N, N))
    NE = int(N*0.8)
    NI = N-NE
    # II
    for i in range(NI):
        for j in range(NI):
            if np.random.rand() > 0.8:
                w[i,j] = np.random.rand() + 0.1
    # IE
    for i in range(NI,N):
        for j in range(NI):
            if np.random.rand() > 0.8:
                w[i,j] = np.random.rand() + 1.2
    # EI
    for i in range(NI):
        for j in range(NI,N):
            if np.random.rand() > 0.8:
                w[i,j] = np.random.rand() + 0.7
    # EE
    for i in range(NI,N):
        for j in range(NI,N):
            if np.random.rand() > 0.8:
                w[i,j] = np.abs(np.random.rand() + 0.1)
    
    # prohibit self-connections
    for i in range(N):
        w[i, i] = 0

    i = 0
    t = np.arange(0, T, dt)
    V = V * 0 + EL
    
    
    ampa = np.zeros((N, N))
    nmda = np.zeros((N, N))
    gaba = np.zeros((N, N))
    in_refractory = np.zeros((1, N))
    VV = np.zeros((N, len(t)))
    AMPA = np.zeros((N, N, len(t)))
    NMDA = np.zeros((N, N, len(t)))
    GABA = np.zeros((N, N, len(t)))

    print(np.random.rand())
    tt = time.time()
    for i in range(len(t)):
        if i%100 == 0:
            print(i, f'{(time.time() - tt):.2f}')
            tt = time.time()
       
        preAP = AP.dot(np.logical_not(AP.T)*1.0)  # turn postsynaptic APs to presynaptic
 
        inc, out = np.where(preAP == 1)
        syn_timer[inc, out] = 2.0 #np.round(1 + np.random.rand() * 2, 2)  # randomized synaptic delay timer
        inc, out = np.where(np.round(syn_timer, 4) == 0)
        AP_delayed[inc, out] = 1

        ampa += (-ampa / tau_ampa + neur_type_mask * AP_delayed * w) * dt
        nmda += (-nmda / tau_nmda + neur_type_mask * AP_delayed * w) * dt
        gaba += (-gaba / tau_gaba + (1.0 - neur_type_mask) * AP_delayed * w) * dt

        AP = AP * 0.0
        AP_delayed = AP_delayed * 0.0
        where_reset = np.where(V.flatten() >= Vspike)[0]
        V[:, where_reset] = Vr
        in_refractory[:, where_reset] = refractory_period
        where_refractory = np.where(in_refractory.flatten() > 0)[0]

        I_E = np.sum(-ampa * (V - V_E) - 0.1 * nmda * (V - V_E), 0)
        I_I = np.sum(-gaba * (V - V_I), 0)

        dV = (-(V - EL) / tau + I_E + I_I) * dt
        
        for j in where_refractory:
            dV[0, j] = 0.0
        V += dV

        where_is_AP = np.where(V.flatten() > Vth)[0]

        V[:, where_is_AP] = Vspike
        VV[:, i] = V.flatten()
        
        AP[where_is_AP, :] = 1
        AMPA[:, :, i] = ampa
        NMDA[:, :, i] = nmda
        GABA[:, :, i] = gaba
        in_refractory -= dt

        syn_timer -= dt
        AP_delayed *= 0.0
    return w, neur_type_mask, t, AMPA, NMDA, GABA, VV

tt = time.time()
w, neur_type_mask, t, AMPA, NMDA, GABA, VV = run()
print(f'{(time.time() - tt):.2f} s')

plot_sim_res('V', neur_type_mask, t, AMPA, NMDA, GABA, VV, neur=[0,10])
plt.show()
# plot_sim_res('V', net, neur=[2,3])
# plot_sim_res('V', net, neur=[4,5])
plot_sim_res('G', neur_type_mask, t, AMPA, NMDA, GABA, VV, neur=[3], syn=[10])
plt.show()
# plot_sim_res('E', net, neur=[4,5])

plt.figure(figsize=(15,5))

x, y = np.where(VV > 0)
plt.plot(y, x, 'bo', ms=0.5)
plt.show()