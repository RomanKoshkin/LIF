# same as above, but jitted with numba (gives a 2x performance boost)

# from numba import jit, config, threading_layer
# config.THREADING_LAYER = 'omp'
import time
import numpy as np
from LIFclasses import *
from numba import njit
import multiprocessing as mp


seed = 20
np.random.seed(seed)

N = 300
T = 30
n_cores = 30
stride = 10
refractory_period = 1

# N = 300
# T = 30
# n_cores = 30
# stride = 10
# refractory_period = 1

dt = 0.01
t = np.arange(0, T, dt)



AP = np.zeros((N, 1))
mp_AP = mp.Array('d', N*1)
AP = np.frombuffer(mp_AP.get_obj()).reshape(*AP.shape)

syn_timer = -np.ones((N, N))
mp_syn_timer = mp.Array('d', N*N)
for a in range(N*N):
    mp_syn_timer[a] = -1.0
syn_timer = np.frombuffer(mp_syn_timer.get_obj()).reshape(*syn_timer.shape)

AP_delayed = np.zeros((N, N))
mp_AP_delayed = mp.Array('d', N*N)
AP_delayed = np.frombuffer(mp_AP_delayed.get_obj()).reshape(*AP_delayed.shape)

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


# for xx in AP[0:10, 0]:
#     if xx == 1:
#         print('OK')

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
mp_V = mp.Array('d', 1*N)
for a in range(N):
    mp_V[a] = EL
V = np.frombuffer(mp_V.get_obj()).reshape(*V.shape)

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

t = np.arange(0, T, dt)
V = V * 0 + EL          # ????????????????????????????

ampa = np.zeros((N, N))
mp_ampa = mp.Array('d', N*N)
ampa = np.frombuffer(mp_ampa.get_obj()).reshape(*ampa.shape)

nmda = np.zeros((N, N))
mp_nmda = mp.Array('d', N*N)
nmda = np.frombuffer(mp_nmda.get_obj()).reshape(*nmda.shape)

gaba = np.zeros((N, N))
mp_gaba = mp.Array('d', N*N)
gaba = np.frombuffer(mp_gaba.get_obj()).reshape(*gaba.shape)

in_refractory = np.zeros((1, N))
mp_in_refractory = mp.Array('d', N)
in_refractory = np.frombuffer(mp_in_refractory.get_obj()).reshape(*in_refractory.shape)

VV = np.zeros((N, len(t)))
mp_VV = mp.Array('d', N*len(t))
VV = np.frombuffer(mp_VV.get_obj()).reshape(*VV.shape)

AMPA = np.zeros((N, N, len(t)))
mp_AMPA = mp.Array('d', N*N*len(t))
AMPA = np.frombuffer(mp_AMPA.get_obj()).reshape(*AMPA.shape)

NMDA = np.zeros((N, N, len(t)))
mp_NMDA = mp.Array('d', N*N*len(t))
NMDA = np.frombuffer(mp_NMDA.get_obj()).reshape(*NMDA.shape)

GABA = np.zeros((N, N, len(t)))
mp_GABA = mp.Array('d', N*N*len(t))
GABA = np.frombuffer(mp_GABA.get_obj()).reshape(*GABA.shape)

Ie = np.zeros((N, len(t)))
mp_Ie = mp.Array('d', N*len(t))
Ie = np.frombuffer(mp_Ie.get_obj()).reshape(*Ie.shape)

cc = np.zeros((2,))
mp_cc = mp.Array('i', 2)
cc = np.frombuffer(mp_cc.get_obj())


class MyObject:
    def __init__(self):
        self.pool = mp.Pool(processes=4)

    def function(self, k):
        _ = self.pool.map(on_core, k)
    
    def destroy(self):
        self.pool.close()

# @jit (nopython=True, fastmath=True, nogil=True, parallel=True)
def on_core(k):
    global cc, N, T, dt, n_cores, stride, t, refractory_period, AP, syn_timer, AP_delayed, V_E, V_I, EL, Vth, Vr, Vspike, neur_type_mask, tau, tau_ampa, tau_nmda, tau_gaba, V, w, ampa, nmda, gaba, VV, AMPA, NMDA, GABA, Ie
    i = int(cc[0])
    sp = range(k, k+stride)

    ampa[sp, :] += (-ampa[sp, :] / tau_ampa + neur_type_mask[sp, :] * AP_delayed[sp, :] * w[sp, :]) * dt
    nmda[sp, :] += (-nmda[sp, :] / tau_nmda + neur_type_mask[sp, :] * AP_delayed[sp, :] * w[sp, :]) * dt
    gaba[sp, :] += (-gaba[sp, :] / tau_gaba + (1.0 - neur_type_mask[sp, :]) * AP_delayed[sp, :] * w[sp, :]) * dt

    AP[sp,0] = AP[sp,0] * 0.0

    AP_delayed[sp, :] = AP_delayed[sp, :] * 0.0                     # !

    where_reset = np.where(V[:, sp].flatten() >= Vspike)[0]
    V[0, [i+k for i in where_reset]] = Vr
    in_refractory[0, [i+k for i in where_reset]] = refractory_period
    
    where_refractory = np.where(in_refractory[:, sp].flatten() > 0)[0]

    I_E = np.sum(-ampa[:,sp] * (V[:, sp] - V_E) - 0.1 * nmda[:,sp] * (V[:, sp] - V_E), axis=0) # !
    I_I = np.sum(-gaba[:,sp] * (V[:, sp] - V_I), axis=0)                                        # !

    dV = (-(V[:, sp].flatten() - EL) / tau[:,sp].flatten() + I_E + I_I + Ie[sp, i].flatten()) * dt

    dV[where_refractory] = 0
    V[0, sp] += dV

    where_is_AP = np.where(V[:, sp].flatten() > Vth)[0]

    V[0, [i+k for i in where_is_AP]] = Vspike
    VV[sp, i] = V[:, sp].flatten()

    AP[[i+k for i in where_is_AP], :] = 1
    AMPA[sp, :, i] = ampa[sp, :]    # !
    NMDA[sp, :, i] = nmda[sp, :]    # !
    GABA[sp, :, i] = gaba[sp, :]   # !
    in_refractory[:, sp] -= dt

# @jit (nopython=True, fastmath=True, nogil=True, parallel=True)
def run_numba(tt):
    global cc, my_object, N, T, dt, n_cores, stride, t, refractory_period, AP, syn_timer, AP_delayed, V_E, V_I, EL, Vth, Vr, Vspike, neur_type_mask, tau, tau_ampa, tau_nmda, tau_gaba, V, w, ampa, nmda, gaba, VV, AMPA, NMDA, GABA, Ie

    for i in range(len(t)):
        if i%100 == 0:
            print(i, f'{(time.time() - tt):.2f}')
            tt = time.time()
        
        
        preAP = AP.dot(np.logical_not(AP.T)*1.0)  # turn postsynaptic APs to presynaptic
        inc, out = np.where(preAP == 1) # !
        syn_timer[inc, out] = 2.0 # np.round(1 + np.random.rand() * 2, 2)  # randomized synaptic delay timer
        inc, out = np.where(np.round(syn_timer, 4) == 0) # !
        AP_delayed[inc, out] = 1.0     # !


        # my_object.function(np.arange(0, n_cores*stride, stride))
        for k in np.arange(0, n_cores*stride, stride):
            on_core(k)

            # maybe flip
        syn_timer -= dt
        AP_delayed *= 0.0
        cc += 1
    
    my_object.destroy()
            
    return w, neur_type_mask, t, AMPA, NMDA, GABA, VV

tt = time.time()

my_object = MyObject()

w, neur_type_mask, t, AMPA, NMDA, GABA, VV = run_numba(tt)
print(f'{(time.time() - tt):.2f} s')

plot_sim_res('V', neur_type_mask, t, AMPA, NMDA, GABA, VV, neur=[0,10])
plt.show()
# plot_sim_res('V', net, neur=[2,3])
# plot_sim_res('V', net, neur=[4,5])
plot_sim_res('G', neur_type_mask, t, AMPA, NMDA, GABA, VV, neur=[30,31], syn=[10,11,12,13,14])
# plot_sim_res('E', net, neur=[4,5])
plt.show()

plt.figure(figsize=(15,5))
x, y = np.where(VV > 0)
plt.plot(y, x, 'bo', ms=0.5)

plt.figure(figsize=(15,5))
_ = plt.plot(VV.T)
plt.show()
