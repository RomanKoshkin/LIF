import numpy as np
import matplotlib.pyplot as plt
# from NeuroTools import stgen
# from utils import axvlines as pltX
from scipy.signal import find_peaks
import time as TIME
from scipy.stats import binom
from sklearn import preprocessing


class Net2():
    def __init__(self, N=4, T=100, dt=0.01, seed=10):

        np.random.seed(seed)
        self.N = N
        self.T = T
        self.dt = dt
        # self.eligibility_lr = 0.1
        self.refractory_period = 1
        self.AP = np.zeros((self.N, 1))

        self.syn_timer = -np.ones((self.N, self.N))
        self.AP_delayed = np.zeros((self.N, self.N))

        # equilibrium potentials:
        self.V_E = 0
        self.V_I = -80  # equilibrium potential for the inhibitory synapse
        self.EL = -65  # leakage potential, mV

        # critical voltages:
        self.Vth = -55  # threshold after which an AP is fired,   mV
        self.Vr = -70  # reset voltage (after an AP is fired), mV
        self.Vspike = 10

        # define neuron types in the network:
        self.neur_type_mask = np.random.choice([0, 1], self.N, p=[0.5, 0.5]).reshape(*self.AP.shape).astype('float64')

        # first 2 (input) and last 2 (output) neurons are excitatory
        self.neur_type_mask[0] = 1
        self.neur_type_mask[1] = 1
        self.neur_type_mask[-1] = 1
        self.neur_type_mask[-2] = 1
        print('NEUR_TYPE_MASK: {}'.format(self.neur_type_mask))

        # taus
        self.tau = np.zeros((1, self.N))
        self.tau[0, np.where(self.neur_type_mask == 0)[0]] = 10
        self.tau[0, np.where(self.neur_type_mask == 1)[0]] = 20

        self.tau_ampa = 8
        self.tau_nmda = 100
        self.tau_gaba = 8
        # self.postsyn_tau = 0.1
        # self.presyn_tau = 5
        # self.tau_eligibility = 40

        self.V = np.ones((1, self.N)) * self.EL
        self.init_4_1_trial()

        # define weights:
        self.w = np.ones((self.N, self.N)).astype('float') * 0.8
        self.w = self.w + np.random.rand(*self.w.shape) * 0.4

        for i in range(self.N):
            self.w[i, i] = 0
        self.w[:, :2] = 0  # input neurons don't receive synapses
        self.w[:2, -2:] = 0  # output neurons don't listen to input neurons
        self.w[-2:, -2:] = 0  # output neurons don't listen to themselves
        self.w[-2:,:] = 0 # output neurons don't feed back to reservoir
        #         self.w[-2, -1] = 1
        #         self.w[-1, -2] = 1

        self.w_mask = np.ones_like(self.w)
        self.w_mask[:,:2] = 0
        self.w_mask[-2:,:] = 0
        self.w_mask[:,:-2] = 0
#         self.w_mask[2,2] = 0
#         self.w_mask[3,3] = 0
        self.w_mask[:2,-2:] = 0
#         self.w_mask = np.zeros_like(self.w)  # fix all weights except readout connections
#         self.w_mask[2:-2, -2:] = 1
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.imshow(self.w)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(self.w_mask)
        plt.colorbar()
        #         self.w += self.w_mask * np.random.randn(*self.w.shape) * 0.01


        self.i = 0

    def init_4_1_trial(self, stim_type=0):
        self.i = 0                                                      # time step index (integer)
        self.t = np.arange(0, self.T, self.dt)                          # time steps (model time)
        self.V = self.V * 0 + self.EL                                   # membrane potentials. Initialized to leak voltage
        self.AP = self.AP * 0                                           # clear the spikes at the beginning of a trial
        self.ampa = np.zeros((self.N, self.N))                          # set ampa, nmda and gaba conductances to zero
        self.nmda = np.zeros((self.N, self.N))
        self.gaba = np.zeros((self.N, self.N))
        self.eligibility = np.zeros((self.N, self.N))
        # self.postsyn_act = np.zeros((1, self.N))                        # ?????
        # self.presyn_act = np.zeros((self.N, self.N))                    # ?????
        self.in_refractory = np.zeros((1, self.N))
        self.VV = np.zeros((self.N, len(self.t)))                       # voltages N x T
        self.I_EE = np.zeros((self.N, len(self.t)))                     # excitatory currents (N x T)
        self.I_II = np.zeros((self.N, len(self.t)))                     # inhibitory currents (N x T)
        self.AMPA = np.zeros((self.N, self.N, len(self.t)))             # log AMPA matrix N x N x T
        self.NMDA = np.zeros((self.N, self.N, len(self.t)))             # log NMDA matrix N x N x T
        self.GABA = np.zeros((self.N, self.N, len(self.t)))             # log GABA matrix N x N x T    
        # self.POSTSYN = np.zeros((self.N, len(self.t)))                  # ?????
        # self.PRESYN = np.zeros((self.N, self.N, len(self.t)))           # ?????
        # self.ELIGIBILITY = np.zeros((self.N, self.N, len(self.t)))
        self.Ie = np.zeros((self.N, len(t)))                            # ?????
        stim_t = range(int(15 / dt), int(40 / dt))                      # stimuls currents (N x T)
        if stim_type == 0:
            self.Ie[0, stim_t] = 12
            self.Ie[1, stim_t] = 0
        if stim_type == 1:
            self.Ie[0, stim_t] = 0
            self.Ie[1, stim_t] = 12

    def step(self, stimulus=0):

        # AP are spikes registerd at 'postsynaptic' neurons
        # preAP are spikes that have already "arrived" at their target synapses
        preAP = self.AP.dot(np.logical_not(self.AP.T).astype(int))  # turn postsynaptic APs to presynaptic (AP from presynaptic neurons propagate to postsynaptic target neurons)
        inc, out = np.where(preAP == 1)
        self.syn_timer[inc, out] = np.round(1 + np.random.rand() * 2, 2)  # randomized synaptic delay timer
        inc, out = np.where(np.round(self.syn_timer, 4) == 0)
        self.AP_delayed[inc, out] = 1                               # delayed APs 

        self.ampa += (-self.ampa / self.tau_ampa + self.neur_type_mask * self.AP_delayed * self.w) * self.dt
        self.nmda += (-self.nmda / self.tau_nmda + self.neur_type_mask * self.AP_delayed * self.w) * self.dt
        self.gaba += (-self.gaba / self.tau_gaba + (1.0 - self.neur_type_mask) * self.AP_delayed * self.w) * self.dt
        # self.postsyn_act += (-self.postsyn_act / self.postsyn_tau + self.AP.T) * self.dt
        # self.presyn_act += (-self.presyn_act / self.presyn_tau + self.AP_delayed) * self.dt
        # self.eligibility += (self.eligibility_lr * (
        #     self.presyn_act * self.postsyn_act - self.eligibility / self.tau_eligibility)) * self.dt

        self.AP = self.AP * 0
        self.AP_delayed = self.AP_delayed * 0
        where_reset = np.where(self.V.flatten() >= self.Vspike)[0]  # reset from spike
        self.V[:, where_reset] = self.Vr                            # reset from spike
        self.in_refractory[:, where_reset] = self.refractory_period # set refractory timer for each neuron that has spiked
        where_refractory = np.where(self.in_refractory.flatten() > 0)[0] # neurons whose refractory timers are > 0 remain in a refractory state

        # compute total Excitatory and Inhibitory currents arriving at each neuron
        self.I_E = np.sum(-self.ampa * (self.V - self.V_E) - 0.1 * self.nmda * (self.V - self.V_E), 0)
        self.I_I = np.sum(-self.gaba * (self.V - self.V_I), 0)

        # compute the membrane potential change at the current step
        dV = (-(self.V - self.EL) / self.tau + self.I_E + self.I_I + self.Ie[:, self.i]) * self.dt

        dV[0, where_refractory] = 0         # the change is zero for every neuron that is in refractory state
        self.V += dV                        # update the membrane potential

        where_is_AP = np.where(self.V.flatten() > self.Vth)[0]  # detect threshold crossing

        self.V[:, where_is_AP] = self.Vspike    # set the V to spike if the threshold is crossed
        self.AP[where_is_AP, :] = 1             # register a spike for those neurons

        self.I_EE[:, self.i] = self.I_E
        self.I_II[:, self.i] = self.I_I
        self.VV[:, self.i] = self.V
        self.AMPA[:, :, self.i] = self.ampa
        self.NMDA[:, :, self.i] = self.nmda
        self.GABA[:, :, self.i] = self.gaba
        # self.POSTSYN[:, self.i] = self.postsyn_act.flatten()
        # self.PRESYN[:, :, self.i] = self.presyn_act
        # self.ELIGIBILITY[:, :, self.i] = self.eligibility
        self.i += 1
        self.in_refractory -= self.dt

        self.syn_timer -= self.dt
        self.AP_delayed *= 0
        
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    def learn(self, alpha=None, U=None):

#         dw = self.w_mask * alpha * U * np.tanh(self.eligibility)
        dw = self.w_mask * (alpha * U * self.softmax(self.eligibility) + np.random.rand(*self.w.shape)*0.01)
#         dw = self.w_mask * alpha * U * preprocessing.minmax_scale(self.eligibility, axis=0)
        self.w += dw
        negs0, negs1 = np.where(dw<0)
        Sn = np.sum(dw[negs0, negs1])
        pos0, pos1 = np.where(dw>0)
        Sp = np.sum(dw[pos0, pos1])
        if np.any(self.w < 0):
            sig, rec = np.where(self.w < 0)
            self.w[sig, rec] = 0
            print('Negative weight DETECTED, not corrected {}'.format(np.min(self.w)))
        return Sn, Sp


stim = 1
N = 6
T = 100
dt = 0.01
t = np.arange(0, T, dt)
seed = 20


net = Net2(N=N, T=T, dt=dt, seed=222)
W_init = np.copy(net.w)


net.init_4_1_trial(stim_type=stim)
for i in range(len(t)):
    net.step()

