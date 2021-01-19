# same as above, but jitted with numba (gives a 2x performance boost)

import time
import numpy as np
from LIFclasses import *
from numba import jit, prange
import matplotlib.pyplot as plt

seed = 20
np.random.seed(seed)
N = 3000
T = 100
p = 0.2

@jit (nopython=True, fastmath=True, nogil=True, parallel=True)
def run_numba():
    global N, T, p
    dt = 0.01
    t = np.arange(0, T, dt)
    
    muII = 0.02
    muEE = 0.16
    muIE = 0.12
    muEI = 0.46
    sigJ = 0.3
    
    Jmin = 0.0015
    Jmax = 1.75
    
    avg_within_ass = 0.6
    
    refractory_period = 1
    AP = np.zeros((N, ))
    
    # equilibrium potentials:
    V_E = 0.0
    V_I = -80.0  # equilibrium potential for the inhibitory synapse
    EL = -65.0  # leakage potential, mV

    # critical voltages:
    Vth = -55.0  # threshold after which an AP is fired,   mV
    Vr = -70.0  # reset voltage (after an AP is fired), mV
    Vspike = 10.0
    
    U = 0.98
    taustf = 50.0
    taustd = 250.0
    
    
    
    # define neuron types in the network:
    neur_type_mask = np.zeros_like(AP)
    neur_type_mask[:int(N*0.2)] = 0
    neur_type_mask[int(N*0.2):] = 1  

    # initialize spikes
#     exc_id = np.where(neur_type_mask == 1)[0]
#     ons = np.random.choice(exc_id, int(0.4*len(exc_id)), replace=False)
#     for a in ons:
#         AP[a] = 1

    ons = np.array([2554,  762,   32, 2065, 2892, 2561,  412, 2760,  431, 2489,  312,
        250, 1399, 1385, 1265, 1157, 2899, 2122,  344, 2013, 1665, 2681,
       1954,  369,  842,  737,  464,  438, 1972, 1029,  888,  467,  265,
        279,  553, 2640, 1111, 1164,  391, 1306,  840, 1619, 1591,  139,
        555, 1462,  317,  313, 1497, 2479, 1510,  712, 1050, 2845, 2866,
       2808,  569,  448, 2399,  973, 2484, 2030, 2516, 2382, 2753,   14,
        640,   18,  627,  748, 1946, 1174,  143,  338, 1235,   45,  961,
        216, 2691,  937,  193, 1933, 2041,  307, 1676, 2851, 2673,  759,
       1741, 2225,  680,  207, 1286, 1280, 1631,  500, 2163, 2530, 2474,
       1347, 2087, 2384,  392, 1260, 1536, 2930, 1522, 1719, 2099, 1155,
       2374, 2769, 2098,  181, 2402,  583, 1468, 2265,  852, 2676, 2535,
        476, 2170, 2742, 2720, 1093, 2499, 2486, 1531, 2081,   12, 1820,
       2409,   49,  895,  326, 2305,  659, 1044,  256,  396, 1871,  685,
       1362, 1895,  590, 1049,  644,  155,  815,  335,  389, 1649, 1932,
       2339, 2888, 2263, 1355, 2581, 1319, 2498,  632, 1642, 1720,  540,
       1561, 1601,  735, 2207,  189,  496, 1173, 1315, 1766,  935,  302,
        168,  974, 1239, 2804,  163, 1745, 1892,  986,  269, 1423, 2128,
       2829,  998,  642,  743, 1627,  845, 2795, 1414, 2568, 2939, 1690,
       1981, 2773,  270, 2989, 1456,  403,  987, 2981,  303, 2381, 2653,
       2573,  501, 1086, 1407,  249, 1821,  792, 2033,   33, 1071, 2160,
       2869, 1796,  572,  989, 2429, 1969, 1912, 2849,  509,  355, 2836,
       2093, 2164, 2460, 1091, 1035, 1316, 1143,  171, 2820,  421,  972,
       2400, 2584, 2118,  272,  615,  985, 2847, 1838,  709, 2528, 2838,
        323,  328,  542, 2588, 2800, 1872, 1729, 2657,  843, 2822,   13,
        673, 2096,  480, 2921, 1628,   56,  940,  641,  766, 1860, 1382,
         55, 1753, 2718, 2540,  965, 1212,  948, 1137, 2619, 1476, 2931,
       1160,    4,  984,  918,  214,  524, 2044, 1204, 2834,   53, 1682,
        351, 2008,  580])
    
    for a in ons:
        AP[a] = 1

    # taus
    tau = np.zeros((N, ))
    idx = np.where(neur_type_mask == 0)[0]
    for j in idx:
        tau[j] = 10
    idx = np.where(neur_type_mask == 1)[0]
    for j in idx:
        tau[j] = 20
    

    tau_ampa = 8
    tau_nmda = 100
    tau_gaba = 8
    
    V = np.ones((N, )) * EL


    # define weights:
    w = np.zeros((N, N))
    NE = int(N*0.8)
    NI = N-NE
    # II
    for i in range(NI):
        for j in range(NI):
            if np.random.rand() < p:
                w[i,j] = (1.0 + np.random.randn() * sigJ) * muII
                if w[i][j] < Jmin:
                    w[i][j] = Jmin
                if w[i][j] > Jmax:
                    w[i][j] = Jmax
    # IE
    for i in range(NI,N):
        for j in range(NI):
            if np.random.rand() < p:
                w[i,j] = (1.0 + np.random.randn() * sigJ) * muIE
                if w[i][j] < Jmin: 
                    w[i][j] = Jmin;
                if w[i][j] > Jmax:
                    w[i][j] = Jmax
    # EI
    for i in range(NI):
        for j in range(NI,N):
            if np.random.rand() < p:
                w[i,j] = (1.0 + np.random.randn() * sigJ) * muEI
                if w[i][j] < Jmin: 
                    w[i][j] = Jmin;
                if w[i][j] > Jmax:
                    w[i][j] = Jmax
    # EE
    for i in range(NI,N):
        for j in range(NI,N):
            if np.random.rand() < p:
                w[i,j] = (1.0 + np.random.randn() * sigJ) * muEE
                
                if (i >= 500) & (i < 1000) & (j >= 500) & (j < 1000):
                    w[i][j] = avg_within_ass * (1.0 + sigJ*np.random.randn())
                if (i >= 1000) & (i < 1500) & (j >= 1000) & (j < 1500):
                    w[i][j] = avg_within_ass * (1.0 + sigJ*np.random.randn())
                if (i >= 1500) & (i < 2000) & (j >= 1500) & (j < 2000):
                    w[i][j] = avg_within_ass * (1.0 + sigJ*np.random.randn())
                if (i >= 2000) & (i < 2500) & (j >= 2000) & (j < 2500):
                    w[i][j] = avg_within_ass * (1.0 + sigJ*np.random.randn())
                if (i >= 2500) & (i < 3000) & (j >= 2500) & (j < 3000):
                    w[i][j] = avg_within_ass * (1.0 + sigJ*np.random.randn())
                
                if w[i][j] < Jmin: 
                    w[i][j] = Jmin;
                if w[i][j] > Jmax:
                    w[i][j] = Jmax
                    
                
            
    
    # prohibit self-connections
    for i in range(N):
        w[i, i] = 0

    i = 0
    t = np.arange(0, T, dt)
    V = V * 0 + EL
    
    EPSILON = 0.001
    
    ampa = np.zeros((N, N))
    nmda = np.zeros((N, N))
    gaba = np.zeros((N, N))
    in_refractory = np.zeros((N, )) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------------------------
    VV = np.zeros((N, len(t)))
    FF = np.zeros((N, len(t)))
    DD = np.zeros((N, len(t)))
#     AMPA = np.zeros((N, N, len(t)))
#     NMDA = np.zeros((N, N, len(t)))
#     GABA = np.zeros((N, N, len(t)))
    F = np.ones((N, )) * U
    D = np.ones((N, ))
    dV = np.zeros((N, ))
    I_E = np.zeros((N, ))
    I_I = np.zeros((N, ))
    delayed_spike = np.zeros((N, ))
    spike_timer = -np.ones((N, ))
    
    for i in range(len(t)):
        if i%120==0:
            print(i)
#             AP[7] = 1.0
#             V[7] = Vspike + 5.0

        
        for ii in prange(N):

            if AP[ii] == 1.0:
                
                if (t[i] > 1):
                    spike_timer[ii] = 0.01 + np.random.rand()*0.2
                    in_refractory[ii] = refractory_period + np.random.rand()*0.1
                else:
                    spike_timer[ii] = 0.01
                    in_refractory[ii] = refractory_period;
                
                AP[ii] = 0.0
                
            if np.abs(spike_timer[ii]) < EPSILON:
                delayed_spike[ii] = 1.0
            else:
                delayed_spike[ii] = 0.0

            I_E[ii] = 0.0
            I_I[ii] = 0.0

            for jj in range(N):
                
#                 ampa[ii, jj] += (-ampa[ii, jj] / tau_ampa + neur_type_mask[jj]         * delayed_spike[jj] * w[ii, jj]) * dt
#                 nmda[ii, jj] += (-nmda[ii, jj] / tau_nmda + neur_type_mask[jj]         * delayed_spike[jj] * w[ii, jj]) * dt
#                 gaba[ii, jj] += (-gaba[ii, jj] / tau_gaba + (1.0 - neur_type_mask[jj]) * delayed_spike[jj] * w[ii, jj]) * dt
                
                
                ampa[ii, jj] += (-ampa[ii, jj] / tau_ampa + neur_type_mask[jj]         * F[jj] * D[jj] * delayed_spike[jj] * w[ii, jj]) * dt
                nmda[ii, jj] += (-nmda[ii, jj] / tau_nmda + neur_type_mask[jj]         * F[jj] * D[jj] * delayed_spike[jj] * w[ii, jj]) * dt
                gaba[ii, jj] += (-gaba[ii, jj] / tau_gaba + (1.0 - neur_type_mask[jj]) * F[jj] * D[jj] * delayed_spike[jj] * w[ii, jj]) * dt

                I_E[ii] += -ampa[ii,jj] * (V[ii] - V_E) - 0.1 * nmda[ii,jj] * (V[ii] - V_E)
                I_I[ii] += -gaba[ii,jj] * (V[ii] - V_I)


            dV[ii] = (-(V[ii] - EL) / tau[ii] + I_E[ii] + I_I[ii] ) * dt

            if V[ii] >= Vspike:
                V[ii] = Vr
        
            if in_refractory[ii] > EPSILON:
                dV[ii] = 0.0
            
            V[ii] += dV[ii]
        
            if V[ii] > Vth:
                V[ii] = Vspike
                AP[ii] = 1
                
                # STPonSpike
                F[ii] += U * (1.0 - F[ii]);
                D[ii] -= D[ii] * F[ii];
                
                

            VV[ii, i] = V[ii]
#             AMPA[ii, :, i] = ampa[ii, :]
#             NMDA[ii, :, i] = nmda[ii, :]
#             GABA[ii, :, i] = gaba[ii, :]
            
            # STPonNoSpike
            F[ii] += dt * (U - F[ii])/taustf
            D[ii] += dt * (1.0 - D[ii])/taustd
            
            FF[ii, i] = F[ii]
            DD[ii, i] = D[ii]
            
        
            in_refractory[ii] -= dt
            spike_timer[ii] -= dt
            
    AMPA, NMDA, GABA = 0, 0, 0
    ons = 0

    return w, neur_type_mask, t, AMPA, NMDA, GABA, VV, ons, FF, DD

tt = time.time()
w, neur_type_mask, t, AMPA, NMDA, GABA, VV, ons, FF, DD = run_numba()
print(f'{(time.time() - tt):.2f} s')

x, y = np.where(VV > 0)
plt.plot(y, x, 'bo', ms=3)
# plt.xlim(0, 100)
_ = plt.ylim(0, 3000)
plt.show()


plt.figure(figsize=(5,5))
plt.imshow(w)
plt.show()


