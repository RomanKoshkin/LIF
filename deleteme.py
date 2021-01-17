import numpy as np

N = 5

AP = np.zeros((N, 1))
# AP[[2],0] = 1

preAP1 = np.zeros((N,N))
preAP = AP.dot(np.logical_not(AP.T)*1.0)

for i in range(N):
    for j in range(N):
        preAP1[i,j] = AP[i] if AP[j] == 0.0 else 0.0


pass

