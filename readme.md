# Parallelization

// compiled with 
// "#pragma omp parallel for" in front of the main loop of ONE time step
// (YOU CAN'T PARALLELIZE CODE OVER TIME)
// g++ -std=gnu++11 -Ofast -ftree-vectorize -march=native -mavx -fopenmp LIF-2.cpp -o lif



# Numba is awesome
## don't 
	- use huge matrices (3d especially) to store data
	- try to 'help' Numba to parallelize the code by creating loops
	- do the same in c++ and #pragma

## numba is even slightly better than super optimized c++ code
	- uses the cores efficiently

## however, if you complile with std=gnu++17 (not 11) you'll beat numba

on a 32 core Intel machine
	3000 neurons - 1 minute per 100 ms of simulation with dt=0.01
on a 128 core AMD ROC node
	3000 neurons - 1 minute per 1000 ms of simulation with dt=0.01
