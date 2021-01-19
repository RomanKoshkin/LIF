# Parallelization

Compiled with `#pragma omp parallel for` in front of the main loop of ONE time step
YOU CAN'T PARALLELIZE CODE OVER TIME

# Compile with 
`g++ -std=gnu++11 -Ofast -ftree-vectorize -march=native -mavx -fopenmp LIF-2.cpp -o lif`
	- On the Mac, OpenMP is not installed by default. Install it separately.

# Numba is awesome

## don't 
	- use huge matrices (3d especially) to store data
	- try to 'help' Numba to parallelize the code by creating loops
	- do the same in c++ and #pragma

## numba is even slightly better than super optimized c++ code
	- uses the cores efficiently

## however, if you complile with std=gnu++17 (not 11) you'll beat numba

- on a 32 core Intel machine
	-- 3000 neurons - 1 minute per 100 ms of simulation with dt=0.01
- on a 128 core AMD ROC node
	-- 3000 neurons - 1 minute per 1000 ms of simulation with dt=0.01


# Most recent code

## LIF-4.cpp
	- latest with STP
	- embedded assemblies

## LIF_numba_parallel-2.py
	- same as LIF-4.cpp
	- almost as fast
	- easier to debug

## Plots.ipynb
	- let's you plot the rasters and weights from LIF-4.cpp

## Illustration of the LIF network DONT CHANGE.ipynb
	- essentially a tutorial, where you can plot AMPA, NMDA, GABA and other dynamics of a network of LIF neurons. The code is essentially the same as LIF_numba_parallel-2.py



