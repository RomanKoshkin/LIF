#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <set>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm> 
#include <random>
#include <tuple>
#include <math.h>

// compiled with 
// "#pragma omp parallel for" in front of the main loop of ONE time step
// (YOU CAN'T PARALLELIZE CODE OVER TIME)
// g++ -std=gnu++11 -Ofast -ftree-vectorize -march=native -mavx -fopenmp LIF-2.cpp -o lif

using namespace std;

double a;

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);

double dice() {
// 	return rand()/(RAND_MAX + 1.0);
    return dist(mt);
}

int main() {
    for(int i=0; i < 20; i++) {
        a = dice();
    }
    return 0;
}