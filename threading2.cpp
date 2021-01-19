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

// to complile
// g++ -std=gnu++11 -O3 -dynamiclib -ftree-vectorize -march=native -mavx bmm_7_haga.cpp -o ./bmm.dylib
// sudo /usr/bin/g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx bmm_6_haga.cpp -o ./bmm.dylib
// icc -std=gnu++11 -O3 -shared -fPIC bmm_5_haga.cpp -o ./bmm.dylib

using namespace std;

struct Timer
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	std::chrono::duration<float> duration;

	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	/* when the function where this object is created returns,
	this object must be destroyed, hence this destructor is called */
	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		float ms = duration.count() * 1000.0f;
		std::cout << "Elapsed: " << ms << " ms." << std::endl;
	}
};

int N = 3000;
int step = 100;
vector<double> dvec;
vector< vector<double> > J;

std::random_device m_randomdevice;
std::mt19937 m_mt;
std::uniform_real_distribution<double> dist(0.0, 1.0);

void calc(int a, int b) {
    for(int jj = a; jj < b; jj++){
        for(int ii = 0; ii < N; ii++){
            J[ii][jj] += 1.0; // * dist(m_mt);
        }
    }
}

int main(int argc, char **argv){
	


    for(int i = 0; i < N; i++){
		J.push_back(dvec);
		for(int j = 0; j < N; j++){
			J[i].push_back(0.0);
        }
    }

    Timer timer;

    for(int t = 0; t < 500; t ++) {

        // 40100 ms without parfor
        // 1869 ms  with parfor
        // 47000 ms  with parfor (and random function)
        #pragma omp parallel for
        for(int i = 0; i < N; i += step) {
            calc(i, i+step);
        }

    }
	return 0;
}