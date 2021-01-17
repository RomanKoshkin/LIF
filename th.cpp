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
		std::cout << "Elapsed (c++ timer): " << ms << " ms." << std::endl;
	}
};

const int T_ms = 10000;
const int N = 1000;
const int step = 100;

vector<double> dvec;
vector<vector<double>> J;

void sim(int a, int b) {
    for (int j = a; j < b; j++) {
        for (int i = 0; i < N; i++) {
            J[i][j] += 1.0;
        }
    }
}



int main() {

    Timer timer;
    
    for (int i=0; i<N; i++){
        J.push_back(dvec);
        for (int j=0; j<N; j++) {
            J[i].push_back(0.0);
        }
    }

    for (int ti=0; ti < T_ms; ti++) {
        
        #pragma omp parallel for
        for (int i = 0; i < N; i += step) {
            sim(i, i+step);
        }

    }

    return 0;

}