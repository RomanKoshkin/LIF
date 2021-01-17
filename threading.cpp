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

class Model {
    public:
        int N;

        vector<double> dvec;
        vector< vector<double> > J;
        
        Model(int); // constructor
        double dice();
        void sim(int, int);


    private:
        // init random number generator (see Rand.cpp for explanation and example)
        std::random_device m_randomdevice;
        std::mt19937 m_mt;
};

double Model::dice(){
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(m_mt);
}

// class construction _definition_. Requires no type specifiction.
Model::Model(int N): m_mt(m_randomdevice()) {

	for(int i = 0; i < N; i++){
		J.push_back(dvec);
		for(int j = 0; j < N; j++){
			J[i].push_back(0.0);
        }
    }
}

void Model::sim(int a, int b) {
    // std::cout << a << " " << b << std::endl;
    for(int j = a; j < b; j++){
        for(int i = 0; i < N; i++){
            J[i][j] += 1.0; // * dice();
        }
    }
}

int main(int argc, char **argv){
	
    int N = 3000;
    int step = 100;

	Model m(N);
    Timer timer;

    for(int t = 0; t < 500; t ++) {

        // #pragma omp parallel for
        for(int i = 0; i < N; i += step) {
            // std::cout << a << " " << b << std::endl;
            m.sim(i, i+step);
        }

    }
	return 0;
}