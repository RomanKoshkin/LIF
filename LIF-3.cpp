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
#include <cmath>

// compiled with 
// "#pragma omp parallel for" in front of the main loop of ONE time step
// (YOU CAN'T PARALLELIZE CODE OVER TIME)
// g++ -std=gnu++11 -Ofast -ftree-vectorize -march=native -mavx -fopenmp LIF-2.cpp -o lif

using namespace std;

const double EPSILON = 0.001;
const double dt = 0.01;

const int N = 3000;
const double T = 100;

double STP;
vector<double> F;
vector<double> D;
const double taustf = 235.0;
const double taustd = 100.0;
const double U = 0.78;


double refractory_period = 1.0;

int NE = int(N * 0.8);
int NI = N-NE;
vector<double> t;
vector<int> ons; 


vector<double> AP;

vector<double> delayed_spike;

const double V_E = 0.0;
const double V_I = -80.0;
const double EL = -65.0;

// critical voltages:
const double Vth = -55.0;         // threshold after which an AP is fired,   mV
const double Vr = -70.0;          // reset voltage (after an AP is fired), mV
const double Vspike = 10.0;

// define neuron types in the network:
vector<int> neur_type_mask;
vector<int> exc_id;
vector<double> tau;

const double tau_ampa = 8.0;
const double tau_nmda = 100.0;
const double tau_gaba = 8.0;

vector<vector<double>> ampa;
vector<vector<double>> nmda;
vector<vector<double>> gaba;
vector<double> in_refractory;
vector<vector<double>> VV;
vector<vector<double>> Ie;
vector<vector<vector<double>>> AMPA;
vector<vector<vector<double>>> NMDA;
vector<vector<vector<double>>> GABA;
vector<double> I_E;
vector<double> I_I;
vector<double> dV;
vector<vector<double>> w;
vector<double> V;

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

// initialize spikes
template <class A, class B>
vector<tuple<int, int>> where (vector<vector<A>> arr, B val, int test) {
    
    vector<tuple<int, int>> whid;
    int I = arr.size();
    int J = arr[0].size();
    
    if (test == 0) {
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                if (arr[i][j] == val){
                    whid.push_back(make_tuple(i,j));
                }  
            }
        }
    }
    if (test == -1) {
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                if (arr[i][j] < val){
                    whid.push_back(make_tuple(i,j));
                }  
            }
        }
    }
    if (test == 1) {
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                if (arr[i][j] > val){
                    whid.push_back(make_tuple(i,j));
                }  
            }
        }
    }
    return whid;
}

template <class A, class B>
vector<int> where1d (vector<A> arr, B val, int test) {
    
    vector<int> whid;
    int I = arr.size();
    
    if (test == 0) {
        for (int i = 0; i < I; i++) {
            if (arr[i] == val){
                whid.push_back(i);
            }  
        }
    }
    if (test == -1) {
        for (int i = 0; i < I; i++) {
            if (arr[i] < val){
                whid.push_back(i);
            }  
        }
    }
    if (test == 1) {
        for (int i = 0; i < I; i++) {
            if (arr[i] > val){
                whid.push_back(i);
            }  
        }
    }
    return whid;
}

template <class T>
vector<T> random_choice(vector<T> samples, int outputSize) {

    vector<T> vec(outputSize);

    vector<double> probabilities;
    for (int i = 0; i < samples.size(); i++) {
        probabilities.push_back((double) 1/samples.size());
    }
    
    std::default_random_engine generator;
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

    vector<int> indices(vec.size());
    std::generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });

    std::transform(indices.begin(), indices.end(), vec.begin(), [&samples](int index) { return samples[index]; });

    return vec;
}

double dice() {
	return rand()/(RAND_MAX + 1.0);
}

void save_spts() {
	ostringstream ossw;
	ossw << "spts.txt";
	string fstrw = ossw.str();
	ofstream ofsw;
	ofsw.open( fstrw.c_str() );
	for(int i = 0; i < N; i++){
		for(int j = 0; j < t.size(); j++){
            if (VV[i][j] > Vth + 0.1){
                ofsw << i << "," << j*dt << endl;
            }	
		}
	}
}

void init() {

    vector<double> row;
    vector<vector<double>> rectangle;

    for (int i = 0; i < NE; i++) {
		F.push_back(U);
        D.push_back(1.0);
	}

    for (int i = 0; i < N; i++) {
        dV.push_back(0.0);
    }

    for (int i = 0; i < N; i++) {
        delayed_spike.push_back(0.0);
    }

    for (int i = 0; i < N; i++) {
        I_E.push_back(0.0);
    }

    for (int i = 0; i < N; i++) {
        I_I.push_back(0.0);
    }

    // times
    for (double i = 0; i < T; i += dt) {
        t.push_back(i);
    }

    // AP vector
    for (int i = 0; i < N; i++) {
        AP.push_back(0.0);
    }
    
    // neur type mask
    for (int i = 0; i < 10; i++) {
        neur_type_mask.push_back(0);
    }
    for (int i = 10; i < N; i++) {
        neur_type_mask.push_back(1);
    }
    
    // vector<tuple<int, int>> exc_id = where(neur_type_mask, 0, 1);
    exc_id = where1d(neur_type_mask, 0, 1);

    // !!!!!!!!! sample without REPLACEMENT
    // ons = random_choice( exc_id, (int)(0.4 * exc_id.size()) );
    ons = {63, 12, 22, 96, 97, 60, 54, 33, 92, 15, 88, 78, 91, 10, 41, 51, 45,
       57, 47, 77, 70, 98, 31, 11, 93, 76, 29, 46, 49, 65, 64, 48, 18, 62,
       37, 14};
    for (int i: ons) {
        AP[i] = 1;
    }

    // # taus
    for (int i = 0; i < N; i++) {
        if (neur_type_mask[i] == 1) {
            tau.push_back(20.0);
        }
        else {
            tau.push_back(10.0);
        }
    }

    for (int i = 0; i < N; i++) {
        V.push_back(EL);
    }

    // define weights:
    for (int i = 0; i < N; i ++) {
        w.push_back(row);
        for (int j = 0; j < N; j ++) {
            w[i].push_back(0.0);
        }
    }
    // II
    for (int i = 0; i < NI; i ++) {
        for (int j = 0; j < NI; j ++) {
            if (dice() > 0.8) {
                w[i][j] = dice() * 0.3;
            }  
        }
    }
    // IE
    for (int i = NI; i < N; i ++) {
        for (int j = 0; j < NI; j ++) {
            if (dice() > 0.8) {
                w[i][j] = dice() + 1.4;
            }  
        }
    }
    // EI
    for (int i = 0; i < NI; i ++) {
        for (int j = NI; j < N; j ++) {
            if (dice() > 0.8) {
                w[i][j] = dice() * 1.7;
            }  
        }
    }
    // EE
    for (int i = NI; i < N; i++) {
        for (int j = NI; j < N; j++) {
            if (dice() > 0.8) {
                w[i][j] = dice() + 0.1;
            }
        }
    }
    // prohibit self-connections
    for (int i = 0; i < N; i++) {
        w[i][i] = 0.0;
    }

    
    // define conductances:
    for (int i = 0; i < N; i++) {
        ampa.push_back(row);
        for (int j = 0; j < N; j++) {
            ampa[i].push_back(0.0);
        }
    }
    
    for (int i = 0; i < N; i ++) {
        nmda.push_back(row);
        for (int j = 0; j < N; j ++) {
            nmda[i].push_back(0.0);
        }
    }
    
    for (int i = 0; i < N; i ++) {
        gaba.push_back(row);
        for (int j = 0; j < N; j ++) {
            gaba[i].push_back(0.0);
        }
    }

    for (int i = 0; i < N; i ++) {
        in_refractory.push_back(-1.0);
    }

    for (int i = 0; i < N; i ++) {
        VV.push_back(row);
        for (int j = 0; j < t.size(); j ++) {
            VV[i].push_back(0.0);
        }
    }

    for (int i = 0; i < N; i ++) {
        Ie.push_back(row);
        for (int j = 0; j < t.size(); j ++) {
            Ie[i].push_back(0.0);
        }
    }

    for (int i = 0; i < N; i ++) {
        rectangle.push_back(row);
        for (int j = 0; j < t.size(); j ++) {
            rectangle[i].push_back(0.0);
        }
    }

    // for (int i = 0; i < N; i ++) {
    //     AMPA.push_back(rectangle);
    // }
    
    // for (int i = 0; i < N; i ++) {
    //     NMDA.push_back(rectangle);
    // }
    
    // for (int i = 0; i < N; i ++) {
    //     GABA.push_back(rectangle);
    // }

}

int main() {
    
    init();

    auto start = std::chrono::steady_clock::now();
    
    Timer timer;
    for (int ts = 0; ts < t.size(); ts++) {
        
        if (ts%1000 == 0){
            
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            cout << t[ts] << " " << elapsed_seconds.count() << endl;
            start = std::chrono::steady_clock::now();
            
        }
        
        #pragma omp parallel for
        for (int ii = 0; ii < N; ii++) {

            if (AP[ii] == 1.0){                                  
                in_refractory[ii] = refractory_period + dice();     
                AP[ii] = 0.0;
            }                         

            if (abs(in_refractory[ii]) < EPSILON) {
                delayed_spike[ii] = 1.0;
            } 
            else {
                delayed_spike[ii] = 0.0;
            }

            I_E[ii] = 0.0;
            I_I[ii] = 0.0;

            for (int jj = 0; jj < N; jj++) {
                if (w[ii][jj] > 0) {
                    STP = F[jj] * D[jj];
                    ampa[ii][jj] += (-ampa[ii][jj] / tau_ampa + neur_type_mask[ii] * STP * delayed_spike[jj] * w[ii][jj]) * dt;
                    nmda[ii][jj] += (-nmda[ii][jj] / tau_nmda + neur_type_mask[ii] * STP *delayed_spike[jj] * w[ii][jj]) * dt;
                    gaba[ii][jj] += (-gaba[ii][jj] / tau_gaba + (1.0 - neur_type_mask[ii]) * STP * delayed_spike[jj] * w[ii][jj]) * dt;
                }

                I_E[ii] += -ampa[ii][jj] * (V[ii] - V_E) - 0.1 * nmda[ii][jj] * (V[ii] - V_E);
                I_I[ii] += -gaba[ii][jj] * (V[ii] - V_I);

                VV[ii][ts] = V[ii];
                // AMPA[ii][jj][ts] = ampa[ii][jj];
                // NMDA[ii][jj][ts] = nmda[ii][jj];
                // GABA[ii][jj][ts] = gaba[ii][jj];
            }
            
            dV[ii] = (-(V[ii] - EL) / tau[ii] + I_E[ii] + I_I[ii] ) * dt;

            if (V[ii] >= Vspike)
                V[ii] = Vr;
        
            if (in_refractory[ii] > 0)
                dV[ii] = 0.0;
            
            V[ii] += dV[ii];
        
            if (V[ii] > Vth) {
                V[ii] = Vspike;
                AP[ii] = 1;
                
                // STPonSpike
                F[ii] += U * (1.0 - F[ii]);  // U = 0.6
                D[ii] -= D[ii] * F[ii];
            }
            
            // STPonNoSpike
            F[ii] += dt * (U - F[ii])/taustf; // @@ don't forget about hsd!!!
            D[ii] += dt * (1.0 - D[ii])/taustd;

            in_refractory[ii] -= dt;
        }
            
    }

    save_spts();

    return 0;

}