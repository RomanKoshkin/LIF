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


using namespace std;

const double EPSILON = 0.001;
const double dt = 0.01;

const int N = 300;
const double T = 30;
const int n_cores = 30;
const int stride = 10;
double refractory_period = 1.0;

int NE = int(N * 0.8);
int NI = N-NE;
vector<double> t;
vector<int> ons; 


vector<vector<double>> syn_timer; //-np.ones((N, N))
vector<double> AP;

vector<vector<double>> AP_delayed; // zeros 

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
vector<vector<double>> preAP;
vector<vector<vector<double>>> AMPA;
vector<vector<vector<double>>> NMDA;
vector<vector<vector<double>>> GABA;
double I_E;
double I_I;
vector<double> dV;
vector<vector<double>> w;
vector<double> V;

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


    // syn timer
    for (int i = 0; i < N; i ++) {
        syn_timer.push_back(row);
        for (int j = 0; j < N; j ++) {
            syn_timer[i].push_back(-1.0);
        }
    }

    // preAP
    for (int i = 0; i < N; i++) {
        preAP.push_back(row);
        for (int j = 0; j < N; j++) {
            preAP[i].push_back(0.0);
        }
    }

    //AP_delayed
    for (int i = 0; i < N; i++) {
        AP_delayed.push_back(row);
        for (int j = 0; j < N; j++) {
            AP_delayed[i].push_back(0.0);
        }
    }

    for (int i = 0; i < N; i++) {
        dV.push_back(0.0);
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
                w[i][j] = dice() + 0.1;
            }  
        }
    }
    // IE
    for (int i = NI; i < N; i ++) {
        for (int j = 0; j < NI; j ++) {
            if (dice() > 0.8) {
                w[i][j] = dice() + 1.2;
            }  
        }
    }
    // EI
    for (int i = 0; i < NI; i ++) {
        for (int j = NI; j < N; j ++) {
            if (dice() > 0.8) {
                w[i][j] = dice() + 0.7;
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

    // for (int i = 0; i < N; i ++) {
    //     if (find(ons.begin(), ons.end(), i) != ons.end()) {
    //         in_refractory.push_back(refractory_period);
    //     } else {
    //         in_refractory.push_back(0.0);
    //     }
    // }

    for (int i = 0; i < N; i ++) {
        in_refractory.push_back(0.0);
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

    for (int i = 0; i < N; i ++) {
        AMPA.push_back(rectangle);
    }
    
    for (int i = 0; i < N; i ++) {
        NMDA.push_back(rectangle);
    }
    
    for (int i = 0; i < N; i ++) {
        GABA.push_back(rectangle);
    }

}

void on_core(int k, int ts) {

    for (int i = k; i < k+stride; i++) {
        
        dV[i] = 0.0;
        I_E = 0.0;
        I_I = 0.0;

        for (int j = 0; j < N; j++) {
            
            ampa[i][j] += (-ampa[i][j] / tau_ampa + neur_type_mask[i] * AP_delayed[i][j] * w[i][j]) * dt;
            nmda[i][j] += (-nmda[i][j] / tau_nmda + neur_type_mask[i] * AP_delayed[i][j] * w[i][j]) * dt;
            gaba[i][j] += (-gaba[i][j] / tau_gaba + (1.0 - neur_type_mask[i]) * AP_delayed[i][j] * w[i][j]) * dt;

            AMPA[i][j][ts] = ampa[i][j];
            NMDA[i][j][ts] = nmda[i][j];
            GABA[i][j][ts] = gaba[i][j];
 
            AP_delayed[i][j] = AP_delayed[i][j] * 0.0;

            I_E += -ampa[j][i] * (V[i] - V_E) - 0.1 * nmda[j][i] * (V[i] - V_E);
            I_I += -gaba[j][i] * (V[i] - V_I);           

        }

        if (V[i] >= Vspike) {
            V[i] = Vr;
            in_refractory[i] = refractory_period;
        }

        AP[i] = AP[i] * 0.0;
        dV[i] = -((V[i] - EL) / tau[i] + I_E + I_I) * dt; //  REMOVED Ie
            
        if (in_refractory[i] <= 0) {
            V[i] += dV[i];
        }
            
        if (V[i] > Vth) {
            V[i] = Vspike;
            AP[i] = 1;
        }

        VV[i][ts] = V[i];
        in_refractory[i] -= dt;
    }
}

int main() {
    
    init();

    auto start = std::chrono::steady_clock::now();

    for (int ts = 0; ts < t.size(); ts++) {
        if (ts%100 == 0){
            
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            cout << t[ts] << " " << elapsed_seconds.count() << endl;
            start = std::chrono::steady_clock::now();
            
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (AP[j] == 0) {
                    preAP[i][j] = AP[i];
                } else {
                    preAP[i][j] = 0;
                }
                if (preAP[i][j] == 1) {
                    syn_timer[i][j] = 2.0;
                }
                if (syn_timer[i][j] < EPSILON) {
                    AP_delayed[i][j] = 1.0;
                }
            }
        }

        #pragma omp parallel for
        for (int k = 0; k < n_cores * stride; k += stride) {
            on_core(k, ts);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                syn_timer[i][j] -= dt;
                AP_delayed[i][j] *= 0.0;
            }
        }
       
    }

    save_spts();

    return 0;

}