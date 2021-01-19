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

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);

const double sigJ = 0.3;
const double Jmin = 0.0015;
const double Jmax = 1.75;

const double muII = 0.02;
const double muEE = 0.38;
const double muIE = 0.12;
const double muEI = 0.46;

const double PI = 3.14159265359;
const double EPSILON = 0.001;

const double dt = 0.01;

const int N = 3000;
const double T = 100;

const double avg_within_ass = 1.2;

vector<double> F;
vector<double> D;
const double taustf = 50.0;
const double taustd = 250.0;
const double U = 0.98;


double refractory_period = 1.0;

int NE = 2500; //int(N * 0.8);
int NI = N-NE;
vector<double> t;
vector<int> ons;
vector<int> ons2;


vector<double> AP;

vector<double> delayed_spike;
vector<double> spike_timer;

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
// 	return rand()/(RAND_MAX + 1.0);
    return dist(mt);
}

double ngn(){
	double u = dice();
	double v = dice();
	return sqrt(-2.0*log(u))*cos(2.0*PI*v);
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
    
    cout << "embedded assemblies" << endl;
    cout << "STP on" << endl;
    cout << "Stimulating 500:1000 at the beginning" << endl;
    cout << "Stimulating 1500:2000 at 20 ms" << endl;
    cout << "See Plots.ipynb to plot distributions and rasters" << endl;

    
    

    vector<double> row;
    vector<vector<double>> rectangle;

    for (int i = 0; i < N; i++) {
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
    for (int i = 0; i < NI; i++) {
        neur_type_mask.push_back(0);
    }
    for (int i = NI; i < N; i++) {
        neur_type_mask.push_back(1);
    }
    
    // vector<tuple<int, int>> exc_id = where(neur_type_mask, 0, 1);
    exc_id = where1d(neur_type_mask, 0, 1);

    ons = random_choice( exc_id, (int)(0.8 * exc_id.size()) );
    // ons = {63, 12, 22, 96, 97, 60, 54, 33, 92, 15, 88, 78, 91, 10, 41, 51, 45,
    //    57, 47, 77, 70, 98, 31, 11, 93, 76, 29, 46, 49, 65, 64, 48, 18, 62,
    //    37, 14};

    ons = {2554,  762,   32, 2065, 2892, 2561,  412, 2760,  431, 2489,  312,
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
        351, 2008,  580};
    
    ons = {785, 602, 648, 987, 888, 513, 994, 737, 529, 894, 993, 743, 975,
       556, 845, 675, 820, 606, 936, 517, 531, 877, 721, 896, 972, 514,
       591, 832, 621, 563, 808, 976, 647, 886, 685, 681, 729, 594, 502,
       958, 884, 717, 750, 830, 574, 516, 810, 584, 650, 876, 939, 679,
       510, 777, 900, 859, 999, 873, 856, 722, 535, 549, 547, 629, 695,
       858, 689, 747, 842, 989, 819, 995, 905, 631, 833, 706, 836, 590,
       534, 640, 728, 768, 966, 711, 917, 572, 922, 576, 971, 519, 822,
       938, 581, 860, 509, 716, 741, 869, 811, 507, 615, 580, 530, 945,
       879, 700, 795, 826, 897, 546, 930, 595, 784, 916, 672, 853, 940,
       944, 790, 932, 852, 617, 965, 865, 740, 918, 607, 592, 854, 931,
       909, 926, 656, 776, 840, 951, 561, 815, 665, 667, 562, 948, 867,
       727, 781, 604, 823, 610, 806, 709, 698, 919, 545, 964, 528, 933,
       500, 761, 669, 598, 868, 518, 970, 875, 589, 521, 628, 697, 985,
       663, 558, 891, 726, 968, 846, 754, 874, 963, 566, 609, 921, 553,
       738, 783, 583, 657, 779, 508, 520, 524, 704, 532, 882, 613, 841,
       797, 608, 649, 533, 587};
    
    ons2 = {1778, 1957, 1618, 1772, 1643, 1766, 1743, 1605, 1933, 1661, 1789,
       1755, 1974, 1704, 1866, 1686, 1872, 1907, 1776, 1824, 1636, 1628,
       1826, 1601, 1999, 1803, 1982, 1849, 1518, 1683, 1993, 1665, 1950,
       1822, 1947, 1931, 1960, 1523, 1641, 1525, 1881, 1640, 1792, 1573,
       1807, 1634, 1842, 1613, 1749, 1615, 1989, 1860, 1579, 1612, 1725,
       1632, 1566, 1619, 1790, 1794, 1816, 1700, 1659, 1616, 1625, 1513,
       1527, 1701, 1506, 1689, 1921, 1736, 1642, 1739, 1912, 1556, 1799,
       1617, 1589, 1990, 1782, 1943, 1830, 1995, 1727, 1850, 1773, 1939,
       1663, 1885, 1716, 1568, 1674, 1729, 1515, 1886, 1685, 1538, 1738,
       1614};
    
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
                w[i][j] = muII*(1.0 + sigJ*ngn());
                if( w[i][j] < Jmin ) w[i][j] = Jmin;
				if( w[i][j] > Jmax ) w[i][j] = Jmax;
            }  
        }
    }
    // IE
    for (int i = NI; i < N; i ++) {
        for (int j = 0; j < NI; j ++) {
            if (dice() > 0.8) {
                w[i][j] = muIE*(1.0 + sigJ*ngn());
                if( w[i][j] < Jmin ) w[i][j] = Jmin;
				if( w[i][j] > Jmax ) w[i][j] = Jmax;
            }  
        }
    }
    // EI
    for (int i = 0; i < NI; i ++) {
        for (int j = NI; j < N; j ++) {
            if (dice() > 0.8) {
                w[i][j] = muEI*(1.0 + sigJ*ngn());
                if( w[i][j] < Jmin ) w[i][j] = Jmin;
				if( w[i][j] > Jmax ) w[i][j] = Jmax;
            }  
        }
    }
    // EE
    for (int i = NI; i < N; i++) {
        for (int j = NI; j < N; j++) {
            if (dice() > 0.8) {
                w[i][j] = muEE*(1.0 + sigJ*ngn());
                
                // embed an assembly:
				if (i >= 500 && i < 1000 && j >= 500 && j < 1000) {
					w[i][j] = avg_within_ass * (1.0 + sigJ*ngn());			
				}
				if (i >= 1000 && i < 1500 && j >= 1000 && j < 1500) {
					w[i][j] = avg_within_ass * (1.0 + sigJ*ngn());				
				}
				if (i >= 1500 && i < 2000 && j >= 1500 && j < 2000) {
					w[i][j] = avg_within_ass * (1.0 + sigJ*ngn());				
				}
				if (i >= 2000 && i < 2500 && j >= 2000 && j < 2500) {
					w[i][j] = avg_within_ass * (1.0 + sigJ*ngn());				
				}
				if (i >= 2500 && i < 3000 && j >= 2500 && j < 3000) {
					w[i][j] = avg_within_ass * (1.0 + sigJ*ngn());				
				}
                
                if( w[i][j] < Jmin ) w[i][j] = Jmin;
				if( w[i][j] > Jmax ) w[i][j] = Jmax;
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
        spike_timer.push_back(-1.0);
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

void saveWts(vector<vector<double>> Jo, double t) {
	ostringstream ossw;
	ossw << "wts_t_" << int(t) << ".txt";
	string fstrw = ossw.str();
	ofstream ofsw;
	ofsw.open( fstrw.c_str() );
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			ofsw << Jo[i][j] << ",";
		}
		ofsw << endl;
	}
}

int main() {
    
    init();
    saveWts(w, 0);

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
                
                if (t[ts] > 500) {
                    spike_timer[ii] = 0.01 + dice()*0.2;
                    in_refractory[ii] = refractory_period + dice()*0.1;
                } else {
                    spike_timer[ii] = 0.01;
                    in_refractory[ii] = refractory_period;
                }   
                    
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
                    ampa[ii][jj] += (-ampa[ii][jj] / tau_ampa + neur_type_mask[jj]          * F[jj] * D[jj] * delayed_spike[jj] * w[ii][jj]) * dt; // F and D should be delayed too !!!
                    nmda[ii][jj] += (-nmda[ii][jj] / tau_nmda + neur_type_mask[jj]          * F[jj] * D[jj] * delayed_spike[jj] * w[ii][jj]) * dt;
                    gaba[ii][jj] += (-gaba[ii][jj] / tau_gaba + (1.0 - neur_type_mask[jj])  * F[jj] * D[jj] * delayed_spike[jj] * w[ii][jj]) * dt;
//                     ampa[ii][jj] += (-ampa[ii][jj] / tau_ampa + neur_type_mask[jj]          * delayed_spike[jj] * w[ii][jj]) * dt; // F and D should be delayed too !!!
//                     nmda[ii][jj] += (-nmda[ii][jj] / tau_nmda + neur_type_mask[jj]          * delayed_spike[jj] * w[ii][jj]) * dt;
//                     gaba[ii][jj] += (-gaba[ii][jj] / tau_gaba + (1.0 - neur_type_mask[jj])  * delayed_spike[jj] * w[ii][jj]) * dt;
                } else {
                    ampa[ii][jj] += (-ampa[ii][jj] / tau_ampa) * dt;
                    nmda[ii][jj] += (-nmda[ii][jj] / tau_nmda) * dt;
                    gaba[ii][jj] += (-gaba[ii][jj] / tau_gaba) * dt;
                }

                I_E[ii] += -ampa[ii][jj] * (V[ii] - V_E) - 0.1 * nmda[ii][jj] * (V[ii] - V_E);
                I_I[ii] += -gaba[ii][jj] * (V[ii] - V_I);

                // AMPA[ii][jj][ts] = ampa[ii][jj];
                // NMDA[ii][jj][ts] = nmda[ii][jj];
                // GABA[ii][jj][ts] = gaba[ii][jj];
            }
            
            dV[ii] = (-(V[ii] - EL) / tau[ii] + I_E[ii] + I_I[ii] ) * dt;

            if (V[ii] >= Vspike)
                V[ii] = Vr;
        
            if (in_refractory[ii] > EPSILON)
                dV[ii] = 0.0;
            
            V[ii] += dV[ii];
            
            
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (ts == 2000) {
                if (count(ons2.begin(), ons2.end(), ii)) { 
                    V[ii] = Vth + 0.5;
                    in_refractory[ii] = dt;
                }
            }

            if (V[ii] > Vth) {
                V[ii] = Vspike;
                AP[ii] = 1;
                
                // STPonSpike
                F[ii] += U * (1.0 - F[ii]);  // U = 0.6
                D[ii] -= D[ii] * F[ii];
            }
            
            VV[ii][ts] = V[ii];

            // STPonNoSpike
            F[ii] += dt * (U - F[ii])/taustf; // @@ don't forget about hsd!!!
            D[ii] += dt * (1.0 - D[ii])/taustd;

            in_refractory[ii] -= dt;
            spike_timer[ii] -= dt;
        }
            
    }

    save_spts();

    return 0;

}