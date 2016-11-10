#include "svm.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <unordered_set>
#include <dirent.h>
#include "pca.h"

#define TOTAL_NUM 15358 + 6214//72057
#define TOTAL_AU 17
#define Dimension 2
#define NUM_FILE 2
using namespace std;

struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
DIR *dpdf;
struct dirent *epdf;
int Total[8] = {15358, 6214, 17022, 17031, 7390, 9042, 14474, 17505};
unordered_set<double> timeFrame;

void get_total_num() {
	int total = 0;
	int fileNum = 0;
	dpdf = opendir("./data/input");
	if (dpdf != NULL){
   		while (epdf = readdir(dpdf)){
   			fileNum++;
   			cout << fileNum << epdf->d_name << endl;
   			//cout << epdf->d_name << endl;

	        ifstream input;
	        string name = "./data/input/" + string(epdf->d_name);
	        input.open(name);
	        string iline;
	        if (input.is_open()) {
	            getline(input, iline);

	            while (getline(input, iline)) {
	                istringstream iss(iline);
	                string temp;
	                for (int la = 0; la < 3; la++)
	                    iss >> temp;
	                int success = 0;
	                iss >> success;
	                if (success) {
	                	total++;
	                }
	            }
	            input.close();
	        }
	    }
    }
    closedir(dpdf);
    cout << total << " " << fileNum << endl;
}

// void read_problem() {
//     prob.l = TOTAL_NUM;
//     prob.y = new double[prob.l];
//     prob.x = new svm_node*[prob.l]; 

//     int in = 0, out = 0;
//     dpdf = opendir("./data/input");
// 	if (dpdf != NULL){
//    		while (epdf = readdir(dpdf)){
// 	      	printf("Filename: %s",epdf->d_name);
// 	      // std::cout << epdf->d_name << std::endl;


// 	        string iline, oline;
// 	        ifstream input;
// 	        input.open(epdf->d_name);

// 	        if (input.is_open()) {
// 	            getline(input, iline);

// 	            for (int i = 0; i < Total[f]; i++, in++) {
// 	                getline(input, iline);
// 	                istringstream iss(iline);

// 	                prob.x[in] = new svm_node[TOTAL_AU + 1];
// 	                string temp;
// 	                for (int la = 0; la < 396; la++)
// 	                    iss >> temp;
// 	                for (int j = 0; j < TOTAL_AU; j++) {
// 	                    char c;
// 	                    iss >> prob.x[in][j].value >> c;
// 	                    prob.x[in][j].index = j + 1;
// 	                }
// 	            }
// 	            input.close();
// 	        }


// 	        //output

// 	    }
//     }
//     closedir(dpdf);
// }

//     string oline;
//     for (int f = 0; f < NUM_FILE; f++) {
//         ifstream output;
//         if (f < 4)
//             output.open("S" + to_string(2 + f) + "y.txt");
//         else
//             output.open("S" + to_string(4 + f) + "y.txt");
//         if (output.is_open()) {
//             for (int i = 0; i < Total[f]; i++, out++) {
//                 getline(output, oline);
//                 istringstream iss(oline);

//                 prob.x[out][TOTAL_AU].index = -1; 
//                 iss >> prob.y[out] >> prob.y[out];
//                 //cout << prob.y[out] << endl;
//             }
//             output.close();
//         }
//     }
// }

void set_param() {
    param.svm_type = 3; // e_SVR
    param.kernel_type = RBF;
    param.gamma = (double)1 / Dimension;  // change!!! 1/features
    param.cache_size = 200;
    param.eps = 0.1;
    param.C = 1;
    param.p = 0.1;
    //param.shrinking = 1;
    //param.probability = 0;
    //param.nr_weight = 0;

    const char* result = svm_check_parameter(&prob, &param);
    if (result != NULL){
        cout << "result is not null " << endl;
        return;
    }
}

void predict() {
    string iline, oline;
    double result;
    ifstream input, output;

    for (int f = 6; f < 8; f++) {//7
        string iline, oline;
        if (f < 4) {
            input.open("Session" + to_string(2 + f) + ".txt");
            output.open("S" + to_string(2 + f) + "y.txt");
        }
        else {
            input.open("Session" + to_string(4 + f) + ".txt");
            output.open("S" + to_string(4 + f) + "y.txt");
        }

	    if (input.is_open() && output.is_open()) {
    	    getline(input, iline);

        	for (int i = 0; i < Total[f]; i++) {
            	getline(input, iline);
            	getline(output, oline);
        	    istringstream iss(iline);
            	istringstream oss(oline);

            	svm_node* predictX = new svm_node[TOTAL_AU + 1];
            	string temp;
            	for (int la = 0; la < 396; la++)
                	iss >> temp;

            	for (int j = 0; j < TOTAL_AU; j++) {
                	char c;
                	iss >> predictX[j].value >> c;
                	predictX[j].index = j + 1;
            	}
            	predictX[Dimension].index = -1;

            	double predictY = svm_predict(model, predictX);
            	double originalY;
            	oss >> originalY >> originalY;
            	cout << originalY << "  " << predictY << endl;
            	result += pow(originalY - predictY, 2);
        	}
        	input.close();
        	output.close();
        }
    }
    result = sqrt(result / (Total[6] + Total[7]));
    cout << "result " << result << endl;
}

int main() {
	clock_t begin = clock();
	get_total_num();
    /*read_problem();
    set_param();
    model = svm_train(&prob, &param);
cout << "after" <<endl;
	predict();

    svm_destroy_param(&param);
    delete prob.y;
    delete prob.x;*/
    cout << "Time elasped: " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
    return 0;
}