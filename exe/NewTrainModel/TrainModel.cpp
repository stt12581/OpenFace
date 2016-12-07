#include "svm.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <dirent.h>
#include <unordered_set>
#include "pca.h"

#define TOTAL_AU 17//775//35
#define INPUT_FILE "./data/train_DP_PCA"
//#define INPUT_FILE "./data/inputSelectedData_DV"

using namespace std;
#define TYPE 0 // 0:AU, 1:HOG

struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
int TOTAL_NUM;
int Dimension = 7;

void get_total_num() {
    int total = 0;
    string iline;
    ifstream input;
    input.open(string(INPUT_FILE) + ".txt");

    if (input.is_open()) {
        while (getline(input, iline)) {
            total++;
        }
        input.close();
    }

    TOTAL_NUM = total;
}

void set_pca() {
    stats::pca pca(TOTAL_AU);
    vector<double> y(TOTAL_NUM, 0);
    vector<vector<double>> data(Dimension, vector<double>(TOTAL_NUM, 0));

    pca.set_do_bootstrap(true, 100);
    string iline;
    int idx = 0;
    ifstream input;
    input.open(string(INPUT_FILE) + ".txt");
    if (input.is_open()) {
        while (getline(input, iline)) {
            istringstream iss(iline);
            vector<double> record(TOTAL_AU);

            for (int j = 0; j < TOTAL_AU; j++) {
                iss >> record[j];
            }
            iss >> y[idx];
            idx++;
            pca.add_record(record);
        }
        input.close();
    }

    cout << "PCA solving ..." << endl;
    pca.solve();

    for (int i = 0; i < Dimension; i++) {
        data[i] = pca.get_principal(i);
    }
    pca.set_num_retained(Dimension);

    ofstream trainFile;
    trainFile.open ("./data/DV_PCA.txt");
    for (int i = 0; i < TOTAL_NUM; i++) {
        for (int j = 0; j < Dimension; j++) {
            trainFile << data[j][i] << " ";
        }
        trainFile << y[i] << endl;
    }
    trainFile.close();
}

void read_problem() {
    cout<<"Reading problem ..."<<endl;
    prob.l = TOTAL_NUM;
    prob.y = new double[TOTAL_NUM];
    prob.x = new svm_node*[TOTAL_NUM]; 

    string iline;
    ifstream input;
    int i = 0;
    input.open(string(INPUT_FILE) + ".txt");

    if (input.is_open()) {
        while (getline(input, iline)) {
            istringstream iss(iline);
            prob.x[i] = new svm_node[Dimension + 1];
            for (int j = 0; j < Dimension; j++) {
                iss >> prob.x[i][j].value;
                prob.x[i][j].index = j + 1;
            }
            iss >> prob.y[i];
            prob.x[i][Dimension].index = -1;
            i++;
        }
        input.close();
    }
}

void set_param(char* c, char* eps, char* gamma) {
    cout << "param: eps " << atof(eps) << endl;
    param.svm_type = 3; // e_SVR
    param.kernel_type = RBF;
    param.gamma = atof(gamma);  // change!!! 1/features
    param.cache_size = 200;
    param.eps = 0.1;
    param.C = atof(c);
    param.p = atof(eps);

    const char* result = svm_check_parameter(&prob, &param);
    if (result != NULL){
        cout << "result is not null " << endl;
        return;
    }
}

void calculate_standard_err(double *target) {
    double err = 0;
    for (int i = 0; i < TOTAL_NUM; i++) {
        err += pow(target[i] - prob.y[i], 2);
        //cout << "original: " << prob.y[i] << " predict: " << target[i] << endl;
    }
    err = sqrt(err / TOTAL_NUM);
    cout << "standard err: " << err << " Total:" << TOTAL_NUM << endl;
}

int main(int argc, char* argv[]) {
    clock_t begin = clock();
    get_total_num();

    //set_pca();
    read_problem();
    set_param(argv[1], argv[2], argv[3]);

    cout << "Training model ..."<<endl;
    double *target = new double[TOTAL_NUM];
    svm_cross_validation(&prob, &param, 4, target);
    //model = svm_train(&prob, &param);
    cout << "After training. Predicting ..." <<endl;

    calculate_standard_err(target);

    svm_destroy_param(&param);
    delete prob.y;
    for (int i = 0; i < TOTAL_NUM; i++) {
        delete prob.x[i];
    }
    delete prob.x;

    cout << "Time elasped: " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
    return 0;
}
