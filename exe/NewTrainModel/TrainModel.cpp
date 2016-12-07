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
#define INPUT_FILE "/u5/z4shang/Documents/research/data/inputSelectedData/train_DA"
#define TEST_FILE "/u5/z4shang/Documents/research/data/inputSelectedData/test_DA"
//#define INPUT_FILE "./onlineAUwithFlag_shuffle_DA"

using namespace std;
#define TYPE 0 // 0:AU, 1:HOG

struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
int TOTAL_NUM;
int Dimension;//3

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
    //data = vector<vector<double>>(TOTAL_AU, vector<double>(0, TOTAL_NUM));
}

void read_problem() {
    cout<<"Reading problem ..."<<endl;
    prob.l = TOTAL_NUM;
    prob.y = new double[TOTAL_NUM];
    prob.x = new svm_node*[TOTAL_NUM]; 
    Dimension = TOTAL_AU;

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

void predict() {
    string iline;
    ifstream input;
    double err = 0;
    int testTotal = 0;
    input.open(string(TEST_FILE) + ".txt");

    if (input.is_open()) {
        while (getline(input, iline)) {
            istringstream iss(iline);
            svm_node* newPredictX = new svm_node[Dimension + 1];
            for (int j = 0; j < Dimension; j++) {
                iss >> newPredictX[j].value;
                newPredictX[j].index = j + 1;
            }
            newPredictX[Dimension].index = -1;
            double original = 0;
            iss >> original;
            err += pow(original - svm_predict(model, newPredictX), 2);
            testTotal++;
            delete newPredictX;
        }
        input.close();
    }
    err = sqrt(err / testTotal);
    cout << "standard err: " << err << " Total: " << testTotal << endl;
}

int main(int argc, char* argv[]) {
    clock_t begin = clock();
    get_total_num();
    read_problem();
    set_param(argv[1], argv[2], argv[3]);

    cout << "Training model ..."<<endl;
    model = svm_train(&prob, &param);
    cout << "After training. Predicting ..." <<endl;

    if (svm_save_model("SVMModel_A", model) != 0) {
        cout << "Failed to save the model :(" << endl;
    }
    predict();

    svm_destroy_param(&param);
    delete prob.y;
    for (int i = 0; i < TOTAL_NUM; i++) {
        delete prob.x[i];
    }
    delete prob.x;

    cout << "Time elasped: " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
    return 0;
}
