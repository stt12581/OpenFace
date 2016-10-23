#include "svm.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define TOTAL_NUM 104036//15356
#define TOTAL_AU 18
using namespace std;

struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
struct svm_node *predictX;
struct svm_node *predictX2;
double oriValue;
double oriValue2;
int Total[8] = {15358, 6214, 17022, 17031, 7390, 9042, 14474, 17505};

void read_problem() {
    prob.l = 15358;//  + 6214 + 17022 + 17031;
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l]; 

    int in = 0, out = 0;
    for (int f = 0; f < 1; f++) {//7
        string iline, oline;
        ifstream input;
        if (f < 4)
            input.open("Session" + to_string(2 + f) + ".txt");
        else
            input.open("Session" + to_string(4 + f) + ".txt");

        if (input.is_open()) {
            getline(input, iline);

            for (int i = 0; i < 15358; i++, in++) {
                getline(input, iline);
                istringstream iss(iline);

                prob.x[in] = new svm_node[TOTAL_AU + 1];
                string temp;
                for (int la = 0; la < 413; la++) //396
                    iss >> temp;
                for (int j = 0; j < TOTAL_AU; j++) {
                    char c;
                    iss >> prob.x[in][j].value >> c;
                    if (in == 0) cout << prob.x[in][j].value<<endl;
                    prob.x[in][j].index = j + 1;
                }
            }
            input.close();
        }

        ifstream output;
        if (f < 4)
            output.open("S" + to_string(2 + f) + "y.txt");
        else
            output.open("S" + to_string(4 + f) + "y.txt");
        if (output.is_open()) {
            for (int i = 0; i < 15358; i++, out++) {
                getline(output, oline);
                istringstream iss(oline);

                prob.x[out][TOTAL_AU].index = -1; 
                iss >> prob.y[out] >> prob.y[out];
                //cout << prob.y[out] << endl;
            }
            output.close();
        }
    }
}

void pca() {

}

void set_param() {
    param.svm_type = 3; // e_SVR
    param.kernel_type = LINEAR;
    //param.gamma = (double)1/TOTAL_AU;  // change!!! 1/features
    param.cache_size = 200;
    param.eps = 0.1;
    param.C = 10;
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

int main() {
    pca();
    read_problem();
    set_param();
    model = svm_train(&prob, &param);
cout << "after" <<endl;
    string iline, oline;
    ifstream input;
    input.open("Session10.txt");
    if (input.is_open()) {
        getline(input, iline);

        for (int i = 0; i < 100; i++) {
            getline(input, iline);
            istringstream iss(iline);

            predictX = new svm_node[TOTAL_AU];
            string temp;
            for (int la = 0; la < 413; la++)
                iss >> temp;
            for (int j = 0; j < TOTAL_AU; j++) {
                char c;
                iss >> predictX[j].value >> c;
                predictX[j].index = j + 1;
            }
            cout << svm_predict(model, predictX) <<endl;
        }
        input.close();
    }

    svm_destroy_param(&param);
    delete prob.y;
    delete prob.x;

    return 0;
}
