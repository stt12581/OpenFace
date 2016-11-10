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

//#define TOTAL_NUM 910574 // get from get_total_num() - remove every first line
#define TOTAL_AU 17
#define Dimension 3
#define TRAIN_DIR "./data/input/train"
#define TRAIN_OUTPUT_DIR "./data/output/train/DP"
#define TEST_DIR "./data/input/test"
#define TEST_OUTPUT_DIR "./data/output/test/DP"
using namespace std;

struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
DIR *dpdf;
struct dirent *epdf;
stats::pca pca(TOTAL_AU);
int TOTAL_NUM;

vector<vector<double>> data(Dimension, vector<double>(0, TOTAL_NUM)); //convert data

vector<double> eigenvector1;
vector<double> eigenvector2;

void get_total_num() {
    int total = 0;
    int fileNum = 0;
    int count = 0;
    dpdf = opendir(TRAIN_DIR);
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            fileNum++;
            cout << fileNum << epdf->d_name << endl;

            ifstream input;
            string name = string(TRAIN_DIR) + "/" + string(epdf->d_name);
            input.open(name);
            string iline;
            if (input.is_open()) {
                getline(input, iline);

                while (getline(input, iline)) {
                    istringstream iss(iline);

                    float time;
                    string temp;
                    iss >> temp >> time >> temp;
                    iss >> temp;
                    int success = 0;
                    iss >> success;
                    count++;
                    if (!success || abs(time - 0) < 0.01) { //invalid data
                        continue;
                    } else {
                        total++;
                    }
                }
                input.close();
            }
        }
    }
    closedir(dpdf);
    TOTAL_NUM = total;
    cout << total << " " << count << endl;
}

void read_problem() {
    cout<<"Reading problem ..."<<endl;
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            prob.x[i] = new svm_node[Dimension + 1];
            prob.x[i][j].value = data[j][i];
            prob.x[i][j].index = j + 1;
        }
        prob.x[i][Dimension].index = -1;
    }
}

void setY(unordered_set<string>& timeFrame, string fileNum, int& idx) {
    string oline;
    ifstream output;
    output.open(string(TRAIN_OUTPUT_DIR) + "/S" + fileNum + "y.txt");

    if (output.is_open()) {
        while (getline(output, oline)) {
            istringstream iss(oline);
            //prob.x[idx][Dimension].index = -1; 
            float time;
            iss >> time;
            ostringstream oss;
            oss << std::fixed << std::setfill('0') << setprecision(2) << time;

            if (timeFrame.find(oss.str()) != timeFrame.end()) {
                iss >> prob.y[idx++];
            } else {
                //cout << "invalid time frame [" << oss.str() << "]" << endl;
            }
            //cout << prob.y[out] << endl;
        }
        output.close();
    }
}

void set_pca() {
    unordered_set<string> timeFrame;
    prob.l = TOTAL_NUM;
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l]; 

    pca.set_do_bootstrap(true, 100);
    int out = 0;
    int in = 0;
    int total = 0;

    dpdf = opendir(TRAIN_DIR);
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            string iline, oline;
            ifstream input;
            string name = string(TRAIN_DIR) + "/" + string(epdf->d_name);
            if (string(epdf->d_name).size() <= 2) {
                continue;
            }
            input.open(name);

            if (input.is_open()) {
                timeFrame.clear();
                getline(input, iline);

                while (getline(input, iline)) {
                    istringstream iss(iline);

                    vector<double> record(TOTAL_AU);
                    float time;
                    string temp;
                    iss >> temp >> time >> temp;
                    iss >> temp;
                    int success = 0;
                    iss >> success;
                    total++;
                    if (!success || abs(time - 0) < 0.01) { //invalid data
                        continue;
                    } else {
                        ostringstream oss;
                        oss << std::fixed << std::setfill('0') << setprecision(2) << time;

                        //cout << "input time : [" << oss.str() << "]"<< endl; 
                        timeFrame.insert(oss.str());
                        in++;
                    }

                    for (int la = 0; la < 393; la++)
                        iss >> temp;
                    for (int j = 0; j < TOTAL_AU; j++) {
                        char c;
                        iss >> record[j] >> c;
                    }
                    pca.add_record(record);
                }
                input.close();

                string s = string(epdf->d_name);
                size_t found = s.find_first_of(".");
                if (found != 0) {
                    string num = string(epdf->d_name).substr(7, found - 7);//Session*.txt
                    cout << "file:" << num << endl;
                    setY(timeFrame, num, out);
                    cout << "y: " << out << endl;
                    cout << "x: " << in << " " << total << endl;
                } else {
                    cout << "skip " << endl;
                }
            }
        }
    }
    cout<<"PCA solving ..."<<endl;
    pca.solve();

    for (int i = 0; i < Dimension; i++) {
        data[i] = pca.get_principal(i);
    }
    pca.set_num_retained(Dimension);
    closedir(dpdf);
}

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
    double result;
    int in = 0, total = 0;
    ifstream input, output;
    unordered_set<string> timeFrame;

    dpdf = opendir(TEST_DIR);
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            vector<double> predictY;
            string iline, oline;
            string s = string(epdf->d_name);
            string name = string(TEST_DIR) + "/" + string(s);
            if (string(epdf->d_name).size() <= 2) {
                continue;
            }
            input.open(name);

            size_t found = s.find_first_of(".");
            string num = string(epdf->d_name).substr(7, found - 7);//Session*.txt
            cout << "file:" << num << endl;

            if (input.is_open()) {
                timeFrame.clear();
                getline(input, iline);

                while (getline(input, iline)) {
                    istringstream iss(iline);
                    vector<double> predictX(TOTAL_AU, 0);

                    float time;
                    string temp;
                    iss >> temp >> time >> temp;
                    iss >> temp;
                    int success = 0;
                    iss >> success;
                    total++;
                    if (!success || abs(time - 0) < 0.01) { //invalid data
                        continue;
                    } else {
                        ostringstream oss;
                        oss << std::fixed << std::setfill('0') << setprecision(2) << time;

                        //cout << "input time : [" << oss.str() << "]"<< endl; 
                        timeFrame.insert(oss.str());
                        in++;
                    }

                    for (int la = 0; la < 393; la++)
                        iss >> temp;
                    for (int j = 0; j < TOTAL_AU; j++) {
                        char c;
                        iss >> predictX[j] >> c;
                    }

                    vector<double> newPredict = pca.to_principal_space(predictX);
                    svm_node* newPredictX = new svm_node[Dimension + 1];

                    for (int j = 0; j < Dimension; j++) {
                        newPredictX[j].value = newPredict[j];
                        newPredictX[j].index = j + 1;
                        //cout << newPredict[j] << " ";
                    }
                    newPredictX[Dimension].index = -1;
                    //cout << svm_predict(model, newPredictX) << endl;
                    predictY.push_back(svm_predict(model, newPredictX));
                }
                input.close();
            }
            //cout << "in: " << in << endl;

            output.open(string(TEST_OUTPUT_DIR) + "/S" + num + "y.txt");
            if (output.is_open()) {
                int out = 0;
                while (getline(output, oline)) {
                    istringstream iss(oline);
                    double originalY;

                    float time;
                    iss >> time;
                    ostringstream oss;
                    oss << std::fixed << std::setfill('0') << setprecision(2) << time;

                    if (timeFrame.find(oss.str()) != timeFrame.end()) {
                        iss >> originalY;
                        //cout << "predict result: " << originalY << "  " << predictY[out] << endl;
                        result += pow(originalY - predictY[out], 2);
                        cout << "original: " << originalY << " predict: " << predictY[out] << endl;
                        out++;
                    }
                }
                cout << "after reading predict output, size: " << predictY.size() << " out: " << out << endl;
                output.close();
            }
        }
    }

    result = sqrt(result / in); //347911
    cout << "result " << result << "Total" << in << endl;
}

int main() {
    clock_t begin = clock();
    // get_total_num();
    // set_pca();
    // read_problem();
    // set_param();
    // cout << "Training model ..."<<endl;
    // model = svm_train(&prob, &param);
    // cout << "After training. Predicting ..." <<endl;
    
    // if (svm_save_model("SVMModel_P", model) != 0) {
    //     cout << "Failed to save the model :(" << endl;
    // }
    // pca.save("PCAModel_P");
    

    pca.load("PCAModel_P");
    model = svm_load_model("SVMModel_P");
    predict();
    svm_destroy_param(&param);
    delete prob.y;
    delete prob.x;
    cout << "Time elasped: " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
    return 0;
}
