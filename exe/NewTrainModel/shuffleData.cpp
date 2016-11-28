#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <dirent.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>

#define TOTAL_AU 17//775//35
#define TRAIN_DIR "/u5/z4shang/Documents/research/data/onlineAUwithFlag"
#define TRAIN_OUTPUT_DIR "/u5/z4shang/Documents/research/data/output/DP"

using namespace std;
#define TYPE 0 // 0:AU, 1:HOG

DIR *dpdf;
struct dirent *epdf;
vector<vector<double>> data;
 
void setY(string fileNum, int& idx, unordered_map<string, int>& timeStamp) {
    string oline;
    ifstream output;
    output.open(string(TRAIN_OUTPUT_DIR) + "/S" + fileNum + "y.txt");

    if (output.is_open()) {
        while (getline(output, oline)) {
            istringstream iss(oline);

            double time;
            iss >> time;
            ostringstream oss;
            oss << std::fixed << std::setfill('0') << setprecision(2) << time;

            if (timeStamp.find(oss.str()) != timeStamp.end()) {
                //iss >> prob.y[idx++];
                double y;
                iss >> data[timeStamp[oss.str()]][TOTAL_AU];
                idx++;
            }
        }
        output.close();
    }
}

void shuffleDataAndStore() {
	unordered_map<string, int> timeStamp;
    //unordered_set<string> timeFrame;
    int out = 0, in = 0, total = 0;
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
                timeStamp.clear();

                for (int i = 0; i < 31; i++)
                    getline(input, iline);

                while (getline(input, iline)) {
                	vector<double> record(TOTAL_AU + 1);
                    istringstream iss(iline); 
                    double time;
                    string temp, inputTime;
                    int success = 0;

                    if (TYPE == 0) {
                        iss >> time;
                        iss >> success;
                        total++;
                        if (!success || time - 0 < 0.01) { //invalid data
                            continue;
                        } else {
                            ostringstream oss;
                            oss << std::fixed << std::setfill('0') << setprecision(2) << time;

                            //cout << "input time : [" << oss.str() << "]"<< endl; 
                            inputTime = oss.str();
                            //timeFrame.insert(oss.str());
                            in++;
                        }

                        for (int j = 0; j < TOTAL_AU; j++) {
                            iss >> record[j];
                        }
                        data.push_back(record);
                        timeStamp[inputTime] = data.size() - 1;
                    } else if (TYPE == 1) { // change!!!!!
                        for (int j = 0; j < TOTAL_AU; j++) {
                            iss >> record[j];
                        }

                        iss >> time;
                        iss >> success;
                        total++;
                        if (!success || time - 0 < 0.01) { //invalid data
                            continue;
                        } else {
                            ostringstream oss;
                            oss << std::fixed << std::setfill('0') << setprecision(2) << time;

                            //cout << "input time : [" << oss.str() << "]"<< endl; 
                            //timeFrame.insert(oss.str());
                            in++;
                        }
                    }
                }
                input.close();

                string s = string(epdf->d_name);
                size_t found = s.find_first_of(".");
                if (found != 0) {
                    string num = string(epdf->d_name).substr(7, found - 7);//Session*.txt
                    cout << "file:" << num << endl;
                    setY(num, out, timeStamp);
                    cout << "y: " << out << endl;
                    cout << "x: " << in << " " << total << endl;
                } else {
                    cout << "skip " << endl;
                }
            }
        }
    }
    closedir(dpdf);
}

void outputData() {
	ofstream myfile;
	myfile.open ("inputData_DP.txt");
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[0].size(); j++) {
			myfile << data[i][j] << " ";
		}
		myfile << endl;
	}
	myfile.close();
}

int main() {
	shuffleDataAndStore();
	cout << data.size() << " " << data[0].size() << endl;
	random_shuffle(data.begin(), data.end());
	outputData();
	return 0;
}