#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define TOTAL_NUM 12584
#define INPUT_FILE "./data/data_DA_PCA"

using namespace std;

int main() {
	string iline;
    ifstream input;
    input.open(string(INPUT_FILE) + ".txt");

    int count = 0;
    ofstream trainFile;
	trainFile.open ("./data/trainSubData_DA_PCA.txt");

    if (input.is_open()) {
        cout << "ga" << endl;
        while (getline(input, iline) && count < TOTAL_NUM) {
            trainFile << iline;
            trainFile << endl;
            count++;
        }
        input.close();
        trainFile.close();
    }
	return 0;
}