#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define TOTAL_NUM 1258
#define INPUT_FILE "/u5/z4shang/Documents/research/data/inputSelectedData/inputSelectedData_DP"

using namespace std;

int main() {
	string iline;
    ifstream input;
    input.open(string(INPUT_FILE) + ".txt");

    int count = 0;
    ofstream trainFile;
	trainFile.open ("train_DP.txt");

    if (input.is_open()) {
        while (getline(input, iline)) {
            if (count < 50341) {
                trainFile << iline;
                trainFile << endl;
            }
            count++;
        }
        input.close();
        trainFile.close();
    }
	return 0;
}