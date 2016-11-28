#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define TOTAL_NUM 1258
#define INPUT_FILE "/u5/z4shang/Documents/research/data/onlineAUwithFlag_shuffle_DV"

using namespace std;

int main() {
	string iline;
    ifstream input;
    input.open(string(INPUT_FILE) + ".txt");

    int count = 0;
    ofstream trainFile;
	trainFile.open ("trainSubData_DV.txt");

    if (input.is_open()) {
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