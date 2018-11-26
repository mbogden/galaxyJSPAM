#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

using namespace std;


int main(int argc, char *argv[]){

    float gMin = 1, gMax = 10, gInc = 1;
    float rMin = 1, rMax = 10, rInc = 1;
    float nMin = 1, nMax = 10, nInc = 1;
    float gw, rv, nv;

    int startNum = 1;
    int num = startNum;

    string path = argv[1];

    cout << endl;

    for ( gw = gMin; gw <= gMax; gw += gInc ){
        for ( rv = rMin; rv <= gMax; rv += rInc ){
            for ( nv = nMin; nv <= nMax; nv += nInc ){

                stringstream outStrm,nameStrm;

                nameStrm << path << "param";
                nameStrm << setfill('0') << setw(4) << num << ".txt";

                outStrm << "gaussian_size 30\n";
                outStrm << "gaussian_weight " << gw << endl;
                outStrm << "radial_constant " << rv << endl;
                outStrm << "norm_value " << nv << endl;
                outStrm << "image_rows 1000\n";
                outStrm << "image_cols 1000\n";

                //cout << nameStrm.str().c_str() << endl;
                //cout << outStrm.str().c_str() << endl;

                ofstream paramFile(nameStrm.str().c_str());
                paramFile << outStrm.str();
                paramFile.close();

                num++;

            }
        }
    }

    return 0;
}
