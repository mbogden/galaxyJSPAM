#ifndef



#include <iostream>
#include <string>
#define appArray(state) ((state == 'i') ? (initialData) : (finalData))
using namespace std;

double initialData[][3];
double finalData[][3];
//int    n1;                   // Number of particles in galaxy 1
//int    n2;                   // Number of particles in galaxy 2
int    n;                    // Number of particles
char   state;                 // Either 'i' or 'f'

extern "C" {
	void in_to_cpp(double[][*],
			       double&[][*],
				   double&[][*],
				   char,
				   int);
}

void readFromASCIIFile(ifstream&, double&[][3], char state);

int read(int readMethod, char state) {
	/*------------------------------------------------------------------------*/
	/*------------------------------------------------------------------------*/
	switch (readMethod)
	{
		case 1:
			readFromASCIIFile(asciiFile, appArray, state);
			break;
		//case 2:
		//	readFromFortran90()
		default:
			cout << "Invalid read method...";
			return 1;
	}

	return 0;
}

void readFromASCIIFile(ifstream& asciiFile, double& data[][3], char state) {
	

}

/*
void intoCpp(double xOut[][3],
			double& initialData[][3],
			double& finalData[][3],
			char flag,
			int n){
	if (flag == 'i') {
		for (int row = 0; row < n; row++) {
			for (int col = 0; col < 3; col++) {
				initialData[row][col] = xOut[row][col];
			}
		}
	}

	else if (flag == 'f') {
		for (int row = 0; row < n; row++) {
			for (int col = 0; col < 3; col++) {
				finalData[row][col] = xOut[row][col];
			}
		}
	}
}*/
