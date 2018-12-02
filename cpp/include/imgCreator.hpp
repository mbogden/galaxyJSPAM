// Author: Matt Ogden
#ifndef IMGCREATOR_H
#define IMGCREATOR_H

#include <unistd.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <omp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace std;
using namespace cv;


struct point{
	double r,x,y,z,vx,vy,vz;
};

struct paramStruct{
	string name;
	int gaussian_size;
	float gaussian_weight;
	float radial_constant;
	float norm_value;
	int image_rows;
	int image_cols;
	
	void loadParam(ifstream& fin){
		string str;
		fin >> name;
		while( fin >> str ){
			//cout << str << endl;
			if( str.compare("gaussian_size")==0)
				fin >> gaussian_size;
			else if ( str.compare("gaussian_weight")==0)
				fin >> gaussian_weight;
			else if ( str.compare("radial_constant")==0)
				fin >> radial_constant;
			else if ( str.compare("norm_value")==0)
				fin >> norm_value;
			else if ( str.compare("image_rows")==0)
				fin >> image_rows;
			else if ( str.compare("image_cols")==0)
				fin >> image_cols;
			else
				printf("Parameter %s not found\n",str.c_str());
		}
		
		if (gaussian_size%2==0)
			gaussian_size--;
	}
	
	void print(){
		cout << "param: " << name <<endl;
		cout << "gaussian_size " << gaussian_size << endl;
		cout << "gaussian_weight " << gaussian_weight << endl;
		cout << "radial_constant " << radial_constant << endl;
		cout << "norm_value " << norm_value << endl;
		cout << "image_rows " << image_rows << endl;
		cout << "image_cols " << image_cols << endl;
	}
};



class Galaxy
{
public:
	double ix,iy,iz,fx,fy,fz;  // beginning and final x,y,z coordinates;
    double maxr, xmin, xmax, ymin, ymax, maxb, maxb2;
	int npart;
    int numThreads;
	point *ipart, *fpart;

    Galaxy();

	void read(ifstream& infile,int part, char state);
	void write(Mat &img, int gsize, float weight, float radial_constant, point *pts);
	void write(Mat &img, const Mat &blur, int gsize, float rconst, char state);
    void simple_write( Mat &img, char state);
	void dot_write(Mat &img,char state);
	void calc_values();
	void adj_points(int xsize, int ysize, int gsize, point *pts);
	void add_center(double x, double y, double z, char state);
	void add_center_circle(Mat &img);
    void check_points();
    void delMem();
};




class ImgCreator
{
public:
	int numThreads;
	Mat img, dest, blur, imgTemp;
	string runDir, runName, sdssName, infoName, picName;
	string fpartFileName, ipartFileName; 
	vector<string> runFiles;
	ifstream infoFileIn, ipartFile, fpartFile;
	ofstream infoFileOut;
	
	paramStruct param;	
	Galaxy g1, g2;
	int npart1, npart2;
	double x,y,z;
	bool picFound, infoFound , iFileFound , fFileFound , multPFiles;
	bool printStdWarning, overWriteImages;
	
	ImgCreator(string in, paramStruct paramIn);
	ImgCreator(string in, paramStruct paramIn, bool overIn, bool warnIn);
	ImgCreator(int numThreadIn, string in, paramStruct paramIn, bool overIn, bool warnIn);
	ImgCreator(string in, paramStruct paramIn, bool overIn, bool warnIn, int numThreadIn);
	
	bool prepare();	
	void changeParam(paramStruct paramIn);
	void compare(Galaxy &g1, Galaxy &g2);
	void makeGaussianBlur();
	void makeImage2();
	void makeImage();
	void makeImageOLD();
	void writeInfo();
	void writeInfoPlus();
	void normalize_image(float max);
	void normalize_image2();
	void getDir(vector<string> &myVec, string dirPath);
	void delMem();

};







#endif
