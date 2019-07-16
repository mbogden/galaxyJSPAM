// Author: 		Matthew Ogden
// Created: 	Fall 2018
// Altered: 	15 july 2019
// Description: 	Header file for my class for creating images
//
#ifndef IMGCREATOR_H
#define IMGCREATOR_H

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <omp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "spam_galaxy.hpp"


using namespace std;
using namespace cv;



struct paramStruct{
	string name;
	int gaussian_size;
	float gaussian_weight;
	float radial_constant;
	float norm_value;
	int image_rows;
	int image_cols;

    void loadParam( string inName, int gSize, float gWeight, float rConstant, float nValue, int imgRows, int imgCols ){
      name = inName;
      gaussian_size = gSize;
      gaussian_weight = gWeight;
      radial_constant = rConstant;
      norm_value = nValue;
      image_rows = imgRows;
      image_cols = imgCols;
    }
	
	void loadParam(ifstream& fin){
		string str;
		fin >> name;
		while( fin >> str ){
			//cout << str << endl;
			if( str.compare("gaussian_size")==0){
				fin >> gaussian_size;
                cout << gaussian_size <<endl;
              }

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




struct imgCreator_input
{
  public:
	string p1Loc, p2Loc, runDir, infoName, picName;
	bool overWrite, printSTDWarning, unzip, makeMask;

	imgCreator_input(){
	  overWrite = false;
	  printSTDWarning = false;
	  unzip = false;
	  makeMask = false;
	}
};



class ImgCreator
{
public:
	int numThreads;
	Mat img, dest, blur, imgTemp, mask_1;
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
	bool printStdWarning, overWriteImages, imageParamHeaderPresent;
	bool unzip, makeMask;


	
	ImgCreator( imgCreator_input, paramStruct paramIn );
	bool new_prepare();	
	bool readInfoFile();
	void make_mask(string saveLocNmae);



	ImgCreator(string in, paramStruct paramIn);
	ImgCreator(string p1LocIn, string p2LocIn, paramStruct paramIn, bool overIn, bool warnIn);
	ImgCreator(string in, paramStruct paramIn, bool overIn, bool warnIn);
	ImgCreator(int numThreadIn, string in, paramStruct paramIn, bool overIn, bool warnIn);
	ImgCreator(string in, paramStruct paramIn, bool overIn, bool warnIn, int numThreadIn);
	
	bool prepare();	
	void changeParam(paramStruct paramIn);
	void compare(Galaxy &g1, Galaxy &g2);
	void makeGaussianBlur();	
	void makeImage2();
	void makeImage2(bool saveImg);
	void makeImage2(string saveLocNmae);
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
