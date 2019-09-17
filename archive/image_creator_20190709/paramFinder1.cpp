// Author:  Matthew Ogden

#include "imgCreator.hpp"
#include <iostream>
#include <string>
#include "imgCreator.hpp"

using namespace cv;
using namespace std;

double rateDiff_v2(const Mat &m1);

// ./paramFinder.exe pathTo/runFile target.png targetInfo.txt
int main(int argc, char *argv[])
{
  string runFolder = argv[1];
  string targetName = argv[2];
  string targetInfoLoc = argv[3];

  cout << "Run directory: " << runFolder << endl;
  cout << "Target Location: " << targetName << endl;
  cout << "Target Location: " << targetInfoLoc << endl;

  Mat target = imread(targetName);

  int adjust = 10;

  string name = "f";
  int gSize = 25;
  float gWeight = 5;
  float rConstant = 4;
  float nValue = 7;

  int imgRows = 1000;
  int imgCols = 1000;


  paramStruct param;  

  param.loadParam( name, gSize, gWeight, rConstant, nValue, imgRows, imgCols ); 

  ImgCreator myCreator(runFolder, param, true, true); 

  if (myCreator.prepare()){

    myCreator.makeImage2(false);
    Mat sim = myCreator.dest;	
	Mat diffMat;	
	
	imwrite("test1.png",sim);
	imwrite("test2.png",target);
	
	//absdiff(target,sim,diffMat);
	
	//absdiff(target.imgFinal,myRun.images[iImg].imgFinal,diffMat);
	
	//cout << "This should be a real value: " << rateDiff_v2(diffMat) << endl;

	cout << "I'm here!" << endl;
    myCreator.delMem();
  }


  //param.print();
   
}

double rateDiff_v2(const Mat &m1){

    int val;
    long int Sum=0;
    int count;
    double score;
	for (int i=0;i<m1.rows;i++){
		for (int j=0;j<m1.cols;j++){
			val = m1.at<uint8_t>(i,j);
            if (val>0 && val<256){
                count++;
                Sum += val;
            }
            else if (val >= 256)
                cout << val <<endl;
        }
	}

    score = 1.0-1.0*Sum/count/255;
    return score;
    //return score;
}

