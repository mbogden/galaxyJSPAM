#include "imgClass.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;

    imgClass::imgClass(){
        ;
    }

    imgClass::imgClass(string nameIn, string pathIn){
        name = nameIn;
        path = pathIn;
    //  Determine parameter used to create image
        size_t fImg = name.find("param");
        if ( fImg != string::npos ){
            paramName = name.substr(fImg,9);
            //cout << paramName<<endl;
        }
        else {
            cout << "could not find parameter name in " << name << endl;
        }

    }

    //  Add pixel coordinates to centers of modeled galaxies
    void imgClass::addCenter(int x1, int y1, int x2, int y2 ){

        imageCenter[0] = Point2f( x1, y1 );
        imageCenter[1] = Point2f( x2, y2 );
        imageCenter[2] = Point2f(imageCenter[0].x+(imageCenter[0].y-imageCenter[1].y),imageCenter[0].y+(imageCenter[1].x-imageCenter[0].x));  // Creates a triangle of points
    }


    //  Takes target image and centers to prepare model image for comparison
    bool imgClass::loadImage(){

        imgIn = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
        if ( !imgIn.data ){
            cout << "Image failed to load " << path << " Skipping...\n";
            return false;
        }
        else
            return true;
    }

    void imgClass::processImg(const imgClass &target){

        warpMat = getAffineTransform(imageCenter,target.imageCenter);
        warpAffine(imgIn,imgFinal,warpMat,target.imgFinal.size());


    }

    void imgClass::normBrightness(const imgClass &target){

        long int Sum =0;
        for(int i=0;i<imgFinal.rows;i++){
            for(int j=0;j<imgFinal.cols;j++){
                Sum += imgFinal.at<uint8_t>(i,j);
            }
        }
        bSum = 1.0*Sum;
        float bCorr = target.bSum/bSum;

        for(int i=0;i<imgFinal.rows;i++){
            for(int j=0;j<imgFinal.cols;j++){
                imgFinal.at<uint8_t>(i,j) = uint8_t(imgFinal.at<uint8_t>(i,j)*bCorr);
            }
        }

    }

    void imgClass::processTargetImg(){

        imgFinal = imgIn;

        long int Sum =0;
        for(int i=0;i<imgFinal.rows;i++){
            for(int j=0;j<imgFinal.cols;j++){
                Sum += imgFinal.at<uint8_t>(i,j);
            }
        }
        bSum = 1.0*Sum;
    }

    void imgClass::createAltSizes(){
        resize(imgFinal,img1024,Size(1024,1024),0,0,INTER_LINEAR);
        resize(img1024,img512,Size(512,512),0,0,INTER_LINEAR);
        resize(img1024,img256,Size(256,256),0,0,INTER_LINEAR);
        resize(img1024,img128,Size(128,128),0,0,INTER_LINEAR);
        resize(img1024,img64,Size(64,64),0,0,INTER_LINEAR);

    }

    //  Add center to original image.  Useful in troubleshooting
    void imgClass::addCircle(){
        circle(imgIn, imageCenter[0], 10, Scalar(255,255,255),2,8);
        circle(imgIn, imageCenter[1], 10, Scalar(255,255,255),2,8);
        circle(imgIn, imageCenter[2], 10, Scalar(255,255,255),2,8);
    }
