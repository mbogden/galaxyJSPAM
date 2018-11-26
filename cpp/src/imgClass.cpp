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
        imageCenter[2] = Point2f(imageCenter[0].x+(imageCenter[0].y-imageCenter[1].y)/3,imageCenter[0].y+(imageCenter[1].x-imageCenter[0].x)/3);  // Creates a triangle of points
    }


    //  Takes target image and centers to prepare model image for comparison
    bool imgClass::loadImage(const Mat &targetImg, const Point2f targetCenter[]){

        img = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
        if ( !img.data ){
            cout << "Image failed to load " << path << " Skipping...\n";
            return false;
        }

        warpMat = getAffineTransform(imageCenter,targetCenter);
        warpAffine(img,imgFinal,warpMat,targetImg.size());

        return true;
    }

    //  Add center to original image.  Useful in troubleshooting
    void imgClass::addCircle(){
        circle(img, imageCenter[0], 10, Scalar(255,255,255),2,8);
        circle(img, imageCenter[1], 10, Scalar(255,255,255),2,8);
        circle(img, imageCenter[2], 10, Scalar(255,255,255),2,8);
    }
