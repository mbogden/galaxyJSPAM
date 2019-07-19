#ifndef IMGCLASS_H
#define IMGCLASS_H


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


//  Struct that contains useful information and processes for a single model image
struct imgClass{
    double bSum;
    Mat imgIn, imgTemp, imgFinal, warpMat;
    Mat img64, img128, img256, img512, img1024;
    string name;
    string path;
    string paramName;
    Point2f imageCenter[3];

    imgClass();
    imgClass(string nameIn, string pathIn);

    //  Add pixel coordinates to centers of modeled galaxies
    void addCenter(int x1, int y1, int x2, int y2 );

    //  Takes target image and centers to prepare model image for comparison
    bool loadImage();

    //  Takes target image and centers to prepare model image for comparison
    void processImg(const imgClass &target);

    void normBrightness(const imgClass &target);

    //  Takes target image and centers to prepare model image for comparison
    void processTargetImg();

    void createAltSizes();

    //  Add center to original image.  Useful in troubleshooting
    void addCircle();
};

#endif

