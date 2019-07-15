#ifndef GALAXY_H
#define GALAXY_H

#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;


struct point{
	double r,x,y,z,vx,vy,vz;
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
    void simple_write( Mat &img, char state);
	void calc_values();
	void adj_points(int xsize, int ysize, int gsize, point *pts);
	void add_center(double x, double y, double z, char state);
	void add_center_circle(Mat &img);
    void check_points();
    void delMem();
};

#endif
