// Author: 		Matt Ogden
// Created: 	Fall 2018
// Altered: 	July 15, 2019
//Description: 	Header file for particles from the spam galaxy simulation
//

#ifndef SPAM_GALAXY_H
#define SPAM_GALAXY_H


#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"


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
	void write(Mat &img, const Mat &blur, int gsize, float rconst, char state);
    void write_mask( Mat &img);
    void simple_write( Mat &img, char state, int ox, int oy);
    void simple_write( Mat &img, char state);
	void dot_write(Mat &img,char state);
	void calc_values();
	void adj_points(Mat warpMat);
	void adj_points(int height, int xsize, int ysize, float scale, float theta);
	void adj_points(int xsize, int ysize, int gsize, point *pts);
	void add_center(double x, double y, double z, char state);
	void add_center_circle(Mat &img);
    void check_points();
    void delMem();
};

#endif
