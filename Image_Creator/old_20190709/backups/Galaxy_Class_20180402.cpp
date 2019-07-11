#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include "Galaxy_Class.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;


Galaxy::Galaxy(){
	maxr=maxb=ix=iy=iz=fx=fy=fz=0;
    xmax = xmin = ymax = ymin = 0;
    maxb2 = 0;
    numThreads = 1;
    //xmin = ymin = -2.5;  // fixed max and min
    //xmax = ymax = 2.5;
}

void Galaxy::delMem(){
    delete ipart;
    delete fpart;
}

void Galaxy::read(ifstream& infile, int part, char state){
    npart = part;
    if ( state == 'i' )
    {
        ipart = (point *) malloc(npart*sizeof(point));
        for (int i=0;i<npart;i++)
        {
            infile>>ipart[i].x>>ipart[i].y>>ipart[i].z>>ipart[i].vx>>ipart[i].vy>>ipart[i].vz;
        }
    }
    else if ( state == 'f' )
    {
	    fpart = (point *) malloc(npart*sizeof(point));
    	for (int i=0;i<npart;i++){
		    infile>>fpart[i].x>>fpart[i].y>>fpart[i].z>>fpart[i].vx>>fpart[i].vy>>fpart[i].vz;
        }
    }
    else
        printf("Galaxy read particle state unidentified\n");
}


void Galaxy::write(Mat &img, int gsize, float weight, float rconst, point *pts){

	maxb=0;
	for (int i=0;i<npart;i++){
		int row = int(pts[i].y);
		int col = int(pts[i].x);
		float pbright = exp( - rconst* ipart[i].r / maxr);
		for (int k=0;k<gsize;k++){
			for (int l=0;l<gsize;l++){
				float val = pbright*1.0/(2*3.14*weight*weight)*exp( -1.0*((k-1.0*gsize/2)*(k-1.0*gsize/2) +  (l-1.0*gsize/2)*(l-1.0*gsize/2))/(2*weight*weight));

				img.at<float>(row +k-gsize/2, col+l-gsize/2)+=val;
				if (maxb < img.at<float>(row+k-gsize/2,col+l-gsize/2))
					maxb = img.at<float>(row+k-gsize/2,col+l-gsize/2);
			}
		}
	}
}



void Galaxy::simple_write(Mat &img,char state){
    point *pts;
    bool write = false;
    if (state == 'i'){
        pts = ipart;
        write = true;
    }
    else if (state == 'f'){
        pts = fpart;
        write = true;
    }
    else
        cout << "State " << state << " not found.  Skipping simple write\n";

    if (write)
    {
	    for (int i=0;i<npart;i++){
		    img.at<float>(int(pts[i].y),int( pts[i].x))=1.0;
	    }
    }
}


void Galaxy::calc_values(){

	for (int i=0;i<npart;i++){
		ipart[i].calc_radius(ix,iy,iz);  // calc radii from cent of galaxies

        //  Find max and min x/y values
		if (ipart[i].r > maxr)
			maxr = ipart[i].r;
        if (fpart[i].x > xmax)
            xmax = fpart[i].x;
        if (fpart[i].y > ymax)
            ymax = fpart[i].y;
        if (fpart[i].x < xmin)
            xmin = fpart[i].x;
        if (fpart[i].y < ymin)
            ymin = fpart[i].y;
	}
}


void Galaxy::adj_points(int xsize, int ysize, int gsize, point *pts){

        int scale_factor;

        xmax-=xmin;
        ymax-=ymin;

        if( (1.0*(xsize-gsize)/xmax) > (1.0*(ysize-gsize)/ymax))
            scale_factor = 1.0*(ysize-gsize)/ymax;
        else
            scale_factor = 1.0*(xsize-gsize)/xmax;

        fx = (fx-xmin)*scale_factor + gsize/2.0;
        fy = ysize - ( ( fy-ymin)*scale_factor + gsize/2.0);


        for (int i=0; i<npart; i++)
        {
            pts[i].x= (pts[i].x-xmin)*scale_factor + gsize/2.0;
            pts[i].y= ysize-((pts[i].y-ymin)*scale_factor + gsize/2.0);

        }

        xmax = (xmax-xmin)*scale_factor + gsize/2.0;
        ymax = ysize-((ymax-ymin)*scale_factor + gsize/2.0);



}



void Galaxy::add_center(double x, double y, double z, char state){
    //  State is beginning or final state of galaxy;
    if (state == 'i'){
        ix = x;
        iy = y;
        iz = z;
    }
    else if (state == 'f'){
        fx = x;
        fy = y;
        fz = z;
    }
    else
        printf("Galaxy::add_center: state not recognized");
}

void Galaxy::add_center_circle(Mat &img){

            circle( img,Point2f(int(fx),int(fy)),10,Scalar(255,255,255),2,8);
}

void Galaxy::check_points(){
    printf("Maxes: %f %f %f %f\n", xmax, xmin, ymax, ymin);
    //printf("NumPart: %d\n",npart);
    //printf("Checking init particles\n");
    for (int i=0;i<npart;i++){
        ;
        if ( fpart[i].x < 0 || fpart[i].y < 0){
            cout << printf("Below zero: %f %f\n",fpart[i].x,fpart[i].y);
        }
        printf("%d %f %f\n",i,fpart[i].x, fpart[i].y);;
        //printf("%f %f %f %f \n",ipart[i].x,ipart[i].y,fpart[i].x,fpart[i].y);
    }
}


