#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>

//#include "opencv2/utility.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;



float ratediff(const Mat &m1){

    int max, min, var1, var2, N, val;
	max =  m1.at<uint8_t>(0,0);
	min = max;
	var1 = 0;		// Sum of pixel differences
	var2 = 0; 	    // Sum of squared pixel differences
	float std;		// Standard Deviation
	N = m1.rows*m1.cols;	// Total number of pixels
	for (int i=0;i<m1.rows;i++){
		for (int j=0;j<m1.cols;j++){
			val = m1.at<uint8_t>(i,j);
            if(val>=0 && val<255){
                var1 += val;
				var2 += val*val;
				if (val > max)
					max = val;
				else if ( val < min )
					min = val;
			}
            else
                cout << val <<endl;
        }
	}
    cout<<endl;

	std = sqrt(  1.0*var2/N  - 1.0*var1*var1/(N*N) );

	cout <<"Score: " << 1.0-1.0*var1/N/255<<endl;
    return var1/N;
}



int main(int argc, char *argv[]){

	Mat sim_raw = imread("img/img.r2n5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat real = imread("real1.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat sim_warp;

	Point2f rc[3];
	Point2f sc[3];

	rc[1] = Point2f(178,250);
	rc[2] = Point2f(315,105);
	rc[0] = Point2f(rc[1].x+(rc[1].y-rc[2].y)/3,rc[1].y+(rc[2].x-rc[1].x)/3);

	sc[1] = Point2f(830,635);
	sc[2] = Point2f(360,805);
	sc[0] = Point2f(sc[1].x+(sc[1].y-sc[2].y)/3,sc[1].y+(sc[2].x-sc[1].x)/3);

	Mat warp_mat = getAffineTransform(sc,rc);
	warpAffine(sim_raw,sim_warp,warp_mat,real.size());

	Mat diff;
	absdiff(real, sim_warp, diff);
	ratediff(diff);
    ratediff(real);
    ratediff(sim_warp);

    cout << real.type()<< ' '<< real.channels()<<' '<<real.depth()<<endl;
    cout << sim_warp.type()<< ' '<< sim_warp.channels()<<' '<<sim_warp.depth()<<endl;
    cout << diff.type()<< ' '<< diff.channels()<<' '<<diff.depth()<<endl;

	imshow( "Sim", sim_warp);
	imshow( "Real", real);
	imshow( "Diff", diff);
	waitKey(0);


	imwrite("diff.jpg",diff);
	return 0;

	//~ int thickness = 2;
	//~ int lineType = 8;

	//~ for(int i=0;i<3;i++){
		//~ circle( real,rc[i],10, Scalar( 255, 255, 255 ), thickness, lineType );
		//~ circle( sim_raw,sc[i],10, Scalar( 255, 255, 255 ), thickness, lineType );
	//~ }

	imwrite("diff.jpg",diff);

	return 0;
}

