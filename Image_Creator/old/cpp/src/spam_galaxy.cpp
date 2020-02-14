// Author: 		Matt Ogden
// Created: 	Fall 2018
// Altered: 	15 July, 2019
// Description: Source file for the galaxy class for particles from the spam simulation

#include "spam_galaxy.hpp"


using namespace std;
using namespace cv;


Galaxy::Galaxy(){
	maxr = maxb = ix = iy = iz = fx = fy = fz = 0;
    xmax = xmin = ymax = ymin = 0;
    numThreads = 1;
}


void Galaxy::delMem(){
    delete ipart;
    delete fpart;
}

void Galaxy::read(ifstream& infile, int part, char state){
    npart = part;
    if ( state == 'i' ){
        ipart = (point *) malloc(npart*sizeof(point));
        for (int i=0;i<npart;i++){
            infile>>ipart[i].x>>ipart[i].y>>ipart[i].z>>ipart[i].vx>>ipart[i].vy>>ipart[i].vz;
        }
    }
    else if ( state == 'f' ){
	    fpart = (point *) malloc(npart*sizeof(point));
    	for (int i=0;i<npart;i++){
		    infile>>fpart[i].x>>fpart[i].y>>fpart[i].z>>fpart[i].vx>>fpart[i].vy>>fpart[i].vz;
        }
    }
    else
        printf("galaxyClass.read - particle state %c unidentified\n",state);
}

//  Old version of write.
void Galaxy::write(Mat &img, int gsize, float weight, float rconst, point *pts){
	float val;
	maxb=0;
	for (int i=0;i<npart;i++){
		int row = int(pts[i].y);
		int col = int(pts[i].x);
		float pbright = exp( - rconst* ipart[i].r / maxr);
		for (int k=0;k<gsize;k++){
			for (int l=0;l<gsize;l++){
				val = pbright*1.0/(2*3.14*weight*weight)*exp( -1.0*((k-1.0*gsize/2)*(k-1.0*gsize/2) +  (l-1.0*gsize/2)*(l-1.0*gsize/2))/(2*weight*weight));
				
				//printf("write: %d,%d: %f\n",k,l,val);

				img.at<float>(row +k-gsize/2, col+l-gsize/2)+=val;
				if (maxb < img.at<float>(row+k-gsize/2,col+l-gsize/2))
					maxb = img.at<float>(row+k-gsize/2,col+l-gsize/2);
			}
		}
	}
	
	cout << "max brightness " << maxb << endl;
}

//  New version
void Galaxy::write(Mat &img, const Mat &blur, int gsize, float rconst, char state){
	point *pts;
	if (state=='i')
		pts = ipart;
	else if (state=='f')
		pts = fpart;
	else{
		printf("In galaxy.write - state %c not recognized\n",state);
		return;
	}	
	
	float pbright;
	int myNum;
	Rect roi;
	
	
	int mid = gsize/2;
	
	//  Create separate images to write to. 
	vector<Mat> imgV;
	for (int i=0;i<numThreads;i++){
		imgV.push_back(img.clone());
	}
	
	for (int i=0;i<npart;i++){		
		pbright = exp( - rconst* ipart[i].r / maxr);		
		roi = Rect( int(pts[i].x)-mid, int(pts[i].y)-mid, blur.cols, blur.rows );
		addWeighted(img(roi),1,blur,pbright,0.0,img(roi));
		}
	double min;
	Point maxLoc, minLoc;
	minMaxLoc(img,&min,&maxb,&minLoc,&maxLoc);
	//cout << "max brightness " << img.at<float>(maxLoc) <<" at " << maxLoc.x << ' ' << maxLoc.y<<endl;
}


void Galaxy::write_mask(Mat &img){
    point *pts;

	pts = ipart;

	// Offset initial particles to final position
	for (int i=0;i<npart;i++){


		int xVal = int( pts[i].x - ix + fx) ;
		int yVal = int( pts[i].y - iy + fy);

		if ( i%10000 ==0)
		  printf("i = %d: %d %d\n",int(pts[i].x), int(ix), int(fx));

		if ( ( xVal > img.cols ) || ( xVal < 0 ) ){
		  printf("xVal Error - %d", xVal );
		}
		else if ( ( yVal > img.rows ) || ( yVal < 0 ) ){
		  printf("yVal Error - %d", yVal );
		}
		else
		  img.at<float>( yVal , xVal )=1.0;
	}
	printf("In galaxy write_mask_4\n");

}
void Galaxy::simple_write(Mat &img,char state, int ox, int oy){

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

    if (write){
	    for (int i=0;i<npart;i++){
			if (  ( pts[i].x + ox > 0.0 ) &&  ( pts[i].x +ox < img.cols ) &&  ( pts[i].y +oy > 0.0 ) && ( pts[i].y +oy< img.rows ) )
			  img.at<float>( int(pts[i].y + oy), int( pts[i].x + ox) ) = 256;
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

    if (write){
	    for (int i=0;i<npart;i++){
			if (  ( pts[i].x > 0.0 ) &&  ( pts[i].x < img.cols ) &&  ( pts[i].y > 0.0 ) && ( pts[i].y < img.rows ) )
			  img.at<float>(int(pts[i].y),int( pts[i].x))=256.0;
	    }
    }
}

void Galaxy::dot_write(Mat &img,char state){
    point *pts;
    if (state == 'i'){
        pts = ipart;
    }
    else if (state == 'f'){
        pts = fpart;
    }
    else {
        cout << "State " << state << " not found.  Skipping simple write\n";
		return;
	}

    
	for (int i=0;i<npart;i++){
		img.at<float>(int(pts[i].y),int( pts[i].x))=img.at<float>(int(pts[i].y),int( pts[i].x))+1;
	}
    
}

//  Calculate r and find various maxes.
void Galaxy::calc_values(){
	

	maxr = 0;
	xmax = xmin = ipart[0].x;
	ymax = ymin = ipart[0].y;
	

	
	if(numThreads>1){
		
		float tmaxr, txmax, txmin, tymax, tymin;
		tmaxr = 0;
		txmax = txmin = ipart[0].x;
		tymax = tymin = ipart[0].y;
	
			for (int i=0;i<npart;i++){
				ipart[i].r = sqrt((ipart[i].x-ix)*(ipart[i].x-ix) + (ipart[i].y-iy)*(ipart[i].y-iy) + (ipart[i].z-iz)*(ipart[i].z-iz));  // calc radii from center of galaxies

				//  Find max and min x/y values
				if (ipart[i].r > tmaxr)
					tmaxr = ipart[i].r;
				if (fpart[i].x > txmax)
					txmax = fpart[i].x;
				if (fpart[i].y > tymax)
					tymax = fpart[i].y;
				if (fpart[i].x < txmin)
					txmin = fpart[i].x;
				if (fpart[i].y < tymin)
					tymin = fpart[i].y;
				
			}
		maxr = tmaxr;
		xmax = txmax;
		xmin = txmin;
		ymax = tymax;
		ymin = tymin;
		
		//printf("par: %f %f %f %f %f\n",maxr,xmax,xmin,ymax,ymin);
	}
	
	else	
	{
		
		maxr = 0;
		xmax = xmin = ipart[0].x;
		ymax = ymin = ipart[0].y;
	
		for (int i=0;i<npart;i++){
			ipart[i].r = sqrt((ipart[i].x-ix)*(ipart[i].x-ix) + (ipart[i].y-iy)*(ipart[i].y-iy) + (ipart[i].z-iz)*(ipart[i].z-iz));  // calc radii from center of galaxies

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
		//printf("seq: %f %f %f %f %f\n",maxr,xmax,xmin,ymax,ymin);
	}
	

	if ( ipart[npart-1].z > 100 || ipart[npart-1].z < -100 )
		printf("Suspicious value. Please check if real: %f \n",ipart[npart-1].z);
}

//  Changes the location of the points to their corresponding pixel point on an image. 
void Galaxy::adj_points(int height, int xo, int yo, float scale, float theta){
		
		float nx, ny;

		for (int i=0; i<npart; i++)
		{

			nx = fpart[i].x;
			ny = fpart[i].y;

			nx = fpart[i].x*cos(theta) - fpart[i].y*sin(theta);
			ny = fpart[i].x*sin(theta) + fpart[i].y*cos(theta);

			nx= scale*nx;
			ny= scale*ny;

			nx = nx + xo;
			ny = height - (ny + yo);

			fpart[i].x = nx;
			fpart[i].y = ny;

			nx = ( scale * ( fx*cos(theta) - fy*sin(theta) ) ) + xo;
			nx = height - ( scale * ( fx*sin(theta) + fy*cos(theta) )  + yo ) ;
			fx = nx;
			fy = ny;


			nx = ipart[i].x*cos(theta) - ipart[i].y*sin(theta);
			ny = ipart[i].x*sin(theta) + ipart[i].y*cos(theta);

			nx= scale*nx;
			ny= scale*ny;

			nx = nx + xo;
			ny = height - (ny + yo);

			ipart[i].x = nx;
			ipart[i].y = ny;


			nx = ( scale * ( ix*cos(theta) - iy*sin(theta) ) ) + xo;
			nx = height - ( scale * ( ix*sin(theta) + iy*cos(theta) )  + yo ) ;
			ix = nx;
			iy = ny;
			
		}

}


//  Changes the location of the points to their corresponding pixel point on an image. 
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
    printf("NumPart: %d\n",npart);
    printf("Checking init particles\n");
    for (int i=0;i<npart;i++){
        if ( fpart[i].x < 0 || fpart[i].y < 0){
            cout << printf("Below zero: %f %f\n",fpart[i].x,fpart[i].y);
        }
        printf("%d %f %f\n",i,fpart[i].x, fpart[i].y);
        printf("%f %f %f %f \n",ipart[i].x,ipart[i].y,fpart[i].x,fpart[i].y);
    }
}



