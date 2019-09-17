// Author: Matt Ogden
#include "imgCreator.hpp"

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
	
	
	//  If parallel process.
	if (numThreads > 1){
		
		//  Create separate images to write to. 
		vector<Mat> imgV;
		for (int i=0;i<numThreads;i++){
			imgV.push_back(img.clone());
		}
		//  Create threads
		#pragma omp parallel num_threads(numThreads) private(pbright, myNum,roi)
		{
			
			#pragma omp for
			for (int i=0;i<npart;i++){	
				
				myNum = omp_get_thread_num();

				pbright = exp( - rconst* ipart[i].r / maxr);	
				roi = Rect( int(pts[i].x)-mid, int(pts[i].y)-mid, blur.cols, blur.rows );
				//printf("In thread %d\n",myNum);
				addWeighted(imgV[myNum](roi),1,blur,pbright,0.0,imgV[myNum](roi));
			}	
		}
		
		for (int i=0;i<numThreads;i++){	
			//printf("loop %d\n",i);
			addWeighted(imgV[0],1,imgV[i],1,0.0,imgV[0]);
			//imgV[i].release();
		}
				
		img.release();
		img = imgV[0].clone();
		//imgV[0].release();

	}
	
	else if (numThreads == 1){
		for (int i=0;i<npart;i++){		
			pbright = exp( - rconst* ipart[i].r / maxr);		
			roi = Rect( int(pts[i].x)-mid, int(pts[i].y)-mid, blur.cols, blur.rows );
			addWeighted(img(roi),1,blur,pbright,0.0,img(roi));
		}	
	}	
	else
		printf("In galaxy.write...  How are there %d threads?\n",numThreads);
	
	double min;
	Point maxLoc, minLoc;
	minMaxLoc(img,&min,&maxb,&minLoc,&maxLoc);
	//cout << "max brightness " << img.at<float>(maxLoc) <<" at " << maxLoc.x << ' ' << maxLoc.y<<endl;
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
		    img.at<float>(int(pts[i].y),int( pts[i].x))=1.0;
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
	
		#pragma omp parallel num_threads(numThreads)
		{
			#pragma omp for reduction(max:tmaxr,txmax,tymax), reduction(min:txmin,tymin)
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

		#pragma omp parallel num_threads(numThreads)
		{
			#pragma omp for
			for (int i=0; i<npart; i++)
			{
				pts[i].x= (pts[i].x-xmin)*scale_factor + gsize/2.0;
				pts[i].y= ysize-((pts[i].y-ymin)*scale_factor + gsize/2.0);

			}
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













 //  Default constructor.  Assumes:  One thread. Standard Warnings Off. Single img at a time. 
ImgCreator::ImgCreator(string in, paramStruct paramIn){	
	runDir = in;
	param = paramIn;
	printStdWarning = false;
	overWriteImages = true;
	numThreads = 1;
	
	img = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	dest = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	blur = Mat(param.gaussian_size,param.gaussian_size,CV_32F, Scalar(0));
	picFound = infoFound = iFileFound = fFileFound = multPFiles = false;
	
	//printf("In ImgCreator Class. In directory %s\n", runDir.c_str());
}



// 
ImgCreator::ImgCreator(string in, paramStruct paramIn, bool overWriteIn, bool warnIn){
	
	runDir = in;
	printStdWarning = warnIn;
	overWriteImages = overWriteIn;
	param = paramIn;
	numThreads = 1;
	
	img = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	dest = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	picFound = infoFound = iFileFound = fFileFound = multPFiles = false;
	
	//printf("In ImgCreator Class. In directory %s\n", runDir.c_str());
	
}

// Implementation of internal parallel threading. 
ImgCreator::ImgCreator(int numThreadsIn, string in, paramStruct paramIn, bool overWriteIn, bool warnIn){
	
	runDir = in;
	printStdWarning = warnIn;
	overWriteImages = overWriteIn;
	param = paramIn;
	numThreads = numThreadsIn;
	g1.numThreads = numThreads;
	g2.numThreads = numThreads;
	
	img = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	dest = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	picFound = infoFound = iFileFound = fFileFound = multPFiles = false;
	
	//printf("In ImgCreator Class. In directory %s\n", runDir.c_str());
	printf("In imgCreator.  using %d threads\n",numThreads);
	
}


void ImgCreator::changeParam(paramStruct paramIn){
	
	param = paramIn;	
	/************************** CHANGE ME ******************/
    //picName = sdssName + '.' + runName + '.' + param.name + ".model.png";
    picName = "images/" + sdssName + '.' + runName + '.' + param.name + ".model.png";
	picName = runDir + picName;
	
	img.release();
	dest.release();
	blur.release();
	
	img = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	dest = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	blur = Mat(param.gaussian_size,param.gaussian_size,CV_32F, Scalar(0));
	
	makeGaussianBlur();
}


void ImgCreator::makeGaussianBlur(){
	
	int gsize = param.gaussian_size;
	float weight = param.gaussian_weight;
	float val;	
	int mid = gsize/2;
	

	for (int k=0;k<gsize;k++){
		for (int l=0;l<gsize;l++){
			
			val = 1.0/(2*3.1415926535*weight*weight)*exp( -1.0*((k-mid)*(k-mid) + (l-mid)*(l-mid))/(2*weight*weight));
			
			blur.at<float>(k,l) = val;
		}
	}	
	
	//imwrite("test.png",blur);
}

void ImgCreator::makeImage2(){
	
	g1.dot_write(img,'f');
	g2.dot_write(img,'f');
	GaussianBlur(img, img, Size(param.gaussian_size,param.gaussian_size), param.gaussian_weight, 0);
	normalize_image2();
	dest.convertTo(dest,CV_8UC3,255.0);
	imwrite(picName,dest);

}

void ImgCreator::makeImage2(bool saveImg){
	
	g1.dot_write(img,'f');
	g2.dot_write(img,'f');
	GaussianBlur(img, img, Size(param.gaussian_size,param.gaussian_size), param.gaussian_weight, 0);
	normalize_image2();
	dest.convertTo(dest,CV_8UC3,255.0);
	if (saveImg)
		imwrite(picName,dest);
}


void ImgCreator::makeImage2(string saveLocName){
	
	g1.dot_write(img,'f');
	g2.dot_write(img,'f');
	GaussianBlur(img, img, Size(param.gaussian_size,param.gaussian_size), param.gaussian_weight, 0);
	normalize_image2();
	dest.convertTo(dest,CV_8UC3,255.0);
	//imwrite(saveLocName,dest);

}

void ImgCreator::makeImage(){
	
	g1.write(img,blur, param.gaussian_size,param.radial_constant,'f');
	g2.write(img,blur, param.gaussian_size,param.radial_constant,'f');
	normalize_image(g2.maxb);
	dest.convertTo(dest,CV_8UC3,255.0);
	imwrite(picName,dest);

}


void ImgCreator::makeImageOLD(){
	
	g1.write(img,param.gaussian_size,param.gaussian_weight,param.radial_constant, g1.fpart);
	g2.write(img,param.gaussian_size,param.gaussian_weight,param.radial_constant, g2.fpart);
	normalize_image(g2.maxb);
	dest.convertTo(dest,CV_8UC3,255.0);
	imwrite(picName,dest);

}



void ImgCreator::writeInfo(){
	//  Write to info file about pixel centers
	infoFileOut << param.name << ' ' << int(g1.fx) << ' ' << int(g1.fy) << ' ' << int(g2.fx) << ' ' << int(g2.fy) << endl;
}


void ImgCreator::writeInfoPlus(){
	//  Write to info file about pixel centers
	infoFileOut << param.name << ' ' << int(g1.fx) << ' ' << int(g1.fy) << ' ' << int(g2.fx) << ' ' << int(g2.fy);
	infoFileOut << ' ' << param.gaussian_weight << ' ' << param.radial_constant << ' ' << param.norm_value << endl;
}


void ImgCreator::delMem(){
	
	infoFileOut.close();	
	ipartFile.close();
	fpartFile.close();	
	
	img.release();
	dest.release();
	
	g1.delMem();
	g2.delMem();
}


void ImgCreator::getDir(vector<string> &myVec, string dirPath){
     // Search Directory for files
    DIR* dirp = opendir(dirPath.c_str());
    struct dirent * dp;
    while (( dp = readdir(dirp)) != NULL)
        myVec.push_back(dp->d_name);
}


void ImgCreator::compare(Galaxy &g1, Galaxy &g2){

	if (g1.xmin > g2.xmin)
		g1.xmin = g2.xmin;
	else
		g2.xmin = g1.xmin;

	if (g1.xmax < g2.xmax)
		g1.xmax = g2.xmax;
	else
		g2.xmax = g1.xmax;

	if (g1.ymin > g2.ymin)
		g1.ymin = g2.ymin;
	else
		g2.ymin = g1.ymin;

	if (g1.ymax < g2.ymax)
		g1.ymax = g2.ymax;
	else
		g2.ymax = g1.ymax;

	if (g1.maxb>g2.maxb)
		g2.maxb=g1.maxb;
	else
		g1.maxb=g2.maxb;
}

void ImgCreator::normalize_image2(){
	float val, vscaled;
	double min,max;
	Point maxLoc, minLoc;
	minMaxLoc(img,&min,&max,&minLoc,&maxLoc);
	
	
	#pragma omp parallel num_threads(numThreads) private(val, vscaled)
	{
		//printf("In thread %d\n",omp_get_thread_num());
		#pragma omp for
		for (int k=0;k<dest.rows;k++){
			for (int l=0;l<dest.cols;l++){
				val = img.at<float>(k,l)	;
				vscaled = pow(val/max,1/param.norm_value);
				dest.at<float>(k,l) = vscaled;
			}
		}
	}
}

void ImgCreator::normalize_image(float max){
	float val, vscaled;
	
	#pragma omp parallel num_threads(numThreads) private(val, vscaled)
	{
		//printf("In thread %d\n",omp_get_thread_num());
		#pragma omp for
		for (int k=0;k<dest.rows;k++){
			for (int l=0;l<dest.cols;l++){
				val = img.at<float>(k,l)	;
				vscaled = pow(val/max,1/param.norm_value);
				dest.at<float>(k,l) = vscaled;
			}
		}
	}
}



bool ImgCreator::prepare(){
	
	//  Search run Directory for files
	getDir(runFiles,runDir);
	string tempStr; 
	
	//  Find run name.  assumes its in path to directory
	size_t findRun = runDir.find("run");
	if (findRun!= string::npos)
		runName = runDir.substr(findRun,7);
	else {
		printf("Run name not found.  using \"runtemp0\" \n");
		runName = "runtemp0";
	}
	
	// find files
	for (unsigned int i=0; i<runFiles.size();i++)
	{
		size_t foundi = runFiles[i].find(".i.");
		size_t foundf = runFiles[i].find(".f.");
		if ( foundi != string::npos ){
			ipartFileName = runFiles[i];
			if (iFileFound == true)
				multPFiles = true;
			iFileFound = true;
			//printf("Found i file! %s\n",ipartFileName.c_str());
		}
		else if ( foundf != string::npos){
			fpartFileName = runFiles[i];
			if (fFileFound)
				multPFiles = true;
			fFileFound = true;
			//printf("Found f file! %s\n",fpartFileName.c_str());
		}
		else if ( runFiles[i].compare("info.txt") == 0 ) {
		   infoName = runFiles[i];
		   infoFound = true;
		   //printf("InfoName... %s",infoName.c_str());
		}
		else
			;//printf("%s was not i or f\n",runFiles[i].c_str());
	}
	//printf("Found files\n");
	
	infoName = runDir + "info.txt";
	//printf("Info dir/name %s\n",infoName.c_str());
	
	
	size_t pos[5];
	pos[0] = ipartFileName.find(".");
	pos[1] = ipartFileName.find(".",pos[0]+1);
	pos[2] = ipartFileName.find(".",pos[1]+1);
	pos[3] = ipartFileName.find(".",pos[2]+1);
	pos[4] = ipartFileName.find(".",pos[4]+1);

	sdssName = ipartFileName.substr(0,pos[0]);
	
	stringstream strm1, strm2;

	tempStr = ipartFileName.substr(pos[2]+1,pos[3]-pos[2]-1);
	strm1 << tempStr;
	strm1 >> npart1;

	tempStr = ipartFileName.substr(pos[3]+1,pos[4]-pos[3]-1);
	strm2 << tempStr;
	strm2 >> npart2;



	// Check if image already exists for current parameters
	picName = sdssName + '.' + runName + '.' + param.name + ".model.png";

	for (unsigned int i=0; i<runFiles.size();i++)
	{
		if( runFiles[i].compare(picName)==0){
			picFound = true;
			if (printStdWarning)
				printf("Image with %s already present in %s.\n",param.name.c_str(), runDir.c_str());
			if (!overWriteImages)
				return false;
		}
	}

	if (!iFileFound || !fFileFound){
		printf("Point Particles files could not be found in %s Exiting...\n",runDir.c_str());
		return false;
	}
	else {
		ipartFileName = runDir + ipartFileName;
		fpartFileName = runDir + fpartFileName;

		//  Read Initial particle file
		ipartFile.open(ipartFileName.c_str());
		if (ipartFile.fail())    {
			printf("Initial Particle file failed to open in %s\nSkipping...\n",runDir.c_str());
			iFileFound = false;
		}

		//  Final particle file
		fpartFile.open(fpartFileName.c_str());
		if (fpartFile.fail())  {
			printf("Final Particle file failed to open in %s\nSkipping...\n",runDir.c_str());
			fFileFound = false;
		}
	}


	if (multPFiles){
		printf("Multiple sets of point files found in %s skipping...\n", runDir.c_str());
		return false;
	}
	
	
	picName = runDir + picName;
	
	if ( !iFileFound || !fFileFound || multPFiles )
		return false;
	
	
	//  Read in particle files
	
	if(numThreads>=2){		
		#pragma omp parallel  num_threads(2)
		{
			#pragma omp single nowait
			{
				g1.read(ipartFile,npart1,'i');
				g2.read(ipartFile,npart2,'i');
				ipartFile >> x >> y >> z;
				g1.add_center(0,0,0,'i');
				g2.add_center(x,y,z,'i');	
			}	
			
			#pragma omp single nowait
			{
				//printf("read i file\n");
				g1.read(fpartFile,npart1,'f');
				g2.read(fpartFile,npart2,'f');
				fpartFile >> x >> y >> z;
				g1.add_center(0,0,0,'f');
				g2.add_center(x,y,z,'f');
			}	
			#pragma omp barrier
		}
	}
	else{
		
		g1.read(ipartFile,npart1,'i');
		g2.read(ipartFile,npart2,'i');
		ipartFile >> x >> y >> z;
		g1.add_center(0,0,0,'i');
		g2.add_center(x,y,z,'i');

		//printf("read i file\n");
		g1.read(fpartFile,npart1,'f');
		g2.read(fpartFile,npart2,'f');
		fpartFile >> x >> y >> z;
		g1.add_center(0,0,0,'f');
		g2.add_center(x,y,z,'f');
	}
	

	//  Add info to info.txt is it's not already there
	if ( !infoFound )
	{
		if (printStdWarning)
			printf("info.txt not found in %s Creating... \n",runDir.c_str());
		infoFileOut.open(infoName.c_str());
		infoFileOut << "Information file" << endl;
		infoFileOut << "sdss_name " << sdssName << endl;
		infoFileOut << "run_number " << runName << endl;
		infoFileOut << "galaxy1_number_particles " << npart1 << endl;
		infoFileOut << "galaxy2_number_particles " << npart2 << endl;
		infoFileOut << endl;
		infoFileOut << "Images_parameters_centers" << endl;
	   // printf("created new info file\n");
	}
	else
	{
		// append addition data to end
		infoFileOut.open(infoName.c_str(),ios::app);
		//printf("opening existing info file\n");
	}

	//  Perform some Internal calculations
	g1.calc_values();
	g2.calc_values();
	//printf("%s: %f %f %f %f\n",runName.c_str(),g1.xmax,g1.ymax,g2.xmax,g2.ymax);
	compare(g1,g2);
	//printf("calcue g values\n");

	//  Adjust point values to fit on image
	g1.adj_points(img.cols,img.rows,param.gaussian_size, g1.fpart);
	g2.adj_points(img.cols,img.rows,param.gaussian_size, g2.fpart);
	
	
	// Make Gaussian Blur mat.
	blur = Mat(param.gaussian_size,param.gaussian_size,CV_32F,Scalar(0));
	makeGaussianBlur();

	return true;
	
}


