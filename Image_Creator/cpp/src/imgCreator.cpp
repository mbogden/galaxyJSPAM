// Author: Matt Ogden
#include "imgCreator.hpp"
#include "spam_galaxy.hpp"

using namespace std;
using namespace cv;



ImgCreator::ImgCreator(imgCreator_input in, paramStruct paramIn){		
	
	npart1 = npart2 = 0;
	imageParamHeaderPresent = false;


	runDir = in.runDir;
	param = paramIn;
	printStdWarning = in.printSTDWarning;
	overWriteImages = in.overWrite;
	makeMask = in.makeMask;
	numThreads = 1;
	unzip = in.unzip;
	
	if (in.makeMask)
	  printf("imgCreator, in.makemask = true\n");

	if (makeMask)
	  printf("imgCreator, make mask = true\n");

	img = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	mask_1 = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	dest = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	blur = Mat(param.gaussian_size,param.gaussian_size,CV_32F, Scalar(0));
	picFound = infoFound = iFileFound = fFileFound = multPFiles = false;
	
	//printf("In ImgCreator Class. In directory %s\n", runDir.c_str());
	
}


bool ImgCreator::new_prepare(){
	
	if ( runDir.empty() ){
	  printf(" ImageCreator currently requires a run directoly to be passed\n");
	  return false;
	  
	}

	//  Search run Directory for files
	getDir(runFiles,runDir);
	string tempStr; 
	
	// Check if image already exists for current parameters
	picName = param.name + "_model.png";



	string ipartZipName, fpartZipName;

	// find files
	for (unsigned int i=0; i<runFiles.size();i++)
	{
		size_t foundi = runFiles[i].find(".101");
		size_t foundf = runFiles[i].find(".000");
		size_t foundiz = runFiles[i].find("101.zip");
		size_t foundfz = runFiles[i].find("000.zip");

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


		else if ( foundiz != string::npos){
			ipartZipName = runFiles[i];
			//printf("Found i zip file! %s\n", ipartZipName.c_str());
		}

		else if ( foundfz != string::npos){
			fpartZipName = runFiles[i];
			//printf("Found f zip file! %s\n", fpartZipName.c_str());
		}


		else if ( runFiles[i].compare(picName)==0){
			picFound = true;
			if (printStdWarning)
				printf("Image with %s already present in %s.\n",param.name.c_str(), runDir.c_str());
			if (!overWriteImages)
				return false;
		}
	}
	

	if (multPFiles){
		printf("Multiple sets of point files found in %s.\nExiting...\n", runDir.c_str());
		return false;
	}



	if ( unzip ) 
	{

		char unzipCmd[256];

		string tempStr = runDir + ipartZipName;
		sprintf(unzipCmd, "unzip -oq %s -d %s", tempStr.c_str(), runDir.c_str());
		//printf("%s\n",unzipCmd);
		system(unzipCmd);

		tempStr = runDir+fpartZipName;
		sprintf(unzipCmd, "unzip -oq %s -d %s", tempStr.c_str(), runDir.c_str() );
		//printf("%s\n",unzipCmd);
		system(unzipCmd);

		runFiles.clear();
		getDir(runFiles,runDir);

		// find files
		for (unsigned int i=0; i<runFiles.size();i++)
		{
			size_t foundi = runFiles[i].find(".101");
			size_t foundf = runFiles[i].find(".000");


			if ( foundi != string::npos ){
				ipartFileName = runFiles[i];
				iFileFound = true;
				//printf("Found i file! %s\n",ipartFileName.c_str());
			}

			else if ( foundf != string::npos){
				fpartFileName = runFiles[i];
				fFileFound = true;
				//printf("Found f file! %s\n",fpartFileName.c_str());
			}

		}

	}
	
	if ( (! iFileFound) || (! fFileFound) ) { 
		printf("Point Particles files could not be found in %s Exiting...\n",runDir.c_str());
		return false;
	}


	readInfoFile();

	stringstream strm1, strm2;

	ipartFileName = runDir + ipartFileName;
	fpartFileName = runDir + fpartFileName;

	//  Read Initial particle file
	ipartFile.open(ipartFileName.c_str());
	if (ipartFile.fail())    {
		printf("Initial Particle file failed to open in %s\nExiting...\n",runDir.c_str());
		return false;
	}

	//  Final particle file
	fpartFile.open(fpartFileName.c_str());
	if (fpartFile.fail())  {
		printf("Final Particle file failed to open in %s\nExiting...\n",runDir.c_str());
		return false;
	}


	
	picName = runDir + picName;
	
	//  Read in particle files
  
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



	//  Perform some Internal calculations
	g1.calc_values();
	g2.calc_values();
	//printf("%s: %f %f %f %f\n",runName.c_str(),g1.xmax,g1.ymax,g2.xmax,g2.ymax);
	compare(g1,g2);
	//printf("calcue g values\n");

	//  Consider adjusting so galaxies are centered on image 
	//  Adjust point values to fit on image
	g1.adj_points(img.cols,img.rows,param.gaussian_size, g1.fpart);
	g2.adj_points(img.cols,img.rows,param.gaussian_size, g2.fpart);

	
	if (makeMask){
	  g1.adj_points(img.cols,img.rows,param.gaussian_size, g1.ipart);
	  g2.adj_points(img.cols,img.rows,param.gaussian_size, g2.ipart);
	}
	
	// Make Gaussian Blur mat.
	blur = Mat(param.gaussian_size,param.gaussian_size,CV_32F,Scalar(0));
	makeGaussianBlur();

	return true;
	
}


bool ImgCreator::readInfoFile(){
  infoName = runDir + "info.txt";
  ifstream infoFile;
  string line, tempStr;

  infoFile.open(infoName);

  while( getline( infoFile, line) ){
	
	stringstream ss(line);


	//cout << line << endl;
	if (line.empty())
	  continue;

	else if ( line.compare("Image Parameters") == 0){
	  imageParamHeaderPresent = true;
	}

	ss >> tempStr;

	if ( tempStr.compare( "sdss_name" ) == 0 )
	  ss >> sdssName;

	else if ( tempStr.compare( "run_number") == 0  )
	  ss >> runName;

	else if ( tempStr.compare( "g1_num_particles") == 0  )
	  ss >> npart1;

	else if ( tempStr.compare( "g2_num_particles" )== 0  )
	  ss >> npart2;	

  }

  infoFile.close();

	if ( ( npart1 == 0) || (npart2 == 0) ) 
	{
	  printf("Did not find both particle counts\n");
	  return false;
	}

	return true;
  
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
ImgCreator::ImgCreator(string p1LocIn, string p2LocIn, paramStruct paramIn, bool overWriteIn, bool warnIn){
	
	ipartFileName = p1LocIn;
	fpartFileName = p2LocIn;
	printStdWarning = warnIn;
	overWriteImages = overWriteIn;
	param = paramIn;
	numThreads = 1;
	
	img = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
	dest = Mat(param.image_rows,param.image_cols,CV_32F, Scalar(0));
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

	if (makeMask){
	  string maskLoc = runDir + "mask.png";
	  make_mask(maskLoc);
	}

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


void ImgCreator::make_mask(string saveLocName){

	printf("In make_mask\n");
	
	g1.write_mask(mask_1);
	g2.write_mask(mask_1);
	printf("wrote dots\n");
	mask_1.convertTo(mask_1,CV_8UC3,255.0);
	imwrite(saveLocName,mask_1);

}
void ImgCreator::makeImage2(string saveLocName){
	
	g1.dot_write(img,'f');
	g2.dot_write(img,'f');
	GaussianBlur(img, img, Size(param.gaussian_size,param.gaussian_size), param.gaussian_weight, 0);
	normalize_image2();
	dest.convertTo(dest,CV_8UC3,255.0);
	imwrite(saveLocName,dest);

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
	infoFileOut.open(infoName.c_str(),ios::app);

	if ( ! imageParamHeaderPresent )
	  infoFileOut << "Image Parameters" << endl;

	infoFileOut << param.name << ' ' << int(g1.fx) << ' ' << int(g1.fy) << ' ' << int(g2.fx) << ' ' << int(g2.fy) << endl;
}


void ImgCreator::writeInfoPlus(){
	//  Write to info file about pixel centers

	infoFileOut.open(infoName.c_str(),ios::app);

	if ( ! imageParamHeaderPresent )
	  infoFileOut << "Image Parameters" << endl;

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


	// delete unzipped files to save space
	if ( unzip ){
		char rmCmd[256];

		string tempStr = ipartFileName;
		sprintf(rmCmd, "rm %s", tempStr.c_str());
		printf("%s\n",rmCmd);
		system(rmCmd);

		tempStr = fpartFileName;
		sprintf(rmCmd, "rm %s", tempStr.c_str());
		printf("%s\n",rmCmd);
		system(rmCmd);
	}

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
		size_t foundi = runFiles[i].find("101");
		size_t foundf = runFiles[i].find("000");
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


