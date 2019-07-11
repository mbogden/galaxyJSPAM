// Author: 		Matthew Ogden
// Created: 	Fall 2018
// Altered: 	11 July 2019
// About: 		This program is my main program for reading particle files
// 				Created by the SPAM program by Dr. Wallin


#include "imgCreator.hpp"

using namespace cv;
using namespace std;

//  In command line.
//  ./image_creator.exe    run_directory    param_file
int main(int argc, char *argv[]){

  
	bool overWriteImages = false;
	bool printStdWarnings = true;

	string p1Loc = "", p2Loc = "";
	string saveImageLoc = "";

	// Read arguments
	printf("Found %d arguments\n",argc);

	for( int i=0; i<argc; i++){

	  //printf("Arg %d: %s\n",i,argv[i]);

	  if ( string( argv[i] ) == "-overwrite" ){
		printf("overwrite true\n");
		overWriteImages = true;
	  }

	  if ( string( argv[i] ) == "-p1" ){
		p1Loc = string(argv[i+1]);
		printf("Using initial particle file %s\n",p1Loc.c_str());
	  }

	  if ( string( argv[i] ) == "-p2" ){
		p2Loc = string(argv[i+1]);
		printf("Using final particle file %s\n",p2Loc.c_str());
	  }

	  if ( string( argv[i] ) == "-o" ){
		saveImageLoc = string(argv[i+1]);
		printf("Saving image to %s\n",saveImageLoc.c_str());
	  }

	}


	return 0;

    cout << endl;
	cout << "\nIn img.cpp" << endl;
    //  Get main Directory
    string mainPath = argv[1];

	//  Add '/' to directory string if not already present
    string temp = mainPath.substr(mainPath.size()-1,1);
    if (temp != "/")
        mainPath = mainPath + '/';

    // Open parameter file and save name
    string tempParam = argv[2];
	ifstream paramfile(tempParam.c_str());
    if (paramfile.fail())
        printf("Parameter file not found.  Using default param0001\n");

	paramStruct param;
	param.loadParam(paramfile);
    param.print();
	paramfile.close();


    printf("About to create image!\n");
  
	ImgCreator myCreator(mainPath, param, overWriteImages, printStdWarnings);			
	printf("Constructor check\n");

	if( myCreator.prepare() ){
		printf("prepare check\n");
		myCreator.makeImage2("testName.png");
		printf("make image check\n");
		myCreator.writeInfo();
		printf("write info check\n");
	  	myCreator.delMem();
		printf("delete memory check\n");
	}	
	else
	  printf("In img.cpp.  Creating image for %s failed.\n",mainPath.c_str());

    cout<<endl;
    return 0;

}

