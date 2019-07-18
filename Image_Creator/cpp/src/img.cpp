// Author: 		Matthew Ogden
// Created: 	Fall 2018
// Altered: 	11 July 2019
// About: 		This program is my main program for reading particle files
// 				Created by the SPAM program by Dr. Wallin
// 				Very... Very disorganized... Can't rewrite atm


#include "imgCreator.hpp"

using namespace cv;
using namespace std;

//  In command line.
//  ./image_creator.exe    run_directory    param_file
int main(int argc, char *argv[]){

  
	// Read arguments
	printf("Found %d arguments\n",argc);

	string paramLoc;
	imgCreator_input prepInput;


	for( int i=0; i<argc; i++){


	  if ( string( argv[i] ) == "-overwrite" ){
		printf("overwrite true\n");
		prepInput.overWrite = true;
	  }

	  else if ( string( argv[i] ) == "-runDir" ){
		prepInput.runDir = string(argv[i+1]);
	  }

	  else if ( string( argv[i] ) == "-p1" ){
		prepInput.p1Loc = string(argv[i+1]);
	  }

	  else if ( string( argv[i] ) == "-p2" ){
		prepInput.p2Loc = string(argv[i+1]);
	  }

	  else if ( string( argv[i] ) == "-o" ){
		prepInput.picName = string(argv[i+1]);
	  }

	  else if ( string( argv[i] ) == "-param" ){
		paramLoc = string(argv[i+1]);
		printf("Using image parameters found at %s\n",paramLoc.c_str());
	  }

	  else if ( string( argv[i] ) == "-info" ){
		prepInput.infoName = string(argv[i+1]);
	  }

	  else if ( string( argv[i] ) == "-mask" ){
		prepInput.makeMask = true;
	  }

	  else if ( string( argv[i] ) == "-unzip" ){
		prepInput.unzip = true;
	  }

	}


    cout << endl;
	cout << "\nIn img.cpp" << endl;


    // Open parameter file and save name
	ifstream paramfile(paramLoc.c_str());
    if (paramfile.fail()){
        printf("Parameter file not found.\nExiting...\n");
		return 0;
	}

	paramStruct param;
	param.loadParam(paramfile);
    param.print();
	paramfile.close();


    printf("About to create image!\n");

  
	string mainPath;
	ImgCreator myCreator(prepInput, param);			
	

	if( myCreator.new_prepare() ){
		printf("Creating Image\n");
		myCreator.g1.simple_write( myCreator.img, 'f');
		myCreator.g2.simple_write( myCreator.img, 'f');
		myCreator.make_mask("mask.png");
		printf("write info check\n");
		imwrite("test.png",myCreator.img);
		printf("write info check\n");
	  	myCreator.delMem();
		printf("delete memory check\n");
	}	
	else
	  printf("In img.cpp.  Creating image for %s failed.\n",mainPath.c_str());

    cout<<endl;
    return 0;

}

