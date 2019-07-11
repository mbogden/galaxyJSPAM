// Author: 		Matthew Ogden
// Created: 	09 July 2019
// Altered: 	09 July 2019
// Description: 	This is my main program for creating images out of particle files


#include "imgCreator.hpp"

using namespace cv;
using namespace std;



//  In command line.
//  ./image_creator.exe    run_directory    param_file
int main(int argc, char *argv[]){

	bool overWriteImages = true;
	bool printStdWarnings = true;
	string sdss_directory;

    cout << endl;
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
        printf("Parameter file not found.\nExiting...\n");
		return -1;

	paramStruct param;
	param.loadParam(paramfile);
    param.print();

	paramfile.close();


    printf("About to create image!\n");
  
	ImgCreator myCreator(mainPath, param, overWriteImages, printStdWarnings);			
	if( myCreator.prepare() ){
		myCreator.makeImage2();
		myCreator.writeInfo();
		myCreator.delMem();
	}	
	else
	  printf("In img.cpp.  Creating image for %s failed.\n",mainPath.c_str());

    cout<<endl;
    return 0;

}

