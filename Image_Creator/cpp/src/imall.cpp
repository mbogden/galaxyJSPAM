// Author: Matt Ogden
#include "imgCreator.hpp"

using namespace cv;
using namespace std;

//  Function Prototypes
void getDir(vector<string> &myVec, string dirPath);   
bool removeFromDirectory(string in);

//  In command line.
//  ./image_creator.exe    sdss_directory    param_file
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
        printf("Parameter file not found.  Using default param0001\n");

	paramStruct param;
	param.loadParam(paramfile);
    param.print();
	paramfile.close();


	//  Grab run folders
    vector<string> runNames;
    getDir(runNames,mainPath); // get all files/folders in directory.
    runNames.erase(remove_if(runNames.begin(),runNames.end(),removeFromDirectory),runNames.end());  // filter.
	sort(runNames.begin(), runNames.end()); // sort
	
    if (runNames.size()==0){
        printf("No run directories found in %s\nExiting...\n",mainPath.c_str());
		return 0;
    }


    //  Parallization implementation
	int numThreads;
	#pragma omp parallel
		numThreads = omp_get_num_threads();
    numThreads /= 2;  // Only use half of available threads. 
	numThreads = 16;
	
	printf("Using %d threads\n",numThreads);
	
    #pragma omp parallel num_threads(numThreads)  
	{		
		#pragma omp for
		for (int unsigned iRun=0; iRun < runNames.size(); iRun++)
		{
			//printf("In thread %d \n",omp_get_thread_num());
			string tempStr = mainPath + runNames[iRun]+'/';
			//ImgCreator *myCreator;
			//myCreator = new ImgCreator(tempStr, param, overWriteImages, printStdWarnings);			
			ImgCreator myCreator(tempStr, param, overWriteImages, printStdWarnings);			
			if( myCreator.prepare() ){
				myCreator.makeImage2();
				myCreator.writeInfo();
				myCreator.delMem();
			}	
			//delete myCreator;
		}
    }

    cout<<endl;
    return 0;

}

void getDir(vector<string> &myVec, string dirPath){
     // Search Directory for files
    DIR* dirp = opendir(dirPath.c_str());
    struct dirent * dp;
    while (( dp = readdir(dirp)) != NULL)
        myVec.push_back(dp->d_name);
}

// Accompaning function for clearnDirectory
bool removeFromDirectory(string in ){
    return in.compare(0,3,"run")!=0;
}



