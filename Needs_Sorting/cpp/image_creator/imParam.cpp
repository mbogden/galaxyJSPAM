#include "imgCreator.hpp"

using namespace std;
using namespace cv;

//  Function Prototypes
void getDir(vector<string> &myVec, string dirPath);   
bool removeFromDirectory(string in);

//  In command line.
//  ./image_creator.exe    sdss_directory    param_file
int main(int argc, char *argv[]){

	bool overWriteImages = true;
	bool printStdWarnings = true;
	string sdss_directory;
	string temp;

    cout << endl;
	
    // Open parameter file and save name
    string mainPath = argv[1];
	temp = mainPath.substr(mainPath.size()-1,1);
    if (temp != "/")
        mainPath = mainPath + '/';
	
    //  Get main Directory
    string paramPath = argv[2];
    temp = paramPath.substr(paramPath.size()-1,1);
    if (temp != "/")
        paramPath = paramPath + '/';



	cout << "mainPath " << mainPath << endl;
	cout << "paramPath " << paramPath << endl;


	//  Grab param files
    vector<string> paramNames;	// container for param names. 
    getDir(paramNames,paramPath); // get all files/folders in directory.
    paramNames.erase(remove_if(paramNames.begin(),paramNames.end(),removeFromDirectory),paramNames.end());  // filter out non parameter files.
	sort(paramNames.begin(), paramNames.end()); // sort

    if (paramNames.size()==0){
        printf("No run directories found in %s\nExiting...\n",paramPath.c_str());
		return 0;
    }
	else
		printf("%d parameters found\n",int(paramNames.size()));


	paramStruct paramInit;
	temp = paramPath + paramNames[0];
	ifstream initParamFile(temp.c_str());
	if (initParamFile.fail())
	{
		printf("Failed to load param %s \n",temp.c_str());
	}
	else{
		paramInit.loadParam(initParamFile);
		//paramInit.print();
		initParamFile.close();
	}

	//  Parallelization implementation
    int numThreads;
	#pragma omp parallel
		numThreads = omp_get_num_threads();
    numThreads /= 2;
	
	printf("Using %d threads\n",numThreads);

	ImgCreator myCreator(1, mainPath, paramInit, overWriteImages, printStdWarnings);
	
	if(!myCreator.prepare()){
		cout << "failed to prepare imgCreator Classs \n";
		return 0;
	}


	for (int unsigned iRun=0; iRun < paramNames.size(); iRun++)
	{
		if(printStdWarnings)
			printf("In loop %d of %d\n",int(iRun),int(paramNames.size()));
		string tempStr = paramPath + paramNames[iRun];
		ifstream paramFile(tempStr.c_str());
		if (paramFile.fail())
		{
			printf("Failed to load param %s \n",tempStr.c_str());
		}
		else{
			paramStruct param;
			param.loadParam(paramFile);
			paramFile.close();
			
			myCreator.changeParam(param);		
			myCreator.makeImage2();
			myCreator.writeInfoPlus(); 				

		}
	}
   
	
	
	myCreator.delMem();			

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

// Accompanying function for cleanDirectory
bool removeFromDirectory(string in ){
    return in.compare(0,5,"param")!=0;
}



