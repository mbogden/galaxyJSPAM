#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;


//  Global Variables
string dirPath, targetPath, targetName, targetInfoPath, scorePath, hScorePath, temp;
string scoreDirName = "scores";
string targetDirName = "targets";
vector<string> mainDirNames, runDirPath, scoreDirNames, targetDirNames;
vector<int> runNum;
vector<float> humanScores;
bool hScoreFound = false;
ifstream targetInfoFile;
ofstream scoreFile;
Mat targetImg;
Point2f targetCenter[3];

//  Struct that contains useful information and processes for a single model image
struct imgData{
    Mat img, imgFinal, warpMat;
    string name;
    string path;
    string paramName;
    Point2f imageCenter[3];

    imgData(string nameIn, string pathIn){
        name = nameIn;
        path = pathIn;

        //  Determine parameter used to create image
        size_t fImg = name.find("param");
        if ( fImg != string::npos ){
            paramName = name.substr(fImg,9);
            //cout << paramName<<endl;
        }
        else {
            cout << "could not find parameter name in " << name << endl;
        }
    }

    //  Add pixel coordinates to centers of modeled galaxies
    void addCenter(int x1, int y1, int x2, int y2 ){

        imageCenter[0] = Point2f( x1, y1 );
        imageCenter[1] = Point2f( x2, y2 );
        imageCenter[2] = Point2f(imageCenter[0].x+(imageCenter[0].y-imageCenter[1].y)/3,imageCenter[0].y+(imageCenter[1].x-imageCenter[0].x)/3);  // Creates a triangle of points
    }


    //  Takes target image and centers to prepare model image for comparison
    bool loadImage(const Mat &targetImg, const Point2f targetCenter[]){

        img = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
        if ( !img.data ){
            cout << "Image failed to load " << path << " Skipping...\n";
            return false;
        }

        warpMat = getAffineTransform(imageCenter,targetCenter);
        warpAffine(img,imgFinal,warpMat,targetImg.size());

        return true;
    }

    //  Add center to original image.  Useful in troubleshooting
    void addCircle(){
        circle(img, imageCenter[0], 10, Scalar(255,255,255),2,8);
        circle(img, imageCenter[1], 10, Scalar(255,255,255),2,8);
        circle(img, imageCenter[2], 10, Scalar(255,255,255),2,8);
    }
};


struct runData{
    string path, infoName, sdssName, runName;
    vector<string> fileNames;
    vector<imgData> images;
    ifstream infoFile;
};

//  Function Prototypes
double rateDiff(const Mat &m1);
void getDir(vector<string> &vstr, string dirName);
bool processMainDir();
bool processTarget();
void processScore();
bool processRun( runData &myRun );

// compile and run info
// g++ -ggdb difference_code/diff.cpp -o diff.out `pkg-config --cflags --libs opencv`
// ./diff.out  main_directory  target_image
int main(int argc, char *argv[]){
    cout << endl;

    dirPath = argv[1];
    targetPath = argv[2];
    targetInfoPath = argv[3];

    if ( !processMainDir())  //  Finds and sorts through main directory.  Returns true or false to continue.
        return 0;
    if ( !processTarget())   //  Processes target data and info data.
        return 0;
    processScore();     //  Search score directory and check for human and score file

    //*****     Main Loop     *****//
    for ( unsigned int iRun=0 ; iRun < runDirPath.size() ; iRun++ )
    {
        //  Variables

        runData myRun;
        myRun.path = runDirPath[iRun];
        //cout << "mypath "<< myRun.path <<endl;
        //
        bool runGood = processRun(myRun);  // processes run directory for required files
        if (runGood){

            for(unsigned int iImg=0; iImg < myRun.images.size() ; iImg++ ){

                //  tell imgData to read image and alter it according to target data
                bool imgGood = myRun.images[iImg].loadImage(targetImg,targetCenter);
                if (imgGood){

                    Mat  diffMat;
                    double score;
                    absdiff(targetImg,myRun.images[iImg].imgFinal,diffMat);
                    score = rateDiff(diffMat);

                    temp = myRun.path+myRun.sdssName+'.' + myRun.runName + '.' + myRun.images[iImg].paramName+".diff.png",
                    imwrite(temp,diffMat);
                    diffMat.release();

                    scoreFile << myRun.sdssName <<','<< myRun.runName  <<','<< targetName  <<','<< myRun.images[iImg].name  <<','<< myRun.images[iImg].paramName  <<','<< "Absolute_Difference_v1"<<',' <<score<<',';
                    if (hScoreFound)
                        scoreFile<<humanScores[runNum[iRun]];
                    scoreFile << endl;

                }
                else
                    cout << "Image failed to open at " << myRun.images[iImg].path << " Skipping...\n";
            }
        }
    }

    cout <<endl;
    return 0;
}

//  Function is given the path to a directory and an empty vector of string.
//  Will fill the vector with names of files inside the directory
void getDir(vector<string> &dirNames, string dirStr){
    //printf("Loading directory %s into vector\n",dirStr.c_str());
    DIR* dirp = opendir(dirStr.c_str());
    struct dirent * dp;
    while(( dp = readdir(dirp)) != NULL)
        dirNames.push_back(dp->d_name);
}

//  Function does a rudimentary average of pixel brighteness across an image.
//  Unused calculation of standard deviation
double rateDiff(const Mat &m1){

    int max, min, var1, var2, N, val;
    double score;
	max =  m1.at<uint8_t>(0,0);
	min = max;
	var1 = 0;		// Sum of pixel differences
	var2 = 0; 	    // Sum of squared pixel differences
	float std;		// Standard Deviation
	N = m1.rows*m1.cols;	// Total number of pixels
	for (int i=0;i<m1.rows;i++){
		for (int j=0;j<m1.cols;j++){
			val = m1.at<uint8_t>(i,j);
            if(val>=0 && val<256){
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
    //cout<<endl;

	std = sqrt(  1.0*var2/N  - 1.0*var1*var1/(N*N) );

    score = 1.0-1.0*var1/N/255;
    //printf("Score: %f\n",score);
    return score;
}



//  Function does many processes for looking through main Directory
bool processMainDir(){

    string tempStr;

    //  Check if main directory ends with '/' and add if needed.
    tempStr = dirPath.substr(dirPath.size()-1,1);
    if (tempStr != "/")
        dirPath = dirPath +'/';
    printf("reading main directory %s\n",dirPath.c_str());

    //  Save target Name
    printf("Comparing to %s\n",targetPath.c_str());
    size_t fTargetName = targetPath.find_last_of('/');
    if ( fTargetName != string::npos)
        targetName = targetPath.substr(fTargetName+1,targetPath.size()-fTargetName-1);
    else
        targetName = targetPath;


    //******     Find and sort Directories     *****//
    getDir(mainDirNames,dirPath);
    bool foundTargetDir = false, foundScoreDir = false;

    for ( unsigned int i=0; i<mainDirNames.size();i++){
        //  Search for run directories
        size_t fRun = mainDirNames[i].find("run");
        if ( fRun != string::npos ){
            stringstream tempStrm;
            int tempInt;

            runDirPath.push_back(dirPath + mainDirNames[i] + '/' );
            tempStrm <<  mainDirNames[i].substr(3,mainDirNames[i].length()-3);
            tempStrm >> tempInt;
            runNum.push_back(tempInt);
            //cout << tempInt << ' ' << runNum.back() << endl;
        }

         //  Find score directory
        else if ( mainDirNames[i].compare(scoreDirName) == 0 ){
            temp = dirPath + mainDirNames[i] + '/';
            getDir(scoreDirNames, temp);
            foundScoreDir = true;
            //for (unsigned int j=0; j<scoreDirNames.size();j++)
            //    cout<<scoreDirNames[j]<<endl;
        }
        size_t fHumanScores = mainDirNames[i].find("humanscores.txt");
        if ( fHumanScores != string::npos ){
            hScoreFound = true;
            hScorePath = dirPath + mainDirNames[i];
            //cout << "Found human scores "<< hScorePath<<endl;
        }
        //cout << mainDirNames[i] <<endl;
    }

    //  Check if directories were found.
    if (runDirPath.size()==0){
        cout << "No run directories detected. Exiting\n";
        return false;
    }
    else if ( !foundScoreDir ){
        tempStr = dirPath + "scores";
        cout<<"Scores directory not found.  Creating...  ";
        if(mkdir(tempStr.c_str(),0777)==-1){// creating directory and checking success.
            cout<< "Error creating directory\n";
            return false;
        }
        else
            cout << "Created\n";
    }


    //Sort Run directories and associated numbers
    sort( runDirPath.begin(),runDirPath.end() );
    sort( runNum.begin() , runNum.end() );
    for (unsigned int i=0; i<runDirPath.size() && false/*disables for loop*/; i++){
        cout << runDirPath[i] << ' ' << runNum[i] << endl;
    }

    return true;
}

bool processTarget(){

    //  Load and check target image
    targetImg = imread(targetPath,CV_LOAD_IMAGE_GRAYSCALE);
    if ( !targetImg.data ){
        cout << "No data found for image at " << targetImg << endl;
        cout << "Exiting...\n\n"<<endl;
        return false;
    }

    //*****  Read target info  *****//
    targetInfoFile.open(targetInfoPath.c_str());
    if (targetInfoFile.fail()){
        cout << "Target information file not found at " << targetInfoPath << endl;
        cout << "Exiting...\n\n";
        return false;
    }

    //  Read info file searching pixel coordinates for centers of galaxies.
    size_t tFind[5];
    bool tFound[4] = {0,0,0,0};
    float tempFlt;
    int pixel[4];
    while(targetInfoFile>>temp){
        tFind[0] = temp.find("pxc");
        tFind[1] = temp.find("pyc");
        tFind[2] = temp.find("sxc");
        tFind[3] = temp.find("syc");
        tFind[4] = temp.find('.');
        for(int i=0;i<4;i++){
            if(tFind[i] != string::npos){
                tFound[i]=true;
                stringstream tempStrm;
                tempStrm << temp.substr(4,temp.length()-4);
                tempStrm >> tempFlt;
                pixel[i] = int(tempFlt);
                //cout <<temp<<' '<<lines[i]<<' ' << tempFlt << ' ' << pixel[i] << endl;
            }
        }
        //cout << temp << endl;
    }

    if (!tFound[0] || !tFound[1] || !tFound[2] || !tFound[3] ){
        cout << "Could not find all pixel coordinates in target info file: " << targetInfoPath << "Exiting...\n";
        return 0;
    }

    //  Processing and Saving Coordinates
    pixel[3] = targetImg.cols - pixel[3];  // Changing starting count from lower left to upper left.
    targetCenter[0] = Point2f(pixel[0],pixel[1]);
    targetCenter[1] = Point2f(pixel[2],pixel[3]);
	targetCenter[2] = Point2f(targetCenter[0].x+(targetCenter[0].y-targetCenter[1].y)/3,targetCenter[0].y+(targetCenter[1].x-targetCenter[0].x)/3);

    //  Troubleshooting purposes only
    //  Testing to see if point appear to be on centers

    if(false){
        circle(targetImg, targetCenter[0], 10, Scalar(255,255,255),2,8);
        circle(targetImg, targetCenter[1], 10, Scalar(255,255,255),2,8);
        circle(targetImg, targetCenter[2], 10, Scalar(255,255,255),2,8);
        imwrite(dirPath+targetDirName+"/circles.png",targetImg);
    }

    return true;
}

void processScore(){

    //*****  Search for score and human score files *****//
    bool scoreFound = false;
    for (unsigned int i=0 ; i<scoreDirNames.size() ; i++){
        if (scoreDirNames[i].compare("scores.csv") == 0 ){
            scoreFound = true;
            scorePath = dirPath + "scores/" + scoreDirNames[i];
        }

        if (scoreDirNames[i].compare("humanscores.txt") == 0 ){
            hScoreFound = true;
            hScorePath = dirPath + "scores/"+ scoreDirNames[i];
        }
    }

    if (!scoreFound){
        scorePath = dirPath +"scores/scores.csv";
        printf("Score file not found.  Creating... \n");
        scoreFile.open(scorePath.c_str());
        scoreFile << "sdss_name,run_number,target_image,simulated_image,parameters,comparison_method,machine_score,human_score\n";
    }
    else
        scoreFile.open(scorePath.c_str());

    if (hScoreFound){
        ifstream hScoreFile(hScorePath.c_str());
        float tempFlt;
        char tempChar;
        string tempStr;
        while( hScoreFile >> tempFlt >> tempStr ){
            humanScores.push_back(tempFlt);
            //cout <<  tempFlt << ' ' << humanScores.back()<<endl;;
        }
        hScoreFile.close();
    }
    else {
        cout << "Human scores not found\n";
    }
}



bool processRun( runData &myRun ){

    getDir(myRun.fileNames, myRun.path);

    //*****  Check for info and image Files *****//
    size_t fImg;
    bool imgFound = false;
    bool infoFound = false;
    string infoTemp;

    for ( unsigned int i=0 ; i < myRun.fileNames.size() ; i++ ){
        //printf("%s\n",myRunNames[i].c_str());
        fImg = myRun.fileNames[i].find(".model.png");
        if ( fImg != string::npos ){
            myRun.images.push_back(imgData(myRun.fileNames[i], myRun.path + myRun.fileNames[i]));
            imgFound = true;
        }
        if ( myRun.fileNames[i].compare("info.txt") == 0 )
            infoFound = true;
    }

    if ( !imgFound ){
        printf("No images found in %s Skipping directory...\n",myRun.path.c_str());
        return false;
    }
    if ( !infoFound ){
        printf("Info file not found in %s Skipping directory...\n",myRun.path.c_str());
        return false;
    }
    else
    {
        myRun.infoName = myRun.path + "info.txt";
        myRun.infoFile.open(myRun.infoName.c_str());
        if( myRun.infoFile.fail() ){
            infoFound = false;
            printf("info.txt failed to open in %s Skipping....\n",myRun.path.c_str());
            return false;
        }
    }

    //  Read until reaching needed information
    myRun.infoFile >> infoTemp;
    while (infoTemp.compare("sdss_name")!=0)
        myRun.infoFile >> infoTemp;
    myRun.infoFile >> myRun.sdssName;
    while (infoTemp.compare("run_number")!=0)
        myRun.infoFile >> infoTemp;
    myRun.infoFile >> myRun.runName;
    while (infoTemp.compare("Images_parameters_centers") != 0)
        myRun.infoFile >> infoTemp;
    int t1,t2,t3,t4;  //  Temp integers
    while ( myRun.infoFile >> infoTemp >> t1>>t2>>t3>>t4 ){
        for(unsigned int i=0; i<myRun.images.size(); i++){
            if(myRun.images[i].paramName.compare(infoTemp)==0){
                myRun.images[i].addCenter(t1,t2,t3,t4);
                //cout << infoTemp << " should match "<< myRun.images[i].name << endl;
            }
        }
    }

    return true;
}


