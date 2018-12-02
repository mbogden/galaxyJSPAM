// Author: Matt Ogden
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "imgClass.hpp"

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
imgClass target;


struct runData{
    string path, infoName, sdssName, runName;
    vector<string> fileNames;
    vector<imgClass> images;
    ifstream infoFile;
};

//  Function Prototypes
double rateDiff(const Mat &m1);
double rateDiff_v2(const Mat &m1);
Scalar getMSSIM( const Mat& i1, const Mat& i2);
void getDir(vector<string> &vstr, string dirName);
bool processMainDir();
bool processTarget();
void processScore();
bool processRun( runData &myRun );

// compile and run info
// g++ -ggdb difference_code/diff.cpp -o diff.out `pkg-config --cflags --libs opencv`
// ./diff.out  main_directory  target_image target_image_info.txt
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
                bool imgGood = myRun.images[iImg].loadImage();
                if (imgGood){

                    Mat  diffMat;
                    double score;
                    Scalar simScore;
                    myRun.images[iImg].processImg(target);
                    myRun.images[iImg].normBrightness(target);
                    myRun.images[iImg].createAltSizes();
                    absdiff(target.imgFinal,myRun.images[iImg].imgFinal,diffMat);
                    //score = rateDiff(diffMat);
                    //simScore = getMSSIM(target.img256,myRun.images[iImg].img256);
                    //cout << simScore[0] << endl;
                    imwrite(myRun.path + "sample.png",diffMat);
                    //temp = myRun.path+myRun.sdssName+'.' + myRun.runName + '.' + myRun.images[iImg].paramName+".diff.png",
                    //imwrite(temp,diffMat);
                    //diffMat.release();

                    score = rateDiff(diffMat);
                    score = (score -.95)*20;
                    if (score<0)
                        score = 0;

                    scoreFile << myRun.sdssName <<','<< myRun.runName  <<','<< targetName  <<','<< myRun.images[iImg].name  <<','<< myRun.images[iImg].paramName  <<','<< "absDiff_v2"<<',' <<score<<',';
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


Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
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

double rateDiff_v2(const Mat &m1){

    int val;
    long int Sum=0;
    int count;
    double score;
	for (int i=0;i<m1.rows;i++){
		for (int j=0;j<m1.cols;j++){
			val = m1.at<uint8_t>(i,j);
            if (val>0 && val<256){
                count++;
                Sum += val;
            }
            else if (val >= 256)
                cout << val <<endl;
        }
	}

    score = 1.0-1.0*Sum/count/255;
    return score;
    //return score;
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
    target.path = targetPath;
    bool tempBool = target.loadImage();
    if (!tempBool){
         cout << "No data found for image at " << target.path << endl;
        cout << "Exiting...\n\n"<<endl;
        return false;
    }
    target.processTargetImg();
    target.createAltSizes();

    //*****  Read target info  *****//
    targetInfoFile.open(targetInfoPath.c_str());
    if (targetInfoFile.fail()){
        cout << "Target information file not found at " << targetInfoPath << endl;
        cout << "Exiting...\n\n";
        return false;
    }

    bool newTarget = true;
    if (newTarget){
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
            return false;
        }

        //  Processing and Saving Coordinates
        pixel[3] = target.imgFinal.cols - pixel[3];  // Changing starting count from lower left to upper left.
        target.addCenter(pixel[0],pixel[1],pixel[2],pixel[3]);

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
            myRun.images.push_back(imgClass(myRun.fileNames[i], myRun.path + myRun.fileNames[i]));
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


