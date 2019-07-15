#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>

//#include "opencv2/utility.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;


double rateDiff(const Mat &m1);
void getDir(vector<string> &vstr, string dirName);


// $: ./diff.exe  main_directory  target_image
int main(int argc, char *argv[]){

    cout << endl;
    //  Variables

    string dirPath, targetPath, targetName, targetInfoPath, scorePath, temp;
    vector<string> mainDirNames, runDirPath, scoreDirNames, targetDirNames;

    ifstream targetInfoFile;
    ofstream scoreFile;

    Mat targetImg;
    Point2f targetCenter[3];


    /*****     Reading in Directories and Target Info     *****/

    //  Get main directory
    dirPath = argv[1];
    temp = dirPath.substr(dirPath.size()-1,1);  //  Check it has '/' at end
    if (temp != "/")
        dirPath = dirPath +'/';
    printf("reading main directory %s\n",dirPath.c_str());

    targetPath = argv[2];
    printf("Comparing to %s\n",targetPath.c_str());
    size_t fTargetName = targetPath.find_last_of('/');
    targetName = targetPath.substr(fTargetName+1,targetPath.size()-fTargetName-1);


    getDir(mainDirNames,dirPath);
    //for(unsigned int i=0; i < mainDirNames.size() ; i++ )
    //    printf("%s\n",mainDirNames[i].c_str());


    //******     Find and sort Directories     *****//
    for ( unsigned int i=0; i<mainDirNames.size();i++){

        //  Search for run directories
        size_t fRun = mainDirNames[i].find("run");
        if ( fRun != string::npos ){
            temp = dirPath + mainDirNames[i] + '/';
            runDirPath.push_back(temp);
        }

        //  Find target image directory
        else if ( mainDirNames[i].compare("target_images") == 0 ){
            temp = dirPath + mainDirNames[i] + '/';
            getDir(targetDirNames, temp);
            //for (unsigned int j=0; j<targetDirNames.size();j++)
            //    cout<<targetDirNames[j]<<endl;
        }

        //  Find score directory
        else if ( mainDirNames[i].compare("scores") == 0 ){
            temp = dirPath + mainDirNames[i] + '/';
            getDir(scoreDirNames, temp);
            //for (unsigned int j=0; j<scoreDirNames.size();j++)
            //    cout<<scoreDirNames[j]<<endl;
        }
    }
    //for (unsigned int i=0; i<runDirNames.size();i++)
    //    cout << runDirNames[i] << endl;

    if (runDirPath.size()==0){
        cout << "No run directories detected. Exiting\n";
        return 0;
    }


    //*****  Load and check target image  *****//
    targetImg = imread(targetPath,CV_LOAD_IMAGE_GRAYSCALE);
    if ( !targetImg.data ){
        cout << "No target image found at " << targetImg << endl;
        return 0;
    }
    //imshow("window",targetImg);
    //waitKey(0);

    //*****  Read target info  *****//
    targetInfoPath = dirPath + "target_images/info.txt";
    targetInfoFile.open(targetInfoPath.c_str());
    if (targetInfoFile.fail()){
        cout << "Target information file not found at " << targetInfoPath << endl;
        cout << "Exiting\n\n";
        return 0;
    }

    size_t fTarget;
    bool infoFound = false;
    int tx1, tx2, ty1, ty2;
    while (targetInfoFile>>temp)
    {
        fTarget = targetPath.find(temp);
        if (fTarget != string::npos){
            infoFound = true;
            targetInfoFile >> tx1 >> ty1 >> tx2 >> ty2;
            targetCenter[0] = Point2f(tx1,ty1);
            targetCenter[1] = Point2f(tx2,ty2);
		    targetCenter[2] = Point2f(targetCenter[0].x+(targetCenter[0].y-targetCenter[1].y)/3,targetCenter[0].y+(targetCenter[1].x-targetCenter[0].x)/3);
            //printf("\n%d %d %d %d\n",x1,y1,x2,y2);
        }
    }

    if (!infoFound){
        printf("No info found in %s for target image %s\nExiting...\n",targetInfoPath.c_str(),targetPath.c_str());
        return 0;
    }


    //*****  Search and Open score file *****//
    bool scoreFound;
    for (unsigned int i=0 ; i<scoreDirNames.size() ; i++){
        if (scoreDirNames[i].compare("scores.csv") == 0 )
            scoreFound = true;
    }

    scorePath = dirPath + "scores/scores.csv";

    if (!scoreFound){
        printf("Score file not found.  Creating... \n");
        scoreFile.open(scorePath.c_str());
        scoreFile << "sdss_name,run_number,target_image,simulated_image,parameters,comparison_method,machine_score,human_score\n";
    }
    else
        scoreFile.open(scorePath.c_str(),ios::app);





    //*****     Main Loop     *****//

    //printf("\nEntering Main Loop with %d Run Directories\n", runDirPath.size());

    for ( unsigned int iRun=0 ; iRun < runDirPath.size() ; iRun++ )
    {
        //  Variables
        bool foundInfo, foundImages;

        string myPath, myInfoName, sdssName, runName, paramName, infoTemp;
        vector<string> myFileNames, imgNames;
        ifstream myInfoFile;

        myPath = runDirPath[iRun];
        getDir(myFileNames,myPath);

        //*****  Check for info and image Files *****//
        size_t fImg;
        bool imgFound = false;
        bool infoFound = false;
        for ( unsigned int i=0 ; i < myFileNames.size() ; i++ ){
            //printf("%s\n",myRunNames[i].c_str());
            fImg = myFileNames[i].find(".model.png");
            if ( fImg != string::npos ){
                imgNames.push_back(myFileNames[i]);
                imgFound = true;
            }
            if ( myFileNames[i].compare("info.txt") == 0 )
                infoFound = true;
        }
        if ( !imgFound ){
            printf("No images found in %s \nSkipping...\n",myPath.c_str());
        }
        if ( !infoFound ){
            printf("info.txt not found in %s \nSkipping...\n",myPath.c_str());
        }
        else
        {
            myInfoName = myPath + "info.txt";
            myInfoFile.open(myInfoName.c_str());
            if( myInfoFile.fail() ){
                infoFound = false;
                printf("info.txt failed to open in %s\nSkipping....\n",myPath.c_str());
            }
        }


        //*****  Go through Images  *****//
        if ( imgFound && infoFound )
        {
            //  Read until reaching needed information
            myInfoFile >> infoTemp;
            while (infoTemp.compare("sdss_name")!=0)
                myInfoFile >> infoTemp;
            myInfoFile >> sdssName;
            //printf("%s\n",sdssName.c_str());

            while (infoTemp.compare("run_number")!=0)
                myInfoFile >> infoTemp;
            myInfoFile >> runName;

            while (infoTemp.compare("Images_parameters_centers") != 0)
            {
                //printf("Found %s\n",infoTemp.c_str());
                myInfoFile >> infoTemp;
            }

            for(unsigned int iImg=0; iImg < imgNames.size() ; iImg++ )
            {
                int x1,y1,x2,y2;
                double score;
                Point2f imageCenter[3];
                Mat warpMat, imageIn, imageFinal, diffMat;

                //imgNames[iImg] = myPath + imgNames[iImg];
                //printf("%s\n",imgNames[iImg].c_str());

                imageIn = imread(myPath +imgNames[iImg],CV_LOAD_IMAGE_GRAYSCALE);
                if ( !imageIn.data ){
                    cout << "Image failed to open at " << imgNames[iImg] << endl;
                }
                else
                {
                    myInfoFile >> paramName >> x1 >> y1 >> x2 >> y2;
                    //printf("%d %d %d %d\n",x1,y1,x2,y2);
                    imageCenter[0] = Point2f( x1 , y1 );
                    imageCenter[1] = Point2f( x2 , y2 );
		            imageCenter[2] = Point2f(imageCenter[0].x+(imageCenter[0].y-imageCenter[1].y)/3,imageCenter[0].y+(imageCenter[1].x-imageCenter[0].x)/3);
                    warpMat = getAffineTransform(imageCenter,targetCenter);
                    warpAffine(imageIn,imageFinal,warpMat,targetImg.size());

                    absdiff(targetImg,imageFinal,diffMat);
                    score = rateDiff(diffMat);

                    //Mat inv = targetImg;
                    //Mat img2 = imageFinal;
                    //Mat diff2 = diffMat;

                    //bitwise_not(targetImg,inv);
                    //bitwise_not(imageFinal,img2);
                    //bitwise_not(diffMat,diff2);

                    //imwrite(myPath+paramName+"target_inv.jpg",targetImg);
                    //imwrite(myPath+paramName+"img_inv.jpg",img2);
                    //imwrite(myPath+paramName+"diff_inv.jpg",diff2);


                    //printf("%f\n",score);
                    //imwrite(myPath+"imageFinal.jpg",imageFinal);
                    imwrite(myPath+paramName+".diff.png",diffMat);
                    /*
                    imshow("before image",imageIn);
                    imshow("Target",targetImg);
                    imshow("image",imageFinal);
                    */
                    //imshow("difference",diffMat);
                    //waitKey(0);

                    scoreFile << sdssName <<','<< runName  <<','<< targetName  <<','<< imgNames[iImg]  <<','<< paramName  <<','<< "diff_v1"<<',' <<score << endl;;
                }

                warpMat.release();
                imageIn.release();
                imageFinal.release();
                diffMat.release();
            }
        }
    }

    //printf("*****  3  *****\n");
    cout << endl;
    return 0;
}

    /*
	Mat sim_raw = imread("img/img.r2n5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat sim_warp;

	Point2f sc[3];
	rc[1] = Point2f(178,250);

	Mat warp_mat = getAffineTransform(sc,rc);
	warpAffine(sim_raw,sim_warp,warp_mat,real.size());

	Mat diff;
	absdiff(real, sim_warp, diff);
	ratediff(diff);

    cout << real.type()<< ' '<< real.channels()<<' '<<real.depth()<<endl;
    cout << sim_warp.type()<< ' '<< sim_warp.channels()<<' '<<sim_warp.depth()<<endl;
    cout << diff.type()<< ' '<< diff.channels()<<' '<<diff.depth()<<endl;

	imshow( "Sim", sim_warp);
	imshow( "Real", real);
	imshow( "Diff", diff);
	waitKey(0);


	imwrite("diff.jpg",diff);
	return 0;

	//~ int thickness = 2;
	//~ int lineType = 8;

	//~ for(int i=0;i<3;i++){
		//~ circle( real,rc[i],10, Scalar( 255, 255, 255 ), thickness, lineType );
		//~ circle( sim_raw,sc[i],10, Scalar( 255, 255, 255 ), thickness, lineType );
	//~ }

	imwrite("diff.jpg",diff);

	return 0;
}

*/

void getDir(vector<string> &dirNames, string dirStr){
    //printf("Loading directory %s into vector\n",dirStr.c_str());
    DIR* dirp = opendir(dirStr.c_str());
    struct dirent * dp;
    while(( dp = readdir(dirp)) != NULL)
        dirNames.push_back(dp->d_name);
}


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

