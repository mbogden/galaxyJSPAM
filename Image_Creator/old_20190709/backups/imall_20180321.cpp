#include <unistd.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <omp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "Galaxy_Class.hpp"

using namespace cv;
using namespace std;


//  Global Variables
int thread_count = 1;
string sdss_directory;

//  Default values
string paramName = "param0001";
int gaussian_size = 15;
float gaussian_weight = 2.5;
float radial_constant = 6;
int norm_value = 4;
int image_rows = 1000;
int image_cols = 1000;


//  Function Prototypes

void getDir(vector<string> &myVec, string dirPath);
void readParam(ifstream& fin);
void compare(Galaxy &g1, Galaxy &g2);
void normalize_image(const Mat &img,Mat &dest, float max, float in);
bool removeFromDirectory(string in);
void readInfoFile(ifstream& fin, string &sdssName, int &npart1, int &npart2);

//  In command line.
//  ./a.out param.file sdss_directory
int main(int argc, char *argv[]){

    cout << endl;
    //  Get main Directory
    string mainDirPath = argv[1];

    // Open parameter file and save name
    string tempParam = argv[2];
	ifstream paramfile(tempParam.c_str());
    if (paramfile.fail())
        printf("Parameter file not found.  Using default param0001\n");
    else {
        size_t find = tempParam.find("param");
        paramName = tempParam.substr(find,9);
    }
	readParam(paramfile);
	paramfile.close();


    //  Add '/' to directory string if not already present
    string temp = mainDirPath.substr(mainDirPath.size()-1,1);
    if (temp != "/")
        mainDirPath = mainDirPath + '/';

    vector<string> runDirPath;
    getDir(runDirPath,mainDirPath);

    //  filter so only run directories are in vector
    runDirPath.erase(remove_if(runDirPath.begin(),runDirPath.end(),removeFromDirectory),runDirPath.end());


    if (runDirPath.size()==0){
        printf("No run directories found in %s\nExiting...\n",mainDirPath.c_str());
    }

    vector<Mat> images;
    vector<Mat> finals;

    //  Odd bug if each loop doesn't have it's own mat
    for( unsigned int i=0;i<runDirPath.size();i++){
        //printf("%s\n",runDirPath[i].c_str());
        images.push_back(Mat(image_rows,image_cols,CV_32F));
        finals.push_back(Mat(image_rows,image_cols,CV_32F));
    }


    //  Parallization implementation
    int numThreads = omp_get_num_threads();
    numThreads /= 4;
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int unsigned iRun=0; iRun < runDirPath.size(); iRun++)
        {
            string runDir = mainDirPath + runDirPath[iRun] + '/';
            //printf("%s\n",runDir.c_str());
            size_t findRun = runDir.find("run");
            string runName = runDir.substr(findRun,7);
            //printf("This should be the run Name: %s\n",runName.c_str());

            //  Variables
            //Mat img(image_rows,image_cols,CV_32F);
            //Mat dest(image_rows,image_cols,CV_32F);
            Mat img = images[iRun];
            Mat dest = finals[iRun];
            string fpartFileName, ipartFileName, sdssName, infoName, picName, tempStr;
            stringstream strm1, strm2;
            ifstream infoFileIn, ipartFile, fpartFile;
            ofstream infoFileOut;

            Galaxy g1,g2;
            int npart1, npart2;
            double x,y,z;
            bool picFound = false, infoFound = false;
            bool iFileFound = true, fFileFound = true;

            //printf("runDir: %s\n",runDir.c_str());


            //  Search run Directory for files
            vector<string> runFiles;
            getDir(runFiles,runDir);


            // find files
            for (unsigned int i=0; i<runFiles.size();i++)
            {
                size_t foundi = runFiles[i].find(".i.");
                size_t foundf = runFiles[i].find(".f.");
                if ( foundi != string::npos ){
                    ipartFileName = runFiles[i];
                    //printf("Found i file! %s\n",ipartFileName.c_str());
                }
                else if ( foundf != string::npos){
                    fpartFileName = runFiles[i];
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
            if (infoFound){
                infoFileIn.open(infoName.c_str());
                readInfoFile(infoFileIn, sdssName, npart1, npart2);
                infoFileIn.close();
                //printf("Found infoFile %s %d %d \n",sdssName.c_str(),npart1,npart2);
            }
            else {

                //  Parse data from particle file name
                size_t pos[4];
                pos[0] = ipartFileName.find(".");
                pos[1] = ipartFileName.find(".",pos[0]+1);
                pos[2] = ipartFileName.find(".",pos[1]+1);
                pos[3] = ipartFileName.find(".",pos[2]+1);

                sdssName = ipartFileName.substr(0,pos[0]);

                tempStr = ipartFileName.substr(pos[1]+1,pos[2]-pos[1]-1);
                strm1 << tempStr;
                strm1 >> npart1;

                tempStr = ipartFileName.substr(pos[2]+1,pos[3]-pos[2]-1);
                strm2 << tempStr;
                strm2 >> npart2;

                //printf("info File not found\n");
            }

            // Check if image already exists for current parameters

            picName = sdssName + '.' + paramName + ".model.png";

            for (unsigned int i=0; i<runFiles.size();i++)
            {
                if( runFiles[i].compare(picName)==0){
                    picFound = true;
                    printf("Image with %s already present in %s.\n",paramName.c_str(), runDir.c_str());
                }
            }


            ipartFileName = runDir + ipartFileName;
            fpartFileName = runDir + fpartFileName;


            //  Read Initial particle file
            ipartFile.open(ipartFileName.c_str());
            if (ipartFile.fail())    {
                printf("Initial Particle file not found or failed to open in %s\nSkipping...\n",runDir.c_str());
                iFileFound = false;
            }


            //  Final particle file
            fpartFile.open(fpartFileName.c_str());
            if (fpartFile.fail())  {
                printf("Final Particle file not found or failed to open in %s\nSkipping...\n",runDir.c_str());
                fFileFound = false;
            }

            if ( !picFound && iFileFound && fFileFound )
            {

                //printf("Pic not found.  Creating\n");
                picName = runDir + picName;



                g1.read(ipartFile,npart1,'i');
                g2.read(ipartFile,npart2,'i');
                ipartFile >> x >> y >> z;  // Grabbing center of g2
                g2.add_center(x,y,z,'i');


                //printf("read i file\n");


                g1.read(fpartFile,npart1,'f');
                g2.read(fpartFile,npart2,'f');
                fpartFile >> x >> y >> z;
                g2.add_center(x,y,z,'f');


                //printf("Read f file\n");

                //  Add info to info.txt is it's not already there
                if ( !infoFound )
                {
                    printf("info.txt not found in %s\nCreating... \n",runDir.c_str());
                    infoFileOut.open(infoName.c_str());
                    infoFileOut << "Information file" << endl;
                    infoFileOut << "sdss_name " << sdssName << endl;
                    infoFileOut << "run_number " << runName << endl;
                    infoFileOut << "galaxy1_number_particles " << npart1 << endl;
                    infoFileOut << "galaxy2_number_particles " << npart2 << endl;
                    infoFileOut << setprecision(10);
                    infoFileOut << "galaxy1_i_center " << g1.ix << ' ' << g1.iy << ' ' << g1.iz <<endl;
                    infoFileOut << "galaxy2_i_center " << g2.ix << ' ' << g2.iy << ' ' << g2.iz <<endl;
                    infoFileOut << "galaxy1_f_center " << g1.fx << ' ' << g1.fy << ' ' << g1.fz <<endl;
                    infoFileOut << "galaxy2_f_center " << g2.fx << ' ' << g2.fy << ' ' << g2.fz <<endl;
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
                compare(g1,g2);
                //printf("calcue g values\n");

                //  Adjust point values to fit on image
                g1.adj_points(img.cols,img.rows,gaussian_size, g1.fpart);
                g2.adj_points(img.cols,img.rows,gaussian_size, g2.fpart);
                //printf("Adjusted points\n");

                //  Write points to image
                g1.write(img,gaussian_size,gaussian_weight,radial_constant, g1.fpart);
                g2.write(img,gaussian_size,gaussian_weight,radial_constant, g2.fpart);
                //printf("Wrote points to image\n");

                //  Normalize pixel brightness and write image
                normalize_image(img,dest,g2.maxb,4);
                //printf("Writing image to %s\n",picName.c_str());
                dest.convertTo(dest,CV_8UC3,255.0);
                imwrite(picName,dest);

                //  Delete memory from Galaxies
                g1.delMem();
                g2.delMem();

                //  Write to info file about pixel centers
                infoFileOut << paramName << ' ' << int(g1.fx) << ' ' << int(g1.fy) << ' ' << int(g2.fx) << ' ' << int(g2.fy) << endl;
                infoFileOut.close();
                //printf("appended param to info file\n");

            }

            ipartFile.close();
            fpartFile.close();


            img.release();
            dest.release();

        }

    }

    cout<<endl;
    return 0;

}

void getDir(vector<string> &myVec, string dirPath){

    //  Search Directory for files
    DIR* dirp = opendir(dirPath.c_str());
    struct dirent * dp;
    while (( dp = readdir(dirp)) != NULL)
        myVec.push_back(dp->d_name);
}



void readParam(ifstream& fin)
{
    string str;

    while( fin >> str ){
        //cout << str << endl;
        if( str.compare("gaussian_size")==0)
            fin >> gaussian_size;
        else if ( str.compare("gaussian_weight")==0)
            fin >> gaussian_weight;
        else if ( str.compare("radial_constant")==0)
            fin >> radial_constant;
        else if ( str.compare("norm_value")==0)
            fin >> norm_value;
        else if ( str.compare("image_rows")==0)
            fin >> image_rows;
        else if ( str.compare("image_cols")==0)
            fin >> image_cols;
        else
            printf("Parameter %s not found\n",str.c_str());
    }
}


void compare(Galaxy &g1, Galaxy &g2){

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


void normalize_image(const Mat &img,Mat &dest, float max, float in){
	float vv, vscaled;
	for (int k=0;k<dest.rows;k++){
		for (int l=0;l<dest.cols;l++){
			vv =img.at<float>(k,l)	;
			vscaled = pow(vv/max,1/in);
			dest.at<float>(k,l) = vscaled;
		}
	}
}


// Accompaning function for clearnDirectory
bool removeFromDirectory(string in ){
    return in.compare(0,3,"run")!=0;
}


void readInfoFile(ifstream& fin, string &sdssName, int &npart1, int &npart2){

    string str;
    while( fin >> str ){

        if( str.compare("sdss_name")==0)
            fin >> sdssName;
        else if ( str.compare("run_number")==0)
            fin >> str;
        else if ( str.compare("galaxy1_number_particles")==0)
            fin >> npart1;
        else if ( str.compare("galaxy2_number_particles")==0)
            fin >> npart2;
    }
}

