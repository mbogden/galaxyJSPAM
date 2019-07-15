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
bool overWriteImages = false;
bool readRunInfoFile = false;
string sdss_directory;

//  Default values
struct param{
    string paramName;
    int gaussian_size;
    float gaussian_weight;
    float radial_constant;
    int norm_value;
    int image_rows;
    int image_cols;
};


//  Function Prototypes

void getDir(vector<string> &myVec, string dirPath);
void readParam(ifstream& fin, param &paramIn);
void compare(Galaxy &g1, Galaxy &g2);
void normalize_image(const Mat &img,Mat &dest, float max, float in);
bool removeFromDirectory(string in);
void readInfoFile(ifstream& fin, string &sdssName, int &npart1, int &npart2);

//  In command line.
//  ./a.out param.file sdss_directory
int main(int argc, char *argv[]){

    cout << endl;
    //  Get run Directory
    string mainDirPath = argv[1];
    string runDir = mainDirPath;

    // Open parameter folder
    string paramPath = argv[2];
    string tempParam;

    //  Add '/' to directory string if not already present
    string temp = mainDirPath.substr(mainDirPath.size()-1,1);
    if (temp != "/")
        mainDirPath = mainDirPath + '/';

    //  Add '/' to param path if not already present
    temp = paramPath.substr(paramPath.size()-1,1);
    if (temp != "/")
         paramPath = paramPath + '/';

    vector<string> paramNames;
    getDir(paramNames,paramPath);
    paramNames.erase(remove_if(paramNames.begin(),paramNames.end(),removeFromDirectory),paramNames.end());
    cout << paramNames.size()<<endl;

    string fpartFileName, ipartFileName, sdssName, infoName, picName, tempStr;
    stringstream strm1, strm2;
    ifstream ipartFile, fpartFile;
    Galaxy g1,g2;
    int npart1, npart2;
    double x,y,z;
    bool iFileFound = false, fFileFound = false;
    bool multPFiles = false;

    //  Search run Directory for files
    vector<string> runFiles;
    getDir(runFiles,mainDirPath);


    // find files
    for (unsigned int i=0; i<runFiles.size();i++){
        size_t foundi = runFiles[i].find(".i.");
        size_t foundf = runFiles[i].find(".f.");
        if ( foundi != string::npos ){
            ipartFileName = mainDirPath + runFiles[i];
            if (iFileFound == true)
                multPFiles = true;
            iFileFound = true;
            //printf("Found i file! %s\n",ipartFileName.c_str());
        }
        else if ( foundf != string::npos){
            fpartFileName = mainDirPath + runFiles[i];
            if (fFileFound)
                multPFiles = true;
            fFileFound = true;
            //printf("Found f file! %s\n",fpartFileName.c_str());
        }
    }

    //  Parse data from particle file name
    //  assumes sdssName.runName.state.g1Part.g2Part.txt
    size_t pos[5];
    pos[0] = ipartFileName.find(".");
    pos[1] = ipartFileName.find(".",pos[0]+1);
    pos[2] = ipartFileName.find(".",pos[1]+1);
    pos[3] = ipartFileName.find(".",pos[2]+1);
    pos[4] = ipartFileName.find(".",pos[4]+1);

    sdssName = ipartFileName.substr(0,pos[0]);

    tempStr = ipartFileName.substr(pos[2]+1,pos[3]-pos[2]-1);
    strm1 << tempStr;
    strm1 >> npart1;

    tempStr = ipartFileName.substr(pos[3]+1,pos[4]-pos[3]-1);
    strm2 << tempStr;
    strm2 >> npart2;

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

    if (multPFiles)
        printf("Multiple sets of point files found in %s skipping...\n", runDir.c_str());

    infoName = mainDirPath + "info.txt";
    ofstream infoFileOut;
    infoFileOut.open(infoName.c_str());
    infoFileOut << "Information file" << endl;
    infoFileOut << "sdss_name " << sdssName << endl;
    infoFileOut << "run_number run000" << endl;
    infoFileOut << "galaxy1_number_particles " << npart1 << endl;
    infoFileOut << "galaxy2_number_particles " << npart2 << endl;
    infoFileOut << setprecision(10);
    infoFileOut << "galaxy1_i_center " << g1.ix << ' ' << g1.iy << ' ' << g1.iz <<endl;
    infoFileOut << "galaxy2_i_center " << g2.ix << ' ' << g2.iy << ' ' << g2.iz <<endl;
    infoFileOut << "galaxy1_f_center " << g1.fx << ' ' << g1.fy << ' ' << g1.fz <<endl;
    infoFileOut << "galaxy2_f_center " << g2.fx << ' ' << g2.fy << ' ' << g2.fz <<endl;
    infoFileOut << endl;
    infoFileOut << "Images_parameters_centers" << endl;

    //printf("Pic not found.  Creating\n");

    g1.read(ipartFile,npart1,'i');
    g2.read(ipartFile,npart2,'i');
    ipartFile >> x >> y >> z;
    g1.add_center(0,0,0,'i');
    g2.add_center(x,y,z,'i');

    g1.read(fpartFile,npart1,'f');
    g2.read(fpartFile,npart2,'f');
    fpartFile >> x >> y >> z;
    g1.add_center(0,0,0,'f');
    g2.add_center(x,y,z,'f');

    ipartFile.close();
    fpartFile.close();

    g1.calc_values();
    g2.calc_values();
    compare(g1,g2);

    //  Adjust point values to fit on image
    g1.adj_points(1000,1000,25, g1.fpart);
    g2.adj_points(1000,1000,25, g2.fpart);

    vector<Mat> images;
    vector<Mat> finals;

    for(unsigned int i=0; i<paramNames.size(); i++){
        images.push_back(Mat(1000,1000,CV_32F));
        finals.push_back(Mat(1000,1000,CV_32F));
    }



    int numThreads = omp_get_num_threads();
    numThreads /= 2;
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for(unsigned int i=0; i<paramNames.size(); i++){

            string tempStr = paramPath + paramNames[i];
            param myParam;
            Mat img = images[i];
            Mat dest = finals[i];

            ifstream paramfile(tempStr.c_str());
            if (paramfile.fail())
                printf("%s not found.\n",temp.c_str());
            else {
                readParam(paramfile,myParam);

                string picName = paramNames[i] + ".model.png";

                g1.write(img,myParam.gaussian_size,myParam.gaussian_weight,myParam.radial_constant, g1.fpart);
                g2.write(img,myParam.gaussian_size,myParam.gaussian_weight,myParam.radial_constant, g2.fpart);

                normalize_image(img,dest,g2.maxb,myParam.norm_value);

                dest.convertTo(dest,CV_8UC3,255.0);
                imwrite(picName,dest);

                #pragma omp critical
                {
                    infoFileOut << paramNames[i].substr(0,9) << ' ' << int(g1.fx) << ' ' << int(g1.fy) << ' ' << int(g2.fx) << ' ' << int(g2.fy) << endl;
                }

                img.release();
                dest.release();

            }
            paramfile.close();
        }
    }


    infoFileOut.close();
    //  Delete memory from Galaxies
    g1.delMem();
    g2.delMem();

    return 0;
    //printf("appended param to info file\n");


}

void getDir(vector<string> &myVec, string dirPath){

    //  Search Directory for files
    DIR* dirp = opendir(dirPath.c_str());
    struct dirent * dp;
    while (( dp = readdir(dirp)) != NULL)
        myVec.push_back(dp->d_name);
}



void readParam(ifstream& fin, param &paramIn)
{
    string str;

    while( fin >> str ){
        //cout << str << endl;
        if( str.compare("gaussian_size")==0)
            fin >> paramIn.gaussian_size;
        else if ( str.compare("gaussian_weight")==0)
            fin >> paramIn.gaussian_weight;
        else if ( str.compare("guassian_weight")==0)
            fin >> paramIn.gaussian_weight;
        else if ( str.compare("radial_constant")==0)
            fin >> paramIn.radial_constant;
        else if ( str.compare("norm_value")==0)
            fin >> paramIn.norm_value;
        else if ( str.compare("image_rows")==0)
            fin >> paramIn.image_rows;
        else if ( str.compare("image_cols")==0)
            fin >> paramIn.image_cols;
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
    return in.compare(0,5,"param")!=0;
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

