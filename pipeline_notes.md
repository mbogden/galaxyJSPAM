# Streamline Model to Machine score process

## (Todo Jack) Oct 31 Updating parameter

-Tasks: Modify run creator code again
  - Modify para_v3_default.txt


## (TO-DO Matthew ) compare_v2.py
- Tasks
  - Retrieve Target information from info.txt
	- target location
	- target pixel centers
  - Check if comparison method has already been applied to model image in scores.csv file


## (Jack) Begin Building Model-to-Score Code
- Python program.  Use python template I'll provide with notes.
  - This code will read default values in a file and begin running
  - Simulate 100k pts -> Image Creations -> Machine Score

- Add Filter Pipeline ( Once filter is ready to test )
  - Simulate 10k pts -> apply filter
  - If filter decides model is "good", apply basic pipeline


# Completed Work!

## (DONE - Jack) Write new code to initaliaze a model directory
- Reads and processes
  - Zoo models @ Input_Data/zoo_models/from_site/*SDSS*.txt
  - target images and select files @ Input_Data/target_images/*SDSS*/
  - Image parameter file @ Input_Data/image_parameters/

- Creates

  - Directory Structure
	- mainDir (input var) -> sdssDir (#) -> genDir (0) -> runDir (line # in file)

  - Extra Folders in runDir
	- particle_files
	- model_images
	- misc_images

  - info.txt
	- Adds all of the information formated below
	- NOTE: Will be changing image_parameter info as I'm changing. 


```
###  Model Data  ### 		( Found in Input_Data/zoo_models/from_site/**sdss**.txt )
sdss_name 588***14 		( zoo model file name )
generation 0			( all 0 for now.  Have commandline argument for this )
run_number 00100		( Line # in zoo model file )
model_data 	1,3***5,3	( comma seperated values after tab in zoo model file)	
human_score 84.12434 	( zoo model file )
wins/total 23/32 		( zoo Model file ) 

###  Target Image Data  ###			( All info should be found within "Input_Data/targets/SDSS#/" with same sdss name)
/Input_Data/targets/*SDSS*/**.png 		( Use calibrated target image if available )
primary_center_1 508.57234 510.290384 		( "px" and "py" from meta file )
secondary_center_2 288.57234 340.290384 	( "sx" and "sy" from meta file )
primary_luminosity 5.306E10 			( "primaryLuminosity" from .pair file )
secondary_luminosity 5.856E11 			( "secondaryLuminosity" from  .pair file )

###  Model Image Parameters  ### 	( Use param found at Input_Data/image_parameters/param_2.txt for now )
param_2 v2 path_to/param.txt  		( param_name version_# path_to_file )

```



## (DONE - Jack) Oct 8, To-Do:
- In README.  Specify what file1, file2, file3 are.  

- In code, rename "SDSS input file(s)" to "Galaxy Zoo Model file/folder"
- In code, rename "SDSS data dir" to "SDSS Target Information folder"

- Add a commandline argument for the directory I want to place all the folders.

- Only create SDSS directories that have all available target information from 'Input_Data/targets' and model information 'Input_Data/zoo_models'

- The code is still creating too many run folders that have empty human_scores and wins/totals.
  - Ex.  58\*14 should only have about 1300ish run folders.  It goes into the 2800's currently when I run it. 

- In GUI, I still cannot select the location for the "sdss data dir".  
  - It pings and give erro:
	XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-mbo2d' qt.qpa.xcv: QXcbConnection: XCB erro: 1 (BadREquest), sequence: 167, resource id: 142, major code: 130 (Unknown), minor code: 47

## (Done Jack) Oct 9 To-Do
- Info files need to have 'model\_data' and not 'model data'


## (DONE - Jack) Modify simulator_v2.py 
- Tasks
  - Simplify command line to accept...
	- path to run directory
	- number of particles for each galaxy
  - read info.txt to get model info
  - Check if particle files of that quantity already exist in particle folder.
	- Exit with statement if they already exist.
  - Change active directory to particle_file folder before calling JSPAM code
	- jspam automatically creates its files in whatever directory the program was called from
  - Rename particle files
	- Initial particles. From: `a\_#.000`  To: `#pts_pts.000`.  (Ex. 100000_pts.000)
	- Final particles.   From: `a\_#.101`  To: `#pts_pts.101`.  (Ex. 20000_pts.101)
  - Zip particle files together in one zip file named `#pts_pts.zip` (Ex. 1000_pts.zip)
  - Delete unzipped particle files.


## (DONEish - Matthew) Modify image_creator_v3.py ( I will likely work on this )
- Complete simulator_v2.py modification first.
- Tasks
  - read info.txt for most information
  - check for proper image_parameter_verion#. Exit if not compatible
  - Check if model image of that image_parameter exists.  Exit if already present.
  - Save model image and init image in 'model_images' folder.  All other images go to 'misc_images' folder.


## (Done - Jack) General Modifications
- Make the following 3 programs all be callable as a main function or module for importing
  - Ex. if __name__ == '__main__': 
  - Simulator


## (Done) Oct 24 - RunCreator Modifications
- Create new Folder under sdss# along with gen folders
   - sdss# -> sdssParameters

  - Create File 'parameters.txt'

- Copy "Input_Data/image_parameters/param_v3_default.txt" into sdssParameters folder.
- Copy "Input_Data/targets/{sdss#}/sdss{#}.png" to "{sdss#}/sdssParameters/target_zoo.png"

- Do not add the following info from info.txt file.
  - write the following info into parameters.txt instead.

```
###  Target Image Data  ###
target_zoo.png 513 514 309 514  ( imgName  center1x c1y c2x c2y )
primary_luminosity 5.3e+10
seconary_luminosity 5.07e+10

###  Model Image Parameters  ###
default param_v3_default.txt
```

