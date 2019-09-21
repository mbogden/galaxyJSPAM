# Streamline Model to Machine score process

## Write new code to initaliaze a model directory

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
###  Model Data  ### 	( Found in Input_Data/zoo_models/from_site/**sdss**.txt )
sdss_name 588***14 		( zoo model file name )
generation 0			( all 0 for now.  Have commandline argument for this )
run number 00100		( Line # in zoo model file )
model data 	1,3***5,3	( comma seperated values after tab in zoo model file)	
human_score 84.12434 	( zoo model file )
wins/total 23/32 		( zoo Model file ) 

###  Target Image Data  ###					( All info should be found within "Input_Data/targets/SDSS#/" with same sdss name)
/Input_Data/targets/*SDSS*/**.png 			( Use calibrated target image if available )
primary_center_1 508.57234 510.290384 		( "px" and "py" from meta file )
secondary_center_2 288.57234 340.290384 	( "sx" and "sy" from meta file )
primary_luminosity 5.306E10 				( "primaryLuminosity" from .pair file )
secondary_luminosity 5.856E11 				( "secondaryLuminosity" from  .pair file )

###  Model Image Parameters  ### 	( Use param found at Input_Data/image_parameters/param_2.txt for now )
param_2 v2 path_to/param.txt  		( param_name version_# path_to_file )

```


## Modify simulator_v2.py 
- Talk with me, Matthew, before doing
- Tasks
  - Simplify command line to accept...
	- run directory location
	- number of particles for each galaxy
  - leave other command line options for special circumstances
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


## Modify image_creator_v3.py
- Complete simulator_v2.py modification first.
- Tasks
  - read info.txt for most information
  - check for proper image_parameter_verion#. Exit if not compatible
  - Check if model image of that image_parameter exists.  Exit if already present.
  - Save model image and init image in 'model_images' folder.  All other images go to 'misc_images' folder.


## compare_v2.py
- Tasks
  - Retrieve Target information from info.txt
	- target location
	- target pixel centers
  - Check if comparison method has already been applied to model image in scores.csv file


## Begin Building Model Pipeline Code
- Core Pipeline
  - Simulate 100k pts -> Image Creations -> Machine Score

- Filter Pipeline ( Once filter is ready to test )
  - Run before core.
  - Simulate 10k pts -> apply filter
  - If filter decides model is "good", apply basic pipeline

