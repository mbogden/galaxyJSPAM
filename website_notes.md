# Creating usefule interactive webpages.

## Visualize Photos.
- Beginning:
  - Create PHP file.  
  - Displays all images found in a directory.
  - Format page to scroll through images.
  - Point to sdss directory and display model images found in subdirectories.
	- directory structure: mainDir -> sdssDir -> genDir -> runDir
  - Display select info from info.txt files in subdirectories with image
	- sdss_name, gen#, run#, human_score

- Useful additions
  - Add button to specify directory of interest? 

- Webpage 1: Scientist's quick view of models
  - Select range of runs of interest
  - Select sample ratio ( Ex. 1 out of 10 runs) 

- Webpage 2: To classify Images
  - Add Check boxes/buttons that allow users to classify image
	- Ex.  No tidal distortions, Jumbled Mess, goood bridge/tail, etc
  - Append classificaiton response in model's info.txt

- GUI for Score Analysis
  - Displays interactive plot? 
  - Can select models and immediately view mdoel images

- GUI for pipeline
  - To do considered once pipeline code is ready to operate.
  - Select desired pipeline variables and run pipeline code
	- Galaxy Pairs, filters, comparison methods, etc
	- Auto read Parameter files in Input_Data/image_parameters/ 
	- Auto read Comparison methods available in Comparison_Methods/methods/ ...? 
	  - Either go in and read function names or create/read a file kept in methods
