# Creating usefule interactive webpages.

## Beginning Things

- How to view a .php file on system64.
  - 1. Create php file. 
  - 2. Place file in ~/public_html.  
  - 3. In terminal, change acces to file with "chmod 755 hello.php".  This lets any users view it
  - 4. In web browser, go to "cs.mtsu.edu/~mbo2d/hello.php"   (Replace mbo2d with your own username)
  - NOTE:  This php file can now be accessed by anyone around the world. So.... keep that in mind.  

- Getting test data
  - To get example images, directories, and info files...
	- Go to desired location for data.
	- Execute `scp system64.cs.mtsu.edu:/nfshome/mbo2d/Public/data_SE_1.zip .`
	- unzip this
  - Contents
	- dir 1: 58**14.  Contains current directory structure.
	- dir 2: goodImgs.  Just a directory with a bunch of images
	- dif 3: runDir.  Example of a current run dir that's full of info


- Beginning PHP file(s): Rough order of increasing complexity.  Can be seperate php files.

  - Display an image in hard code location.

  - Display all images found in a sub directory.
	- Format page to display images vertically so scrolling in easy.

  - Read single run directory.
	- display only the image ending in 'model.png'
	- open info.txt and display contents of info file
	  - Only display: sdss_name, gen#, run#, human_score
	- Add button that'll list all files in run folder?

  - Display model images found in sdss sub directories.
	- directory structure: mainDir -> sdssDir -> runDir


## Website 1: Quickly view images in SDSS
- Point to a main directory that may contain multiple pairs of galaxies
- Select which galaxy pair to look at
- Select range of runs of interest.  ( Ex. Only 0-100 or 20 - 40 )
- Select sample ratio ( Ex. Only view 1 out of 10 runs in order) 
- Displays select info next to image


## Webpage 2: To classify Images
- Similar to above, but do not display info.
- Next to images are Check boxes/buttons that allow users to classify images
  - Buttons Ex: No tidal distortions, Jumbled Mess, goood bridge/tail, etc
  - Upon button click, start new process or code that will append a line to info.txt 


## Webpage 3: GUI for pipeline
- To do considered once pipeline code is ready to operate.
- Select desired pipeline variables and run pipeline code
  - Galaxy Pairs, filters, comparison methods, etc
  - Auto read Parameter files in Input_Data/image_parameters/ 
  - Auto read Comparison methods available in Comparison_Methods/methods/ ...? 
	- Either go in and read function names or create/read a file kept in methods

