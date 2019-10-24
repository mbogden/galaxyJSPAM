# NEURAL NETWORKS!

## General Purpose Classificatier

### (DONE) Inital Training of classifier

- Input 
  1. Image

- Output classification
  - "good" or "bad" classification

- Inital Training process

  - First training set can be found at '/nsfhome/mbo2d/Public/training_image_set_1.zip'.  Upzip
	- Two directories.
	  - goodDir: Contains image pairs for "good" tidal distortions
	  - badDir: Contains image pairs for "bad" tidal distortions

	- Image pair format: images with same sdss name and run number are a pair
	  - model image: sdssName_runNumber_model.png
	  - init image : sdssName_runNumber_init.png

  - I can create better training sets later.


### Inital Code that uses trained classifier

- Input 
  1. Single image

- Output
  1. Print good or bad classification

### Advanced Training

- Input 
  1. good Images
  2. bad Images



## Tidal Distortion filter  
## Jumpled Mess Filter
- Very similar to above
- Read in two images and identify if they are "too jumbled"
- May need to wait until classification website is up and working to get a training set.
- Write code to classify image and store classification in info.txt


## Integrate to work with data from Classification website
- Classification website will save it's results in info.txt in some manner.
- Once operating, integrate training filters to work with folder structure and files to get data.


## Build Neural network to work with WNDCHRM
- This will require that we implement WNDCHRM.  
