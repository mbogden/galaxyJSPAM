'''
    Author:	 Matthew Ogden
    Created:	21 Feb 2020
Description:	Module for accessing information and files stored on disk.
'''

from os import path, listdir
import json
from copy import deepcopy

import pandas as pd
import numpy as np
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD    
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

from sys import path as sysPath
sysPath.append('../')


from pprint import PrettyPrinter
pp = PrettyPrinter(width=41, compact=True)
pprint = pp.pprint

# For loading in Matt's general purpose python libraries
import general_module as gm

def test():
    print("IM: Hi!  You're in Matthew's information module for SPAM")
# End test print

def main(arg):

    print("IM: Hi!  You're in Matthew Information module's main function.")

    # Print stuff
    if arg.printAll:
        arg.printArg()
        gm.test()

    if arg.runDir != None:

        rInfo = run_info_class( \
                runDir = arg.runDir, \
                printAll = True, \
                newInfo = arg.get('newInfo', None ), \
                newRun = arg.get( 'newRun', False ), \
                )

        if rInfo.status == False:
            return

        rInfo.printInfo()
        if getattr( arg, 'updateInfo', False ):
            rInfo.updateInfo()

        rInfo.printScores()

    if arg.targetDir != None:

        tInfo = target_info_class( \
                targetDir = arg.targetDir, \
                printAll = arg.printAll, \
                newInfo = arg.get('newInfo', False ), \
            )

        if tInfo.status:
            print("IM: target_info_class good")


# End main

class run_info_class:

    rDict = None	# Primary dict of information contained in info.json
    initDict = None	# State of json upon initial reading.
    baseDict = None

    status = False  # State of this class.  For initiating.
    
    # Main directories
    runDir = None
    
    ptsDir = None
    imgDir = None
    miscDir = None
    wndDir = None
    tmpDir = None

    # Useful files in directory
    infoLoc = None
    baseLoc = None

    wndFitLoc = None
    wndAllLoc = None
    
    tLink = None
    
    infoHeaders = [ 'run_id', 'model_data', 'machine_scores' ]


    def __init__( self, runDir=None, rArg = gm.inArgClass(), \
                 printBase=None, printAll=None, ):

        # get arguments
        if printBase == None:  self.printBase = rArg.printBase
        else:  self.printBase = printBase
        
        if printAll == None:  self.printAll = rArg.printAll
        else:  self.printAll = printAll
            
        if runDir != None:
            rArg.runDir = runDir
        
        if rArg.get('tInfo',None) != None: 
            self.tInfo = rArg.tInfo

        # Avoiding confusing things
        if self.printAll: self.printBase = True

        if self.printAll: 
            print("IM: run_info_class.__init__")
            print("\t - runDir: " , runDir )

        # initialize directory structure for run
        dirGood = self.initRunDir( rArg )

        if not dirGood:
            if self.printBase: 
                print("WARNING: IM: Run_info_class: Directory not set up properly")
                gm.tabprint('runDir: %s' % rArg.runDir )
                gm.eprint("WARNING: IM: Run_info_class: Directory not set up properly")
                gm.etabprint('runDir: %s' % rArg.runDir )
            self.status = False
            return
                 
        # Read info file
        if path.exists( self.infoLoc ) and self.rDict == None:

            if self.printAll: print('\t - Reading Info file.')
            with open( self.infoLoc, 'r' ) as iFile:
                self.rDict = json.load( iFile )

        if self.rDict != None:
            self.status = True

        if self.printAll: 
            print("IM: Run.__init__: Initalized: %s" % self.status)

    # end __init__
    
    
    def initRunDir( self, rArg ):
        
        self.runDir = gm.validPath( rArg.runDir )
            
        # Double check if run directory is valid
        if self.runDir == None:
            if self.printBase:
                print("IM: WARNING: run_info_class.__init__: Invalid run dir")
                tabprint('runDir: - %s' % rArg.runDir )
                gm.eprint("IM: WARNING: run_info_class.__init__: Invalid run dir")
                gm.etabprint('runDir: - %s' % rArg.runDir )
            return False

        # Print stuff
        if self.printAll:
            print("IM: run.initRunDir")
            print("\t - runDir: %s" % rArg.runDir )

        # Save base directory
        if self.runDir[-1] != '/': self.runDir += '/'

        # Hard code location of main objects
        self.ptsDir = self.runDir + 'particle_files/'
        self.imgDir = self.runDir + 'model_images/'
        self.miscDir = self.runDir + 'misc_images/'
        self.wndDir = self.runDir + 'wndchrm_files/'
        self.tmpDir = self.runDir + 'tmp/'
        self.infoLoc = self.runDir + 'info.json'
        self.baseLoc = self.runDir + 'base_info.json'
        self.wndFitLoc = self.wndDir + 'wndchrm_all.fit'
        self.wndAllLoc = self.wndDir + 'wndchrm_all.csv'
    
        # If newInfo or newBase
        if rArg.get('newInfo',False):
            self.newRunSetup(rArg)

        # Print stuff if needed
        if self.printAll:
            print("\t -  runDir: (%s) %s" % ( path.exists( self.runDir ), self.runDir ) )
            print("\t -  ptsDir: (%s) %s" % ( path.exists( self.ptsDir ), self.ptsDir ) )
            print("\t -  imgDir: (%s) %s" % ( path.exists( self.imgDir ), self.imgDir ) )
            print("\t - miscDir: (%s) %s" % ( path.exists( self.miscDir ), self.miscDir ) )
            print("\t - wndDir: (%s) %s" % ( path.exists( self.wndDir ), self.wndDir ) )
            print("\t - infoLoc: (%s) %s" % ( path.exists( self.infoLoc ), self.infoLoc ) )
            print("\t - baseLoc: (%s) %s" % ( path.exists( self.baseLoc ), self.baseLoc ) )

        # Check if things are working
        dirGood = True

        if not path.exists( self.ptsDir ) or not path.exists( self.imgDir ) or not path.exists( self.miscDir ) or not path.exists( self.wndDir ):
            dirGood = False

        # If you made it this far.  
        return dirGood

    # End initialize run directory structure
    
    # If asked to create new info file
    def newRunSetup( self, rArg ):
        
        from os import remove, mkdir
        from shutil import copyfile
        
        # Remove current info file
        if path.exists( self.infoLoc ): remove( self.infoLoc )

        # Copy base info file.
        if path.exists( self.baseLoc ): copyfile( self.baseLoc, self.infoLoc )     
        
        # Create directories if not found
        if not path.exists( self.ptsDir ): mkdir( self.ptsDir )
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.miscDir ): mkdir( self.miscDir )
        if not path.exists( self.wndDir ): mkdir( self.wndDir ) 
        if not path.exists( self.tmpDir ): mkdir( self.tmpDir )
            
        # Read info file
        if not path.exists( self.infoLoc ):
            return False
        
        with open( self.infoLoc, 'r' ) as iFile:
                self.rDict = json.load( iFile )
        
        if self.rDict.get( "run_id", None ) == None:
            return False
        
        if self.rDict.get( "model_data", None ) == None:
            return False

            
    def __del__(self,):
        pass

    def delTmp(self,):

        from os import remove
        from shutil import rmtree
        
        for f in listdir( self.tmpDir ):
            
            fLoc = self.tmpDir + f
            
            if path.isfile(fLoc):
                remove(fLoc)
                
            elif path.isdir(fLoc):
                rmtree(fLoc)
                
            
    def readParticles( self, ptsName ):
        
        if self.printAll:
            print("IM.run_info_class.readParticles:")
            gm.tabprint('pts name: %s' % ptsName )
        
        # See if particle files are in temp folder unzipped
        iLoc, fLoc = self.findPtsLoc( ptsName = ptsName )

        # If found, read and return
        if iLoc != None and fLoc != None:
            pts1 = pd.read_csv( iLoc, header=None, delim_whitespace=True ).values
            pts2 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values
            return pts1, pts2
        
        
        # Else need to unzip a file
        ptsZipLoc = self.findZippedPtsLoc( ptsName = ptsName )

        if self.printAll: tabprint("Loading points from file: %s"%ptsZipLoc)

        if ptsZipLoc == None:
            if self.printBase: 
                print("WARNING: IM.run_info_class.readParticles:")
                gm.tabprint("zipped points not found: %s"%ptsZipLoc)
                gm.eprint("WARNING: IM.run_info_class.readParticles:")
                gm.etabprint("zipped points not found: %s"%ptsZipLoc)
            return None, None


        from zipfile import ZipFile

        
        zipLoc = self.findPtsLoc( ptsName )        
        if zipLoc == None:
            return None

        # Check if zip files need rezipping
        self.delTmp()
        rezip = False

        with ZipFile( ptsZipLoc ) as zip:

            for zip_info in zip.infolist():

                if zip_info.filename[-1] == '/':
                    continue
                if len( zip_info.filename.split('/') ) > 1:
                    rezip = True

                zip_info.filename = path.basename( zip_info.filename)
                zip.extract(zip_info, self.tmpDir)

        if rezip:
            from os import remove
            remove(zipLoc)

            with ZipFile( zipLoc, 'w' ) as zip:
                for f in listdir( self.tmpDir ):
                    fLoc = self.tmpDir + f
                    zip.write( fLoc, path.basename(fLoc) )

        files = listdir( self.tmpDir )
        pts1 = None
        pts2 = None

        for f in files:
            fLoc = self.tmpDir + f

            if '.000' in f:
                pts1 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values

            if '.101' in f:
                pts2 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values
        
        return pts1, pts2
    
    
    def findPtsLoc( self, ptsName ):

        # Unique Loc
        iLoc = self.tmpDir + '%s_pts.000' % ptsName
        fLoc = self.tmpDir + '%s_pts.101' % ptsName

        if gm.validPath( iLoc ) and gm.validPath( fLoc ):
            return ( iLoc, fLoc )
        else:
            return (None, None)

    # End findPtsFile
        
            

    def findZippedPtsLoc( self, ptsName ):

        ptsLoc = self.ptsDir + '%s.zip' % ptsName
        oldLoc1 = self.ptsDir + '%s_pts.zip' % ptsName
        oldLoc2 = self.ptsDir + '100000_pts.zip'
        
        # Updating how particile files are saved
        if path.exists( oldLoc1 ) or path.exists( oldLoc2 ):
            from os import rename
            
            if path.exists( oldLoc1 ):
                rename( oldLoc1, ptsLoc )
                
            if path.exists( oldLoc2 ):
                rename( oldLoc2, ptsLoc )

        if gm.validPath( ptsLoc ):
            return ptsLoc
        else:
            return None

    # End findPtsFile
    
    def getParticles( self, ptsName ):
        
        from zipfile import ZipFile    

        zipLoc = self.findPtsLoc( ptsName )        
        if zipLoc == None:
            return None
             
        # Check if zip files need rezipping
        self.delTmp()
        rezip = True
               
        with ZipFile( zipLoc ) as zip:
            
            for zip_info in zip.infolist():
                
                if zip_info.filename[-1] == '/':
                    continue
                if len( zip_info.filename.split('/') ) > 1:
                    rezip = True
                    
                zip_info.filename = path.basename( zip_info.filename)
                zip.extract(zip_info, self.tmpDir)

        if rezip:
            from os import remove
            remove(zipLoc)
            
            with ZipFile( zipLoc, 'w' ) as zip:
                for f in listdir( self.tmpDir ):
                    fLoc = self.tmpDir + f
                    zip.write( fLoc, path.basename(fLoc) )
                    
        files = listdir( self.tmpDir )
        pts1 = None
        pts2 = None
        
        for f in files:
            fLoc = self.tmpDir + f
            
            if '.000' in f:
                pts1 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values
                
            if '.101' in f:
                pts2 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values
        
        return pts1, pts2

    
    def getModelImage( self, imgName = 'zoo_0', imgType = 'model', toType=np.float32, overWrite=False ):
        
        # Create place to store images if needed.
        if self.get('img',None) == None: self.img = {}            
        if self.get('init',None) == None: self.init = {}
        
        # Return model image if already loaded.
        if imgType == 'model' and not overWrite:
            mImg = self.img.get(imgName,None)
            if type(mImg) != type(None) and mImg.dtype == toType:
                return deepcopy( mImg )
        
        elif imgType == 'init' and not overWrite:            
            img = self.init.get(imgName,None)
            if type(img) != type(None) and img.dtype == toType:
                return deepcopy( img )
        
        # Get image location
        imgLoc = self.findImgLoc( imgName = imgName, imgType = imgType )
        if self.printAll: 
            print("IM: Loading: %s:" % imgName)
            gm.tabprint( 'imgType: %s' % imgType )
            gm.tabprint( 'imgLoc: %s' % imgLoc )
            
        if imgLoc == None: return None
        
        # Read image
        img = gm.readImg(imgLoc, toType=toType)
        
        # Store image if called upon later
        if imgType == 'model':  self.img[imgName] = img            
        elif imgType == 'init': self.init[imgName] = img

        # Return image
        return deepcopy( img )
    
    # End getting model, unperturbed image


    def findImgLoc( self, imgName, imgType = 'model', newImg=False ):

        # Assume model image
        if imgType == 'model':
            imgLoc = self.imgDir + imgName + '_model.png'

        elif imgType == 'init':
            imgLoc = self.miscDir + imgName + '_init.png'
            
        elif imgType == 'wndchrm':
            imgLoc = self.wndDir + imgName + '.tiff'
            
        else:
            imgLoc = self.miscDir + imgName + '.png'

        # If new image, return the location it will become
        if newImg: 
            return imgLoc
        
        # Return if path exists
        if path.exists( imgLoc ):
            return imgLoc
        
        else:
            return None

    # End findImgFile

    def getAllImgLocs( self, miscImg=False ):

        imgDir = self.imgDir
        if miscImg:
            imgDir = self.miscDir

        imgList = listdir(imgDir)
        retList = []
        for imgName in imgList:
            retList.append(imgDir+imgName)

        return retList

    def printInfo( self,):
        from pprint import PrettyPrinter
        pp = PrettyPrinter( indent = 2 )

        print( "IM: run_info_class.printInfo():")

        pprint( self.rDict )

    def getScore( self, sName,  ):

        # Make sure rDict exists
        if self.rDict == None:
            return None
        
        # Create machine scores if needed
        elif self.get('machine_scores',None) == None:
            self.rDict['machine_scores'] = {}
        
        return self.rDict['machine_scores'].get( sName, None )

    def getAllScores( self  ):
        return self.get('machine_scores')

    def printScores( self, allScores=True):

        print("IM: run_info_class.printScores()")
        tabprint( 'run_id: %s' % self.get('run_id') )
        tabprint( 'zoo_merger: %s' % str( self.get('zoo_merger_score')) )
        tabprint( 'machine_scores: %d' % len(self.rDict['machine_scores']) )

        if allScores:
            for sKey in self.rDict['machine_scores']:
                print( '\t - %s: %f' %  (sKey, self.getScore( sKey )) )

    def addScore( self, name=None, score=None ):

        if name == None or score == None:
            if self.printAll:
                print("WARNING: IM: run.addScore. name or score not given")
                print('\t - name: ', name)
                print('\t - score: ', score)
                gm.eprint("WARNING: IM: run.addScore. name or score not given")
                gm.eprint('\t - name: ', name)
                gm.eprint('\t - score: ', score)
            return None

        self.rDict['machine_scores'][name] = score

        return self.rDict['machine_scores'][name]


    def get( self, inVal, defaultVal = None ):

        cVal = getattr( self, inVal, defaultVal )

        if cVal != defaultVal:
            return cVal

        dVal = self.rDict.get( inVal, defaultVal )
        if dVal != defaultVal:
            return dVal

        return defaultVal


    def txt2Json( self, ):
        
        print("\n" + "#"*120 + '\n')
        print("THIS FUNCTION SHOULD BE FOREVER OBSOLETE!")
        print("\n" + "#"*120 + '\n')

        if self.printAll: print("IM: Run.txt2Json")

        self.rDict = self.createBlank()

        oldLoc = self.runDir + 'info.txt'

        if self.printAll:
            print( '\t- oldLoc: (%s) %s' % ( path.exists( oldLoc ), oldLoc ) )

        # Check if info location have a file
        if not path.exists( oldLoc ):
            print("IM: Run.txt2Json: ERROR:") 
            print("\t - info.txt file not found.")
            print("\t - oldLoc: %s" % oldLoc)
            return False

        infoData = gm.readFile( oldLoc )

        # Check if retrieved info
        if infoData == None:
            print("Error: IM: Info.txt file not found to create new info.json file")
            print("\t - oldLoc: %s" % oldLoc)
            return None

        # Go through info file and add appropriate information to json

        if self.printAll: print("\t - Reading info.txt")

        tid = None
        rNum = None
        gNum = None
        mData = None
        hScore = None
        wins = None

        for i,l in enumerate( infoData ):
            line = l.strip().split()
            if self.printAll: print('\t-',i,line)
                
            if len(l) < 2: continue

            if line[0] == 'sdss_name':
                tid = line[1]

            if line[0] == 'generation':
                gNum = line[1]

            if line[0] == 'run_number':
                rNum = line[1]

            if line[0] == 'model_data':
                mData = line[1]

            if line[0] == 'wins/total':
                wins = line[1]

            if line[0] == 'human_score':
                hScore = line[1]


        # check if all infor information found
        if ( mData == None or wins == None or hScore == None ):
            if self.printBase: print("Error: IM: Needed information not found in info.txt")
            if self.printBase : print("\t - infoLoc: %s" % oldLoc)
            return None

        # readjust run_id

        # Assume name of directory I'm in is the run_id
        self.rDict['run_id'] = 'r' + str( rNum )
        self.rDict['model_data'] = str(mData)
        self.rDict['zoo_merger_score'] = float(hScore)

        # created initial info.json file from info.txt
        self.baseDict = deepcopy( self.rDict )

        # Save results
        self.saveInfoFile( saveBase = True )

    # End txt2Json


    # end info2Dict

    def saveInfoFile( self, saveBase=False, ):

        if self.printAll: print("IM: Run.saveInfoFile: Saving info data file...")

            
        with open( self.infoLoc, 'w' ) as infoFile:
            json.dump( self.rDict, infoFile, indent=4 )
            retVal = True

            
        if saveBase:
            with open( self.baseLoc, 'w' ) as bFile:
                json.dump( self.baseDict, bFile, indent=4 )

        self.initDict = deepcopy( self.rDict )
    # End save info file

# End Run info class


class target_info_class:

    tDict = None	    # Primary dictionary of info for all data 
    baseDict = None	    # Simple dictionary of info for basic info.
    progDict = None	    # For saving progress of current target
    status = False	    # State of this class.  For initiating.

    sFrame = None	   # Score dataframe
    rInfo = None	   # run_info_class for accessing run directories

    printBase = True   # If printing basic info during code execution
    printAll = False   # Printing of all info during code execution

    targetDir = None     # The directory on disk for storing everyting relating to target
    zooMergerDir = None  # Directory for original Galaxy Zoo: Merger Models for target
    plotDir = None       # Directory for any plots

    baseInfoLoc = None   # Location of basic info file
    allInfoLoc = None    # Location of all file containing nearly all target and model information
    scoreLoc = None      # Location for all scores 
    baseScoreLoc = None  # Location of basic score file.


    # For user to see headers.
    targetHeaders = ( 'target_id', 'target_images', 'simulation_parameters', 'image_parameters', 'zoo_merger_models', 'score_parameters', 'progress', 'model_sets', 'ga_models' )
    
    baseHeaders = ( 'target_id', )
     
    wndchrmInfoHeaders = [ 'run_id', 'target_id', 'image_name', 'zoo_merger_score' ]


    def __init__( self, targetDir = None, tArg = None, \
            printBase = None, printAll = None, ):

        # Creat arg class if none passed
        if tArg == None:
            tArg = gm.inArgClass()
        
        # Tell class what to print
        if printBase != None:
            self.printBase = printBase
        else:
            self.printBase = tArg.printBase
            
        if printAll != None:   
            self.printAll = printAll
        else:            
            self.printAll = tArg.printAll   
        
        if self.printAll == True: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', targetDir)
            print('\t - arg.targetDir: ', tArg.targetDir)
            
        if tArg.targetDir == None: tArg.targetDir = targetDir
                     
        # To avoid future confusion
        if self.printAll:  self.printBase = True

        # Check if directory has correct structure
        varGood = self.initVarNames( tArg )
        
        # Exit if bad
        if not varGood: 
            gm.eprint("WARNING: IM.target_info_class.__init__(): ")
            gm.etabprint("Something went wrong initializing variables.")
            return
        
        # Check if new target
        if tArg.get('newTarget', None) != None:
            setupGood = self.newTargetSetup( tArg )
            if not setupGood: 
                gm.eprint("WARNING: IM.target_info_class.__init__(): ")
                gm.etabprint("Something went wrong initializing new target.")
                return
        
        dirGood = self.initTargetDir( tArg )

        # Complain if not
        if not dirGood:
            if self.printBase:
                gm.eprint("WARNING: IM.target_info_class.__init__(): ")
                gm.etabprint("Something went wrong initializing directory.")
            return


        if self.printAll: 
            print("IM: Target: Opening target info json")

        # Read info file
        with open( self.allInfoLoc ) as iFile:
            self.tDict = json.load( iFile )
        
        # Open score file
        if path.exists( self.scoreLoc ):
            # Read all as string
            self.sFrame = pd.read_csv( self.scoreLoc )
        
        self.status = True

    # End target init 
    
    #####    DEPRECATED FUNCTION.  Is being overwritten later. 
    # Create a new directory of runs based on new SIMR Models
    def create_new_generation( self, simr_models ):

        from os import mkdir

        if self.printAll:
            print("IM.create_new_generation:")

        # Expecting python dictionary for arguments
        sampleDict = { 'type' : 'python_dictionary' }
        if type( simr_models ) != type( sampleDict ):
            print("WARNING: IM.create_new_generation:")
            gm.tabprint("Expecting 'simr_models' of type: %s" % type( sampleDict ) )
            gm.tabprint("Received type: %s" % type( simr_models ) )
            gm.eprint("WARNING: IM.create_new_generation:")
            gm.etabprint("Expecting 'simr_models' of type: %s" % type( sampleDict ) )
            gm.etabprint("Received type: %s" % type( simr_models ) )
            return

        if self.printAll:
            gm.tabprint("simr_model Keys: %s" % str( list( simr_models.keys())))

        # Extract 
        gName = simr_models.get('generation_name', None)
        mData = simr_models.get('model_data', None)

        if type( gName ) == type( None ) \
        or type( mData ) == type( None ):
            
            print("WARNING: IM.target_info_class.create_new_generation:")
            gm.tabprint("input variable 'simr_models' invalid")
            gm.tabprint("Expected Keys: %s" % str( ['evolution_method','generation_name','model_data']) )
            gm.tabprint("generation_name: %s" % type( simr_models.get('generation_name', None) ) )
            gm.tabprint("model_data: %s" % type( simr_models.get('model_data', None) ) )
            
            gm.eprint("WARNING: IM.target_info_class.create_new_generation:")
            gm.etabprint("input variable 'simr_models' invalid")
            gm.etabprint("Expected Keys: %s" % str( ['evolution_method','generation_name','model_data']) )
            gm.etabprint("generation_name: %s" % type( simr_models.get('generation_name', None) ) )
            gm.etabprint("model_data: %s" % type( simr_models.get('model_data', None) ) )
            return 

        # example numpy array. 
        npArr = np.zeros((2,2))

        # Expecting numpy array for parameter data 
        if type( mData ) != type( npArr ):
            
            print("WARNING: IM.create_new_generation:")
            gm.tabprint("Expecting parameter array of type: %s" % type(npArr) )
            gm.tabprint("Received a type: %s" % type(mData) )
            
            gm.eprint("WARNING: IM.create_new_generation:")
            gm.etabprint("Expecting parameter array of type: %s" % type(npArr) )
            gm.etabprint("Received a type: %s" % type(mData) )
            
            return

        # Expectin 2D array
        if mData.ndim != 2:
            if self.printBase: 
                
                print("WARNING: IM.create_new_generation:")
                gm.tabprint("Expecting 2-D parameter array" )
                gm.tabprint("Received shape: %s" % str(mData.shape) )
                
                gm.eprint("WARNING: IM.create_new_generation:")
                gm.etabprint("Expecting 2-D parameter array" )
                gm.etabprint("Received shape: %s" % str(mData.shape) )
                
                return

        # Expecting over 14 parameters for giving to SPAM
        if mData.shape[1] < 14:
            if self.printBase: 
                
                print("WARNING: IM.create_new_generation:")
                gm.tabprint("Expecting at least 14 parameters per model for SPAM:" )
                gm.tabprint("Received shape: %s" % str(mData.shape) )
                
                gm.eprint("WARNING: IM.create_new_generation:")
                gm.etabprint("Expecting at least 14 parameters per model for SPAM:" )
                gm.etabprint("Received shape: %s" % str(mData.shape) )
                
                return

        # Now Then, Create 
        if not path.exists( self.simrDir ): mkdir( self.simrDir )

        genLoc = self.simrDir + gName + '/'

        if path.exists( genLoc ):

            if self.printBase:
                gm.tabprint("Model Generation already exists: %s" % genLoc )

            if simr_models.get( 'overwrite', False ):

                if self.printAll: 
                    gm.tabprint("Removing Previous Folder: %s" % genLoc )

                from shutil import rmtree
                rmtree( genLoc )

            else:
                if self.printBase: 
                    
                    print("WARNING: IM.create_new_generation:")
                    gm.tabprint("Please add overwrite command if you wish to overwrite")
                    gm.tabprint("Returning...")
                    
                    gm.eprint("WARNING: IM.create_new_generation:")
                    gm.etabprint("Please add overwrite command if you wish to overwrite")
                    gm.etabprint("Returning...")
                    
                return


        if self.printAll:
            gm.tabprint("Creating Model Generation: %s" % genLoc )
            gm.tabprint("Model Count: %d" % mData.shape[0])

        mkdir( genLoc )
        # Generation Folder created


        # Create Run folders

        nRuns = mData.shape[0]
        for i in range( nRuns ):

            # Create model name
            model_id = 'run_%s' % str(i).zfill(4)
            model_dir = genLoc + model_id + '/'
            info_file = model_dir + 'base_info.json'

            # Create string out of parameters
            model_data = mData[i,:]
            model_string = ','.join( map(str, model_data) )

            # Create the information file for the run/model
            mInfo = {}
            mInfo['run_id'] = model_id
            mInfo['generation_id'] = gName
            mInfo['model_data'] = model_string

            # Print first if needed
            if i < 1:
                if self.printAll:
                    gm.tabprint('model_dir: %s' % model_dir)
                    gm.tabprint('info_file: %s' % info_file)
                    pprint(mInfo)

            # Create model directories and files
            mkdir( model_dir )
            gm.saveJson( mInfo, info_file )

        if self.printAll:
            gm.tabprint("Created directories: %d" % len( listdir( genLoc ) ) )
            gm.tabprint("Expected directories: %d" % nRuns )

    # End create_new_generation

    def getTargetImage( self, tName = None, overwrite=False, printAll = False ):
        
        if printAll: print("IM: target_info_class: getTargetImage:")
        
        # Create place to store images if needed.
        if self.get('targetImgs',None) == None:
            if printAll: gm.tabprint("Creating dict to store loaded target images")
            self.targetImgs = {}
        
        # Return target image if already loaded.
        tImg = self.targetImgs.get(tName,None)
        if type(tImg) != type(None) and not overwrite:
            if printAll:  gm.tabprint("Returning preloaded target image.")
            return deepcopy( tImg )
        
        # Else find and open target image
        if printAll:  gm.tabprint("Searching for target image: %s" % tName)
        tLoc = self.findTargetImage(tName)
        if printAll:  gm.tabprint("Found: %s" % tLoc)
            
        if not gm.validPath(tLoc,):
            return None
        
        else:
            self.targetImgs[tName] = gm.readImg(tLoc)
            return deepcopy( self.targetImgs[tName] )
        
    # End getTargetImage()


    def findTargetImage( self, tName = None, newImg = False ):

        # Expected location
        tLoc = self.imgDir + 'target_%s.png' % tName
        
        # If needing path for new image, return
        if newImg:
            return tLoc
        
        # Only return if file exists
        if path.exists( tLoc ):
            return tLoc
        else:
            return None
    # End find target image by name

    def printParams( self, ):

        for pKey in self.tDict['score_parameters']:
            gm.pprint( self.tDict['score_parameters'][pKey] )
            

    def addScoreParameters( self, params, overWrite = False ):
    
        for pKey in params:
            
            if self.tDict['score_parameters'].get(pKey) == None or overWrite:
                self.tDict['score_parameters'][pKey] = params[pKey]


    def getScores( self, scrName = None, reload=False): 

        # Reading from file
        if type(self.sFrame) == type(None):
            if gm.validPath( self.scoreLoc, printWarning=False ):
                self.sFrame = pd.read_csv( self.scoreLoc )
                
            else:
                if self.printAll: 
                    print( "WARNING: IM.target_info_class.getScores:" )
                    gm.eprint( "WARNING: IM.target_info_class.getScores:" )
                self.createBaseScore()
                
        if reload:
            self.sFrame = pd.read_csv( self.scoreLoc )
            
        return self.sFrame

    def createBaseScore( self, ):

        zDict = self.tDict['zoo_merger_models']
        nRuns = len(zDict)

        scoreHeaders = [ 'run_id', 'zoo_merger_score' ]

        self.sFrame = pd.DataFrame( \
                index = np.arange( nRuns ), \
                columns = scoreHeaders 
            )

        for i, rKey in enumerate( zDict ):
            rDict = zDict[rKey]
            self.sFrame.at[i,'run_id'] = rDict['run_id'] 
            self.sFrame.at[i,'zoo_merger_score'] = rDict['zoo_merger_score']

        self.sFrame.to_csv( self.baseScoreLoc, index = False, quoting=2 )
        self.sFrame.to_csv( self.scoreLoc, index = False, quoting=2 )

    # End create Base Score file

    def updateScores( self, ):

        # Construct bare sFrame
        if type(self.sFrame) == type(None):
            print("Oh NOOO")
            return

        zDict = self.tDict['zoo_merger_models']
        nRuns = len(zDict)

        # Loop through run directories and increment count
        for i, row in self.sFrame.iterrows():

            rKey = row['run_id']
            rDict = zDict.get(rKey,None)
            
            # Occasional bug when I force end a program
            if rDict == None:
                rInfo = self.getRunInfo( rID = rKey )
                if rInfo == None:
                    continue
                else:
                    rDict = rInfo.rDict
            rScores = rDict['machine_scores']

            for sKey in rScores:
                self.sFrame.at[i,sKey] = rScores[sKey]


        # Update progress in tInfo
        if self.tDict['progress'].get('machine_scores') == None:
            self.tDict['progress']['machine_scores'] = {}

        scoreHeaders = list( self.sFrame.columns )
        for sName in scoreHeaders:
            if 'zoo_merger_score' in sName:
                continue

            sCount = self.sFrame[sName].count()
            self.tDict['progress']['machine_scores'][sName] = int( sCount )

        #print(self.tDict['progress'])

        self.saveInfoFile()
        #self.sFrame.to_csv( self.baseScoreLoc, index = False, quoting=2 )
        self.sFrame.to_csv( self.scoreLoc, index = False, quoting=2 )

    # End gathering scores and progress

    def get( self, inVal, defaultVal = None ):

        cVal = getattr( self, inVal, defaultVal )

        if cVal != defaultVal:
            return cVal
        
        if self.tDict != None:
            dVal = self.tDict.get( inVal, defaultVal )
            if dVal != defaultVal:
                return dVal

        return defaultVal


    # Gather run infos from directories
    def gatherRunInfoFiles( self, rArg=gm.inArgClass(), ):

        if self.printAll: 
            print( "IM: Target.gatherRunInfoFiles" )
            #rArg.printArg()

        runDirList = self.iter_runs()
        nRuns = len(runDirList)
        
        if rArg.get('newInfo',False):
            if self.printBase: print("IM: Target: gatherRunInfoFiles: Adjusting run infos")
            
            for rDir in runDirList:
                rInfo = run_info_class( runDir = rDir, rArg = rArg, printBase = self.printAll   )
                if self.printAll and rInfo.status == False: gm.tabprint( '%s - %s' % (rInfo.status, rDir ) )

        # Prepare model Set
        for h in self.targetHeaders:
            if self.tDict.get(h,None) == None:
                self.tDict[h] = {}

        # Go through directories and read run info files
        for i,rDir in enumerate(runDirList):
            if self.printAll: 
                gm.tabprint("IM: gather_run_info LOOP: %4d / %4d" % (i, nRuns), end='\r')
            rDict = self.getRunDict( rDir )
            if rDict != None and rDict.get('run_id',None) != None:
                self.tDict['zoo_merger_models'][rDict['run_id']] = rDict

        if self.printAll: gm.tabprint("IM: gather_run_info LOOP: %4d / %4d COMPLETE" % (nRuns, nRuns))
            
        self.saveInfoFile()

    # end gather Run Infos

    def getRunDict( self, rDir ):

        if rDir[-1] != '/': rDir += '/'
        dictLoc = rDir + 'info.json'
        rDict = gm.readJson( dictLoc )
        return rDict
    
    # End get Run Dir Info


    def getRunDir( self, rID=None,  ):
           
        runDir = self.zooMergerDir + '%s/' % rID        
        return gm.validPath(runDir)

    def getRunInfo( self, rID=None, rArg=None ):
        
        # If no argument given, just grab the first
        if rID == None:
            rID = list( self.tDict['zoo_merger_models'])[0]
        
        runDir = self.getRunDir(rID=rID)
        


        if rArg == None:
            rInfo = run_info_class( runDir = runDir, )
            
        else: 
            rInfo = run_info_class( runDir = runDir, rArg=rArg )
        
        rInfo.tInfo = self

        if rInfo.status:
            return rInfo

        else:
            return None
    
    def addRunDict( self, rInfo ):
        rID = rInfo.get('run_id',None)
        if rID != None:
            self.tDict['zoo_merger_models'][rID] = rInfo.rDict
    
    def iter_run_dicts( self,  startRun=0, endRun = -1, stepRun = 1 ):
        
        # If wanting all
        if startRun == 0 and endRun == -1 and stepRun == 1:
            return self.tDict['zoo_merger_models']
        
        if self.printAll:
            print("IM.target_info.class.iter_run_dicts: ")
            tabprint("startRun: %s" % startRun)
            tabprint("endRun: %s" % endRun)
            tabprint("stepRun: %s" % stepRun)
            
        keys = list(self.tDict['zoo_merger_models'].keys())
            
        # Check for invalid inputs
        
        if int(endRun) > len(keys):
            if self.printBase: 
                
                print("WARNING: IM.target_info.class.iter_run_dicts: ")
                tabprint("-endRun greater than number of runs")
                tabprint("-endRun: ", endRun)
                tabprint('-run count: ', len(keys))
                
                gm.eprint("WARNING: IM.target_info.class.iter_run_dicts: ")
                gm.etabprint("-endRun greater than number of runs")
                gm.etabprint("-endRun: ", endRun)
                gm.etabprint('-run count: ', len(keys))
                
            return None
        
        # Check if endRun not given
        if endRun == -1:
            endRun = len(keys)
        
        # Create index list of runs to extract based on inputs        
        iList = np.arange( int(startRun), int(endRun), int(stepRun) )
        
        # Extract desired run dicts
        runDictList = {}
        for i in iList:
            runDictList[keys[i]] = self.tDict['zoo_merger_models'][keys[i]]
           
        if self.printAll:
            tabprint("Extracting runs: ")
            tabprint( list(runDictList.keys()) )
        
        return runDictList
    

    def getOrbParam( self, generation_id = 'galaxy_zoo_merger'):

        if generation_id == 'galaxy_zoo_merger':
            rDicts = self.iter_run_dicts()
            nModels = len(rDicts)

            zScores = np.zeros(nModels)
            mData = np.zeros((nModels, 34))

            for i,rd in enumerate(rDicts):

                rDict = rDicts[rd]
                zScores[i] = rDict.get('zoo_merger_score')

                m1 = rDict['model_data']
                m2 = m1.split(",")
                mData[i,:] = np.array(m2)

            return (zScores, mData)

        else:
            print("IM.target_info_class: Generations other than 'galaxy_zoo_merger' not working at this time")
            return (None, None)
    # End getOrbParam


    def saveInfoFile( self ):

        if self.printAll: 
            print("IM: Target.saveInfoFile():")
            print("\t - Saving target info file...")

        if self.allInfoLoc == None:
            print("ERROR: IM: No target info location given...")
            return None

        with open( self.allInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
            self.initDict = deepcopy( self.tDict )

        if type(self.sFrame) != type(None):
            self.sFrame.to_csv( self.scoreLoc, index = False, quoting=2 )

   # End Target info class


    # Read in classes for run infos
    def iter_runs( self,  start=0, n = -1, ):

        # Get list of run directories
        self.runDirs = [ self.zooMergerDir + item + '/' for item in listdir( self.zooMergerDir ) if 'run' in item ]
        self.runDirs.sort()

        if start == 0:
            self.runDirs = self.runDirs[:n]
        elif n == -1:		  
            self.runDirs = self.runDirs[start:n]
        else:
            self.runDirs = self.runDirs[start:start+n]

        return self.runDirs
    
    
    # Initialize variables
    def initVarNames( self, tArg ):
    
        
        if self.printAll:
            print( 'IM: target_info_class.initVarNames:' )
            print( '\t - targetDir: %s' % tArg.targetDir )

        self.targetDir = gm.validPath( tArg.targetDir )

        # Check if creating a new directory
        if tArg.get('newTarget', None) != None and type(self.targetDir) == type(None):
            
            if tArg.printAll:
                gm.tabprint("Creating new target directory")
            
            # Check if given no target location given
            if tArg.get('targetDir',None) == None:
                
                # check for valid dataDir given
                dataDir = gm.validPath( tArg.get('dataDir', None) )
                if dataDir == None:
                    gm.eprint("WARNING: IM: target_info_class.initVarNames:")
                    gm.etabprint("Bad dataDir: %s" % tArg.get("dataDir") )
                    return False

                elif tArg.printAll:
                    gm.tabprint( "Saving in dataDir: %s" % dataDir )
                
                self.targetDir = dataDir + tArg.get('newTarget')
                
            else:
                self.targetDir = tArg.get("targetDir")
                
            # Create new Target directory if not already created. 
            from os import mkdir
            if self.targetDir[-1] != '/': 
                self.targetDir += '/'
            if self.printBase: 
                gm.tabprint( "Creating new Target: %s" % self.targetDir)
            mkdir( self.targetDir )
            
            # Check targetDir was created.
            tCheck = gm.validPath( self.targetDir )
            if tCheck == None:
                gm.etabprint( "Could not create directory: %s" % self.targetDir )
                return False
            else: self.targetDir = tCheck
            
        # See if valid target path
        if self.targetDir == None:
            gm.eprint("WARNING: IM: target_info_class.initVarNames:")
            gm.etabprint("Bad targetDir: %s" % tArg.get('targetDir') )
            return False

        # Define paths for all useful things in target structure
        
        # Main paths in target dir
        self.infoDir = self.targetDir + 'information/'
        self.gen0 = self.targetDir + 'gen000/'
        self.zooMergerDir = self.targetDir + 'zoo_merger_models/'
        self.tmpDir = self.targetDir + 'tmp/'
        self.plotDir = self.targetDir + 'plots/'
        
        # Directires inside info dir
        self.imgDir = self.infoDir + 'target_images/'
        self.maskDir = self.infoDir + 'target_masks/'        
        self.scoreParamDir = self.infoDir + 'score_parameters/'
        self.wndDir = self.infoDir + 'wndchrm_files/'

        # Locations of files inside info dir
        self.allInfoLoc = self.infoDir + 'target_info.json'
        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.baseScoreLoc = self.infoDir + 'base_scores.csv'
        self.imgParamLocOld = self.infoDir + 'param_target_images.json'
        
        # Various files within the wndchrm directory
        self.wndRunRawDFLoc = self.wndDir + 'all_runs_raw.pkl'
        self.wndTargetRawFitLoc = self.wndDir + 'targets_raw.fit'
        self.wndTargetRawCSVLoc = self.wndDir + 'targets_raw.csv'
        self.wndTargetRawDFLoc = self.wndDir + 'targets_raw.pkl'
        
        # Files inside misc directories
        self.imgParamLoc = self.imgDir + 'param_target_images.json'
        
        if tArg.printAll: print("initVarNames: Good")

        return True
    
    # End init variables

    # initialize target directories
    def initTargetDir( self, tArg ):
        
        from shutil import move
        from os import mkdir
        
        if self.printAll:
            print( 'IM: target_info_class.initTargetDir:' )
            print( '\t - targetDir: %s' % tArg.targetDir )
        


        # Check if directories are found        
        if ( not path.exists( self.infoDir ) \
        or not path.exists( self.zooMergerDir ) \
        or not path.exists( self.plotDir ) ):
            
            if self.printBase:
                gm.eprint("IM: Target_init: Some directories not found")
                gm.etabprint('Info: %s' % gm.validPath(self.infoDir))
                gm.etabprint('Plot: %s' % gm.validPath(self.plotDir))
                gm.etabprint('Zoo Merger: %s' % gm.validPath(self.zooMergerDir))
                gm.etabprint("Consider using -newInfo command")
                
            return False
            
        
        # Check if info directory has needed objects
        if not path.exists( self.allInfoLoc   ) \
        or not path.exists( self.baseInfoLoc  ) \
        or not path.exists( self.scoreLoc     ) \
        or not path.exists( self.baseScoreLoc ):
            if self.printBase:
                gm.eprint("IM: Target_init: Needed information files not found.")
                gm.etabprint('Main Info: %s' % gm.validPath(self.allInfoLoc))
                gm.etabprint('Base Info: %s' % gm.validPath(self.baseInfoLoc))
                gm.etabprint('Main Score: %s' % gm.validPath(self.scoreLoc))
                gm.etabprint('Base Score: %s' % gm.validPath(self.baseScoreLoc))
                gm.etabprint("Consider using -newInfo command")
            return False

        if self.printAll:
            print( '\t - targetDir: (%s) %s' % ( path.exists( self.targetDir ), self.targetDir ) )
            print( '\t - infoDir: (%s) %s' % ( path.exists( self.infoDir ), self.infoDir ) )
            print( '\t - baseInfoLoc: (%s) %s' % ( path.exists( self.baseInfoLoc ), self.baseInfoLoc ) )
            print( '\t - allInfoLoc: (%s) %s' % ( path.exists( self.allInfoLoc ), self.allInfoLoc ) )
            print( '\t - zooMergerDir: (%s) %s' % ( path.exists( self.zooMergerDir ), self.zooMergerDir ) )
            print( '\t - plotDir: (%s) %s' % ( path.exists( self.plotDir ), self.plotDir ) )

        return True

    
    # if calling new Info
    def newTargetSetup( self, tArg ):
        
        from os import mkdir, remove
        from shutil import copyfile

        if self.printBase: 
            print("IM: new_target_setup: Creating basic files and directories")
            
        # remove old files
        if path.exists( self.allInfoLoc ):  remove( self.allInfoLoc )
        if path.exists( self.scoreLoc ):    remove( self.scoreLoc )
            
        # Create directories
        if not path.exists( self.infoDir ): mkdir( self.infoDir )
        if not path.exists( self.plotDir ): mkdir( self.plotDir )
        if not path.exists( self.imgDir ):  mkdir( self.imgDir )
        if not path.exists( self.maskDir ): mkdir( self.maskDir )
        if not path.exists( self.tmpDir ):  mkdir( self.tmpDir )
        if not path.exists( self.wndDir ):  mkdir( self.wndDir )
        if not path.exists( self.scoreParamDir ):  mkdir( self.scoreParamDir )
        if not path.exists( self.zooMergerDir ):   mkdir( self.zooMergerDir )
        
        # If newTarget
        
                # Files inside misc directories
        self.imgParamLoc = self.imgDir + 'param_target_images.json'
        
        # Move old directories to new location if needed.
        if gm.validPath( self.imgParamLocOld ) != None:
            print("Hi")
            print("OLD: %s"%gm.validPath(self.imgParamLocOld))
            move( self.imgParamLocOld, self.imgParamLoc)
            print("NEW: %s"%gm.validPath(self.imgParamLoc))
        
        # If using old folder layout, rename        
        if gm.validPath( self.gen0 ):
            from os import rename
            print("Found old path   : %s" % self.gen0)
            print("Propose new path : %s" % self.zooMergerDir)
            rename(self.gen0,self.zooMergerDir)
            print("New Path? : %s" % gm.validPath(self.zooMergerDir) )
        
        
        # Check for base file
        newBase = tArg.get('newBase',False)
        
        if newBase:
            createGood = self.createBaseInfo( tArg )
            if not createGood: return False
            
        elif not path.exists( self.baseInfoLoc ) and not newBase:
            if self.printBase: 
                
                print("IM: WARNING: newTargetSetup:")
                tabprint("No base info file!")
                tabprint("Consider '-newBase' command")
                
                gm.eprint("IM: WARNING: newTargetSetup:")
                gm.etabprint("No base info file!")
                gm.etabprint("Consider '-newBase' command")
                
            return False
        
        
        # Copy files if they exist
        if path.exists( self.baseInfoLoc ): copyfile( self.baseInfoLoc, self.allInfoLoc )
        if path.exists( self.baseScoreLoc ): copyfile( self.baseScoreLoc, self.scoreLoc )

        # Read info file
        with open( self.allInfoLoc ) as iFile:
            self.tDict = json.load( iFile )
        
        # Open score file
        if path.exists( self.scoreLoc ):
            # Read all as string
            self.sFrame = pd.read_csv( self.scoreLoc )

        # Should run infos be modified? 
        rArg = gm.inArgClass()
        rArg.printBase = False
        if tArg.get('newRunInfo',True):
            rArg.newInfo = True
        if tArg.get('newRunBase',True):
            rArg.newBase = True

        self.gatherRunInfoFiles( rArg=rArg )
        
        # Collect info 
        if newBase: self.createBaseScore(  )
        self.updateScores()
        self.saveInfoFile( )
        
        return True

    # End new target info dictionary
    
    
    def createBaseInfo( self, tArg ):
        
        from os import getcwd, listdir, remove, mkdir
        from shutil import copyfile, move
        from copy import deepcopy
        
        # For basic scores later on
        
        # Get parent directory this code suite is located in.
        simrDir = __file__.split('Support_Code/info_module.py')[0]
         
        # Assume directory name of target is target name
        tName = self.targetDir.split('/')[-2]
        printAll = self.printAll
        tInfo = self
        
        if self.printAll: 
            print('IM: createBaseInfo')
            gm.tabprint('Target ID: %s'%tName)  
            gm.tabprint('Target Dir: %s'%self.get('target_dir'))
                
        # Create blank base dict
        self.tDict = {}
        for key in self.baseHeaders:
            self.tDict[key] = {}        
        self.tDict['target_id'] = tName  
        
        # Save target_id
        with open( self.baseInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
        
        # Get starting target files from input data folder
        inputDir = gm.validPath( simrDir  + 'Input_Data/targets/' + tName + '/')        
        if self.printAll: gm.tabprint('Input Dir: %s'%inputDir)
            
        # Check if valid directory, exit if not
        if inputDir == None:
            if self.printBase: 
                
                print("WARNING: IM: Input directory not found: %s"%tName)
                gm.tabprint('Input Dir: %s'%inputDir)
                
                gm.eprint("WARNING: IM: Input directory not found: %s"%tName)
                gm.etabprint('Input Dir: %s'%inputDir)
                
            return False
        
        # Get folder contents
        inputFiles = listdir(inputDir)            

        if self.printAll: 
            gm.tabprint("Input Dir contents:")
            for f in inputFiles:
                gm.tabprint('    - %s'%f)
        
        # Search for target image
        fromLoc = None        
        for fName in inputFiles:
            if '.png' in fName:
                fromLoc = inputDir + fName      
        
        # If found, copy target image from input folder to targetInfo folder
        if gm.validPath( fromLoc ) != None:
            toLoc = self.findTargetImage( 'zoo_0', newImg = True )
            copyfile( fromLoc, toLoc )
            
        # Grab specific target image data
        metaLocRaw1 = inputDir + 'sdss%s.meta'%tName
        metaLocRaw2 = inputDir + '%s.meta'%tName
        metaLoc1 = gm.validPath( metaLocRaw1 )
        metaLoc2 = gm.validPath( metaLocRaw2 )
        
        if self.printAll: 
            gm.tabprint('Meta Data Loc1: %s'% metaLocRaw1 )
            gm.tabprint('Meta Data Loc2: %s'% metaLocRaw2 )
            
        # Open file for target galaxy zoo merger image information
        if metaLoc1 != None:
            mFile = open(metaLoc1,'r')
            copyfile( metaLoc1, self.infoDir + 'target.meta' )
        elif metaLoc2 != None:
            mFile = open(metaLoc2,'r')
            copyfile( metaLoc2, self.infoDir + 'target.meta' )
        else:
            if self.printBase: 
                print("WARNING: IM: Meta data not found:")
                gm.eprint("WARNING: IM: Meta data not found:")
            return False

        # Copy starting target zoo image param
        pLoc = simrDir + 'param/zoo_blank.json'
        blank_param = gm.readJson(pLoc)
        if blank_param == None:        
            if self.printBase: 
                print("WARNING: IM: Start zoo image param not found: %s"%pLoc)
                gm.eprint("WARNING: IM: Start zoo image param not found: %s"%pLoc)
            return False

        # Copy blank parameter
        base_zoo = {}
        zoo_name = 'zoo_0'
        base_zoo[zoo_name] = deepcopy(blank_param['zoo_blank'])

        # Make name and comments initial comments.
        base_zoo[zoo_name]['name'] = zoo_name
        base_zoo[zoo_name]['comment'] = 'Starting score parameters file for %s'%tName
        base_zoo[zoo_name]['imgArg']['comment'] = "Starting image parameters for %s"%tName

        # Grab information
        for l in mFile:
            l = l.strip()
            
            if 'height' in l:
                h = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['image_size']['width'] = int(h)
                
            if 'width' in l:
                w = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['image_size']['width'] = int(w)
            if 'px' in l:
                px = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['px'] = int(px)
            if 'py' in l:
                py = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['py'] = int(py)
            if 'sx' in l:
                sx = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['sx'] = int(sx)
            if 'sy' in l:
                sy = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['sy'] = int(sy)

        if self.printAll: gm.pprint(base_zoo)

        # Save new target image parameter
        newParamLoc = self.imgParamLoc
        gm.saveJson( base_zoo, newParamLoc, pretty=True )
                
        # Create basic scoring parameters
        self.createDirectScoreParameters( base_zoo['zoo_0'] )
        
        ############# WNDCHRM Score Parameter File #################
        # Create a starting score parameter file for WNDCHRM image creation. 
        chime_name = 'chime_0'
        base_chime = {}
        base_chime[chime_name] = deepcopy( base_zoo[zoo_name] )

        # ALWAYS modify the names
        base_chime[chime_name]['name'] = chime_name
        base_chime[chime_name]['comment'] = 'Developing initial WNDCHRM implementation'

        # Resize WNDCHRM image to 100 pixels
        old_size = base_zoo[zoo_name]['imgArg']['image_size']
        chime_size = 100

        max_side = np.amax( [ int(old_size['width']), int(old_size['height']) ] )
        redox_ratio = float( chime_size / max_side )

        if printAll: 
            print("IM: target_info_class.createBaseInfo: Creating WNDCHRM score parameter")
            gm.tabprint("Old Size: %d" % max_side)
            gm.tabprint("New Size: %d" % chime_size)
            gm.tabprint("Reduction Ratio: %d" % redox_ratio)

        # Create new image parameters
        base_chime[chime_name]['imgArg']['name'] = chime_name
        base_chime[chime_name]['imgArg']['type'] = 'wndchrm'
        base_chime[chime_name]['imgArg']['comment'] = 'Starting image for WNDCHRM feature extraction'

        # Adjust Image Size
        base_chime[chime_name]['imgArg']['image_size']['width' ] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['image_size']['width' ]) )
        base_chime[chime_name]['imgArg']['image_size']['height'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['image_size']['height']) )

        # Adjust galaxy centers
        base_chime[chime_name]['imgArg']['galaxy_centers']['px'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['px'] ) )
        base_chime[chime_name]['imgArg']['galaxy_centers']['py'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['py'] ) )
        base_chime[chime_name]['imgArg']['galaxy_centers']['sx'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['sx'] ) )
        base_chime[chime_name]['imgArg']['galaxy_centers']['sy'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['sy'] ) )

        # Add blurring effect
        base_chime[chime_name]['imgArg']['blur'] = {}
        base_chime[chime_name]['imgArg']['blur']['type'] = 'gaussian_blur'
        base_chime[chime_name]['imgArg']['blur']['size'] = 5
        base_chime[chime_name]['imgArg']['blur']['weight'] = .5

        # Add feature arguments
        base_chime[chime_name]['featArg'] = {}
        base_chime[chime_name]['featArg']['type'] = 'wndchrm_all'
        base_chime[chime_name]['featArg']['normalization'] = None

        # If you want to modify the final image brightness normalization
        base_chime[chime_name]['imgArg']['normalization'] = {}
        base_chime[chime_name]['imgArg']['normalization']['type'] = 'type1'
        base_chime[chime_name]['imgArg']['normalization']['norm_constant'] = 2.5

        # WORKING
        # Change to feature comparison
        base_chime[chime_name]['cmpArg']['targetName'] = chime_name
        base_chime[chime_name]['cmpArg']['type'] = 'direct_feature_comparison'

        if printAll: 
            gm.tabprint("Saving Base WNDCHRM parameter")
            gm.pprint(base_chime)


        # Save score param for image creation later
        tInfo.saveScoreParam( base_chime, chime_name )
        
        # Create starting WNDCHRM feature normalization file
        norm_chime_0 = {}
        norm_chime_0['name'] = 'norm_chime_0'
        norm_chime_0['image_group'] = 'chime_0'
        norm_chime_0['normalization_method'] = 'sklearn_StandardScaler'

        self.saveWndchrmNorm( norm_chime_0, norm_chime_0['name'] )

        # Find pair file 
        pairPath1 = gm.validPath( inputDir + 'sdss%s.pair' % tName )
        pairPath2 = gm.validPath( inputDir + '%s.pair' % tName )
        if self.printAll: 
            gm.tabprint( 'pairPath1: %s' % pairPath1 )
            gm.tabprint( 'pairPath2: %s' % pairPath2 )
            
        if pairPath1 != None:            
            pairFile = open(pairPath1, 'r' )
            copyfile( pairPath1, self.infoDir + 'target.pair' )
            
        elif pairPath2 != None:            
            pairFile = open(pairPath2, 'r' )
            copyfile( pairPath2, self.infoDir + 'target.pair' )
            
        else:
            if self.printBase: 
                print( "WARNING: IM: Pair data not found: %s" % pairPath )
                gm.eprint( "WARNING: IM: Pair data not found: %s" % pairPath )
            return False

        # Create a basic mask that covers the target galaxy based on pair file. 
        start_roi_mask = gm.readJson(simrDir+'param/mask_roi_blank.json')

        start_roi_mask['name'] = 'mask_roi_zoo_0'
        start_roi_mask['comment'] = 'Starting mask for %s' % tName
        start_roi_mask['target_name'] =  tName
        start_roi_mask['primary_start']['thickness'] =  10
        start_roi_mask['secondary_start']['thickness'] =  10

        for l in pairFile: 
            
            if ';' in l:                
                l = l[0:-2]

            if 'primaryA=' in l:
                l = l.strip()
                start_roi_mask['primary_start']['A'] =  round( float( l.split('=')[1] ) )

            elif 'primaryB=' in l:
                start_roi_mask['primary_start']['B'] =  round( float( l.split('=')[1] ) )

            elif 'primaryAngle=' in l:
                start_roi_mask['primary_start']['angle'] =   float( l.split('=')[1] ) 

            elif 'primaryX=' in l:
                start_roi_mask['primary_start']['center'][0] =  round( float( l.split('=')[1] ) )

            elif 'primaryY=' in l:
                start_roi_mask['primary_start']['center'][1] =  round( float( l.split('=')[1] ) )

            elif 'secondaryA=' in l:
                l = l.strip()
                start_roi_mask['secondary_start']['A'] =  round( float( l.split('=')[1] ) )

            elif 'secondaryB=' in l:
                start_roi_mask['secondary_start']['B'] =  round( float( l.split('=')[1] ) )

            elif 'secondaryAngle=' in l:
                start_roi_mask['secondary_start']['angle'] =   float( l.split('=')[1] ) 

            elif 'secondaryX=' in l:
                start_roi_mask['secondary_start']['center'][0] =  round( float( l.split('=')[1] ) )

            elif 'secondaryY=' in l:
                start_roi_mask['secondary_start']['center'][1] =  round( float( l.split('=')[1] ) )


        if self.printAll: 
            gm.pprint(start_roi_mask)

        self.saveMaskRoi( start_roi_mask, 'mask_roi_zoo_0')
        
                
        # Create run base info files.

        # If creating new run base infos
        if tArg.get("newRunBase",False):
            
            # Get directory with galaxy zoo merger files. 
            modelLoc = gm.validPath( simrDir  + 'Input_Data/zoo_models/' + tName + '.txt')
            if self.printAll: gm.tabprint("Model File: %s - %s" % ( gm.validPath(modelLoc), modelLoc ) )
            
            modelFile = gm.readFile( modelLoc )
            if modelFile == None:
                if self.printAll: 
                    print("WARNING: IM: target.createBaseInfo: Failed to open zoo model file.")
                    gm.eprint("WARNING: IM: target.createBaseInfo: Failed to open zoo model file.")
                return False
            
            # Create a blank run info dict for copying
            runHeaders = ( 'run_id', 'model_data', \
                    'zoo_merger_score', 'machine_scores', 'human_scores')
            
            blank_run_info = {}
            for rh in runHeaders:
                blank_run_info[rh] = {}
            
            for i,l in enumerate(modelFile):
                
                # Grab galaxy zoo merger model data from file
                score_data, model_data = l.strip().split()                
                scores = score_data.split(',')        
                if len( scores ) != 4:
                    if self.printAll: gm.tabprint("Found Models: %d"%i)
                    break
                
                zoo_merger_score = scores[1]
                wins = scores[2] 
                losses = scores[3]
                
                # Create / goto run directory
                run_id = 'run_%s' % str(i).zfill(4)
                runDir = self.zooMergerDir + run_id + '/'
                
                # Move old directory if found
                oldDir = self.zooMergerDir + 'run_%s' % str(i).zfill(5) + '/'
                if gm.validPath( oldDir ) != None:
                    move( oldDir, runDir )
                    
                if gm.validPath( runDir ) == None:
                    mkdir( runDir )
                
                # Copy and fill in run info
                rInfo = deepcopy( blank_run_info )
                
                rInfo['run_id'] = run_id
                rInfo['zoo_merger_score'] = zoo_merger_score
                rInfo['model_data'] = model_data
                rInfo['human_scores']['zoo_merger_wins_losses'] = '%s/%s' % ( wins, losses )
                rInfo['human_scores']['zoo_merger_score'] = zoo_merger_score
                
                # Save base run info
                rInfoLoc = runDir + 'base_info.json'
                gm.saveJson( rInfo, rInfoLoc, pretty = True )
                
                # Remove old info file if found
                oldInfoLoc = runDir + 'info.txt'
                if gm.validPath( oldInfoLoc ) != None:
                    remove( oldInfoLoc )
                               

        return True
        
    # end creating base info file
    
    def createDirectScoreParameters( self, startParam ):
        
        # Create basic scoring parameters with zoo_0
        import Machine_Score.direct_image_compare as dc
        score_functions = dc.get_score_functions()

        direct_params = {}
        imgName = startParam['imgArg']['name']
        
        modParam = deepcopy( startParam )
        modParam['cmpArg']['type'] = 'direct_image_comparison'
        modParam['comment'] = 'Direct Image Comparison functions for image %s' % imgName
        
        for score_name, ptr in score_functions:
            new_name = '%s_%s' % ( imgName, score_name )
            direct_params[ new_name ] = deepcopy( modParam )
            direct_params[ new_name ]['name'] = new_name            
            direct_params[ new_name ]['cmpArg']['direct_compare_function'] =  score_name
            
        if self.printAll: 
            print("IM: Target_Class: createDirectScoreParameters")
            for name in direct_params.keys():
                gm.tabprint(name)
        
        # Save created score files
        self.saveScoreParam( direct_params, '%s_direct_scores'%imgName)            
        
    

    
    def getImageParams( self, imgName=None):
        
        from copy import deepcopy
        
        # Read image parameters common for target
        img_params = gm.readJson( self.imgParamLoc )
        
        # Return all if none specified.
        if imgName == None:     
            return deepcopy( img_params )
        
        # Return single image parameter if specified.
        else:
            return deepcopy( img_params.get(imgName,None) )
        
    # End getting image parameters
    
    def addImageParams( self, in_params, overWrite = False ):
        
        # If file doesn't exist
        if gm.validPath( self.imgParamLoc ) == None:
            gm.saveJson( in_params, self.imgParamLoc, pretty=True )
            return
        
        # Else file does exsit
        old_params = gm.readJson( self.imgParamLoc )
        
        # Loop through new image parameters
        for sKey in in_params:
            
            # Add to old if overwrite or not found
            if overWrite or sKey not in old_params:
                old_params[sKey] = in_params[sKey]            
        
        gm.saveJson( old_params, self.imgParamLoc, pretty=True )
    
    def overWriteImageParams( self, in_params ):
        gm.saveJson( in_params, self.imgParamLoc )
    
    def saveMaskImage( self, maskImage, maskName ):
        maskLoc = self.maskDir + '%s.png'%maskName
        gm.saveImg( maskImage, maskLoc)
    
    def getMaskImage( self, maskName ):
        
        if self.get('targetMasks',None) == None:
            self.targetMasks = {}
            
        if type( self.targetMasks.get(maskName,None) ) != type( None ):
            return self.targetMasks[maskName]
        
        else:
            mask = self.readMaskImage( maskName )
            self.targetMasks[maskName] = mask
            return self.targetMasks[maskName]
        
    
    def readMaskImage( self, maskName ):
        maskLoc = self.maskDir + '%s.png'%maskName
        mask = gm.readImg(  maskLoc )
        return mask
    
    def saveScoreParam( self, score_params, param_file_name ):
        paramLoc = self.scoreParamDir + '%s.json'%param_file_name
        gm.saveJson( score_params, paramLoc, pretty=True )
    
    def getScoreParam( self, param_name ):
        if self.tDict.get('score_parameters', None ) == None: return None
        param = self.tDict['score_parameters'].get(param_name,None)
        return { param_name : param }
    
    def readScoreParam( self, param_file_name ):        
        paramLoc = self.scoreParamDir + '%s.json'%param_file_name
        score_params = gm.readJson(  paramLoc )
        return score_params
    
    def saveMaskRoi( self, mask_roi, file_name ):
        roiLoc = self.maskDir + '%s.json'%file_name
        gm.saveJson( mask_roi, roiLoc, pretty=True )
    
    def readMaskRoi( self, file_name ):
        roiLoc = self.maskDir + '%s.json'%file_name
        mask_roi = gm.readJson(  roiLoc )
        return mask_roi
    
    def saveWndchrmImage( self, img, img_name ):
        imgLoc = self.wndDir + '%s.tiff' % img_name
        gm.saveImg( imgLoc, img )
        
    def saveWndchrmScaler( self, scaler, name ):
        from pickle import dump
        scalerLoc = self.wndDir + '%s_scaler.pkl' % name
        dump( scaler, open( scalerLoc, 'wb') )
        
    def readWndchrmScaler( self, name ):
        from pickle import load
        scalerLoc = self.wndDir + '%s_scaler.pkl' % name
        scaler = load( open( scalerLoc, 'rb') )
        return scaler
        
    def saveWndchrmDF( self, df, name ):
        dfLoc = self.wndDir + '%s.pkl' % name
        df.to_pickle( dfLoc )
        
    def readWndchrmDF( self, name ):
        dfLoc = self.wndDir + '%s.pkl' % name
        df = pd.read_pickle( dfLoc )
        return df
    
    def saveWndchrmNorm( self, norm_dict, norm_name ):
        jsonLoc = self.wndDir + '%s.json' % norm_name
        gm.saveJson( norm_dict, jsonLoc, pretty=True )
    
    def readWndchrmNorm( self, file_name ):
        jsonLoc = self.wndDir + '%s.json' % file_name
        norm_dict = gm.readJson(  jsonLoc )
        return norm_dict
        
    
    
        
# End target info class
    
def tabprint( inprint, begin = '\t - ', end = '\n' ):
    print('%s%s' % (begin,inprint), end=end )


    
if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main

