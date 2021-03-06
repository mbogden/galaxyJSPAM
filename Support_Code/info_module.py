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

from sys import path as sysPath
sysPath.append('../')




from pprint import PrettyPrinter
pp = PrettyPrinter(width=41, compact=True)
pprint = pp.pprint

# For loading in Matt's general purpose python libraries
import general_module as gm

def test():
    print("IM: Hi!  You're in Matthew Ogden's information module for SPAM")
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

    if arg.dataDir != None:
        print("IM: dataDir: %s" % arg.dataDir )

        saveLoc = getattr( arg, 'saveLoc', None )
        if saveLoc == None:
            print("IM: dataDir: Please specify saveLoc")
            return False

        print("IM: dataDir: creating gathered info file: %s" % saveLoc )

        dataJson = {}
        dataJson['data_dump_name'] = '20200409 score dump'
        dataJson['data_targets'] = []

        print("IM: dataDir: Gathering info files" )

        sdssDirs = listdir( arg.dataDir )
        nDirs = len( sdssDirs )

        for i, sDir in enumerate( sdssDirs ):
            print(" %d / %d " % ( i, nDirs ), end='\r')

            sdssDir = arg.dataDir + sDir +'/'
            infoLoc = sdssDir + 'information/target_info.json'

            if path.exists( infoLoc ):
                with open( infoLoc ) as iFile:
                    dataJson['data_targets'].append( json.load( iFile ) )

        print('')
        print("IM: dataDir: Found %d targets" % len( dataJson['data_targets'] ) )

        with open( saveLoc, 'w' ) as oFile:
            json.dump( dataJson, oFile )

    if getattr( arg, 'param', None ) != None:
        print('yAY')
        sp = score_parameter_class( paramLoc = arg.param, printBase = arg.printBase, printAll = arg.printAll ) 

# End main

class run_info_class:

    rDict = None	# Primary dict of information contained in info.json
    initDict = None	# State of json upon initial reading.
    baseDict = None

    status = False  # State of this class.  For initiating.

    runDir = None
    infoLoc = None
    baseLoc = None

    ptsDir = None
    imgDir = None
    miscDir = None
    
    tLink = None


    runHeaders = ( 'run_id', 'model_data', \
            'zoo_merger_score', 'machine_scores',)

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
            self.status = False
            return
                 
        # Read info file
        if path.exists( self.infoLoc ):

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
        self.tmpDir = self.runDir + 'tmp/'
        self.infoLoc = self.runDir + 'info.json'
        self.baseLoc = self.runDir + 'base_info.json'
    
        # If newInfo or newBase
        if rArg.get('newInfo',False) or rArg.get('newBase',False):
            self.newRunSetup(rArg)

        # Print stuff if needed
        if self.printAll:
            print("\t - runDir: (%s) %s" % ( path.exists( self.runDir ), self.runDir ) )
            print("\t - ptsDir: (%s) %s" % ( path.exists( self.ptsDir ), self.ptsDir ) )
            print("\t - imgDir: (%s) %s" % ( path.exists( self.imgDir ), self.imgDir ) )
            print("\t - miscDir: (%s) %s" % ( path.exists( self.miscDir ), self.miscDir ) )
            print("\t - infoLoc: (%s) %s" % ( path.exists( self.infoLoc ), self.infoLoc ) )
            print("\t - baseLoc: (%s) %s" % ( path.exists( self.baseLoc ), self.baseLoc ) )

        # Check if things are working
        dirGood = True

        if not path.exists( self.ptsDir ) or not path.exists( self.imgDir ) or not path.exists( self.miscDir ):
            if self.printAll: 
                print("IM: Run. WARNING!  Particle directory not found!")
                print("IM: Run. WARNING!  Model Image directory not found!")
                print("IM: Run. WARNING!  Misc Image directory not found!")
            dirGood = False

        # If you made it this far.  
        return dirGood

    # End initialize run directory structure
    
    # If asked to create new info file
    def newRunSetup( self, rArg ):
        
        from os import remove
        from shutil import copyfile
        
        # Remove info file(s) if condition given
        newInfo = rArg.get('newInfo',False)
        newBase = rArg.get('newBase',False)

        # Remove current info file
        if path.exists( self.infoLoc ): remove( self.infoLoc )
            
        # If new Base
        if newBase:  self.newRunDir()

        # WORKING
        if path.exists( self.baseLoc ): copyfile( self.baseLoc, self.infoLoc )
        # End
    
    # If creating directory from scratch or prior state
    def newRunDir( self, ):
        
        from os import mkdir, remove
        from shutil import move        
        
        # Create directories if not found
        if not path.exists( self.ptsDir ): mkdir( self.ptsDir )
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.miscDir ): mkdir( self.miscDir )
        if not path.exists( self.tmpDir ): mkdir( self.tmpDir )
        
        # Check if unperturbed imgs are in model dir
        imgList = listdir( self.imgDir )
        for fName in imgList:
            if 'init' in fName:
                oldImgLoc = self.imgDir + fName
                newImgLoc = self.miscDir + fName
                move(oldImgLoc,newImgLoc)

        # Remove current base file if present
        self.txt2Json( )            
            
    def __del__(self,):
        pass

    def delTmp(self,):

        from os import remove
        for f in listdir( self.tmpDir ):
            #print('Removing: %s'%f)
            remove(self.tmpDir + f)


    def findPtsLoc( self, ptsName ):

        ptsLoc = self.ptsDir + ptsName + '_pts.zip'

        # IF not found, try again without the k
        if not path.exists( ptsLoc ):

            # Check for a letter
            if 'k' in ptsName:
                ptsName = str( int( ptsName.strip('k') ) * 1000 )

            elif 'K' in ptsName:
                ptsName = str( int( ptsName.strip('K') ) * 1000 )

            ptsLoc = self.ptsDir + ptsName + '_pts.zip'

        if path.exists( ptsLoc ):
            return ptsLoc
        else:
            return None

    # End findPtsFile
    
    def getModelImage( self, imgName = 'default', initImg = False ):
        
        # Create place to store images if needed.
        if self.get('img',None) == None:
            self.img = {}
            
        if self.get('init',None) == None:
            self.init = {}
        
        # Return model image if already loaded.
        if not initImg:
            mImg = self.img.get(imgName,None)
            if type(mImg) != type(None):
                return mImg
        
        else:            
            img = self.init.get(imgName,None)
            if type(img) != type(None):
                return img
        
        # Get image location
        imgLoc = self.findImgLoc( imgName = imgName, initImg = initImg )
        if self.printAll: print("IM: Loading imgLoc: %s" % imgName, initImg )
        if imgLoc == None:
            return None
        
        # Read image
        img = gm.readImg(imgLoc)
        
        # Store image if called upon later
        if not initImg: 
            self.img[imgName] = img
        else:
            self.init[imgName] = img

        # Return image
        return img
    
    # End getting model, unperturbed image


    def findImgLoc( self, imgName, initImg = False, newImg=False ):

        # Assume model image
        if not initImg:
            imgLoc = self.imgDir + imgName + '_model.png'

        else:
            imgLoc = self.miscDir + imgName + '_init.png'

        # If new image, return location it will become
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

        if self.rDict == None:
            return None
        elif self.get('machine_scores') == None:
            self.rDict['machine_scores']
            return None

        score = self.rDict['machine_scores'].get( sName, None )

        return score

    def getAllScores( self  ):
        return self.get('machine_scores')

    def printScores( self, allScores=True):

        print("IM: run_info_class.printScores()")
        tabprint( 'run_id: %s' % self.get('run_id') )
        tabprint( 'zoo_merger: %f' % self.get('zoo_merger_score'))
        tabprint( 'machine_scores: %d' % len(self.rDict['machine_scores']) )

        if allScores:
            for sKey in self.rDict['machine_scores']:
                print( '\t - %s: %f' %  (sKey, self.getScore( sKey )) )

    def addScore( self, name=None, score=None ):

        if name == None or score == None:
            if self.printAll:
                print("IM: WARNING: run.addScore. name or score not given")
                print('\t - name: ', name)
                print('\t - score: ', score)
            return None

        self.rDict['machine_scores'][name] = score

        return self.rDict['machine_scores'][name]


    def createBlank( self, ):

        # Create blank dictionary
        tempDict = {}
        for key in self.runHeaders:
            tempDict[key] = {}

        return tempDict
    # End create Blank


    def get( self, inVal, defaultVal = None ):

        cVal = getattr( self, inVal, defaultVal )

        if cVal != defaultVal:
            return cVal

        dVal = self.rDict.get( inVal, defaultVal )
        if dVal != defaultVal:
            return dVal

        return defaultVal


    def txt2Json( self, ):

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
        if ( tid == None or rNum == None or mData == None or wins == None or hScore == None ):
            print("Error: IM: Needed information not found in info.txt")
            print("\t - infoLoc: %s" % infoLoc)
            return None

        # readjust run_id

        self.rDict['run_id'] = 'r' + str( rNum )
        self.rDict['model_data'] = str(mData)
        self.rDict['zoo_merger_score'] = float(hScore)

        # created initial info.json file from info.txt
        self.baseDict = deepcopy(self.rDict)

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

    printBase = True
    printAll = False

    targetDir = None
    zooMergerDir = None
    plotDir = None

    baseInfoLoc = None
    allInfoLoc = None
    scoreLoc = None
    baseScoreLoc = None
    zooMergerLoc = None


    # For user to see headers.
    targetHeaders = ( 'target_id', 'target_images', 'simulation_parameters', 'image_parameters', 'zoo_merger_models', 'score_parameters', 'progress', 'model_sets' )
    
    baseHeaders = ( 'target_id', )


    def __init__( self, targetDir = None, tArg = gm.inArgClass(), \
            printBase = None, printAll = None, ):

        # Tell class what to print
        if printBase != None:
            self.printBase = printBase
        else:
            self.printBase = tArg.printBase
        
        if tArg.targetDir == None: tArg.targetDir = targetDir
            
        if printAll != None:   
            self.printAll = printAll
        else:            
            self.printAll = tArg.printAll            
            
        # To avoid future confusion
        if self.printAll:  self.printBase = True

        if self.printAll: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', tArg.targetDir)
            
        # Check if directory has correct structure
        newInfo = tArg.get('newInfo',False)
        dirGood = self.initTargetDir( tArg )

        # Complain if not
        if not dirGood:
            if self.printBase:
                print("IM: Target.__init__(): ")
                print("\t - WARNING: Something went wrong initializing directory.")
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

    def getTargetImage( self, tName = None ):
        
        # Create place to store images if needed.
        if self.get('targetImgs',None) == None:
            self.targetImgs = {}
        
        # Return target image if already loaded.
        tImg = self.targetImgs.get(tName,None)
        if type(tImg) != type(None):
            return tImg
        
        # Else find and open target image
        tLoc = self.findTargetImage(tName)
        
        if not gm.validPath(tLoc,):
            return None
        
        else:
            self.targetImgs[tName] = gm.readImg(tLoc)
            return self.targetImgs[tName]
        
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
                if self.printAll: print( "IM: WARNING: Target.getScores:" )
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

            # Save score info

            continue


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
    def gatherRunInfos( self, tArg=gm.inArgClass(), rArg=gm.inArgClass(), ):

        if self.printAll: 
            print( "IM: Target.gatherRunInfos." )
            rArg.printArg()

        runDirList = self.iter_runs()
        nRuns = len(runDirList)

        # Prepare model Set
        for h in self.targetHeaders:
            if self.tDict.get(h,None) == None:
                self.tDict[h] = {}

        modelSet = self.tDict['zoo_merger_models']

        # Prepare parallel class
        ppClass = gm.ppClass( tArg.nProc, printProg=True )
        sharedModelSet = ppClass.manager.dict()

        argList = [ dict( rDir=rDir, modelSet=sharedModelSet, rArg=rArg) for rDir in runDirList ]
        ppClass.loadQueue( self.getRunDict, argList )

        # Do parallel
        ppClass.runCores()

        # Save 
        self.tDict['zoo_merger_models'] = sharedModelSet.copy()

        self.saveInfoFile()

    # end gather Run Infos

    def getRunDict( self, rDir, modelSet=None, rArg=gm.inArgClass() ):

        rArg.runDir = rDir
        rInfo = run_info_class( printBase=False, rArg=rArg )

        if rInfo.status == False:
            return None

        # Save info
        rID = rInfo.get('run_id')

        if 'r' not in rID:
            rInfo.run_id = 'r'+str(rID)
            rInfo.rDict['run_id'] = 'r'+str(rID)
            rInfo.saveInfoFile()

        if modelSet != None:
            modelSet[rID] = rInfo.rDict

        # update progress

        return rInfo.rDict

    # End get Run Dir Info


    def getRunDir( self, rID=None,  ):

        if rID[0] == 'r':
            rID = rID[1:]
            
        runDir = self.zooMergerDir + 'run_%s/' % rID

        # Try filling in zeros if short integer
        if not path.exists( runDir ):
            rID = rID.zfill(5)
            runDir = self.zooMergerDir + 'run_%s/' % rID
        
        return gm.validPath(runDir)

    def getRunInfo( self, rID=None, rArg=None ):
        
        runDir = self.getRunDir(rID=rID)
        
        if runDir == None:
            return

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
    
    def getAllRunDicts( self, ):
        return self.tDict['zoo_merger_models']

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



    # initialize target directories
    def initTargetDir( self, tArg ):
        
        from shutil import move
        from os import mkdir
        
        if self.printAll:
            print( 'IM: Target.initTargetDir():' )
            print( '\t - targetDir: %s' % tArg.targetDir )

        self.targetDir = gm.validPath( tArg.targetDir )

        # if Invalid, complain
        if type(self.targetDir) == type(None):
            if self.printBase:
                print("IM: WARNING: Invalid directory.")
            return False

        # If not directory, complain
        elif not path.isdir( self.targetDir ):
            if self.printBase:
                print("IM: WARNING: Target:")
                print("\t - Target not a directory")
            return False

        # Define paths for all useful things in target structure
        
        self.infoDir = self.targetDir + 'information/'
        self.gen0 = self.targetDir + 'gen000/'
        self.zooMergerDir = self.targetDir + 'zoo_merger_models/'
        self.plotDir = self.targetDir + 'plots/'

        self.allInfoLoc = self.infoDir + 'target_info.json'
        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        
        self.imgDir = self.infoDir + 'target_images/'
        self.imgParamLocOld = self.infoDir + 'param_target_images.json'
        self.imgParamLoc = self.imgDir + 'param_target_images.json'        
        
        self.maskDir = self.infoDir + 'target_masks/'
        
        self.scoreParamDir = self.infoDir + 'score_parameters/'
        
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.baseScoreLoc = self.infoDir + 'base_scores.csv'

        self.zooMergerLoc = self.infoDir + 'galaxy_zoo_models.txt'
        
        # Create target image and mask directories
        if not path.exists( self.infoDir ): mkdir( self.infoDir )
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.maskDir ): mkdir( self.maskDir )
        if not path.exists( self.scoreParamDir ): mkdir( self.scoreParamDir )
        
        # Create or move misc directories if not created.
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
        
        if tArg.get('newInfo',False): 
            status = self.newTargetSetup( tArg )
            if status == False: return False

        # Check if directories are found        
        if ( not path.exists( self.infoDir ) \
        or not path.exists( self.zooMergerDir ) \
        or not path.exists( self.plotDir ) ):
            
            if self.printBase:
                print("IM: Target_init: Some directories not found")
                tabprint('Info: %s' % gm.validPath(self.infoDir))
                tabprint('Plot: %s' % gm.validPath(self.plotDir))
                tabprint('Zoo Merger: %s' % gm.validPath(self.zooMergerDir))
                tabprint("Consider using -newInfo command")
                
            return False
            
        
        # Check if info directory has needed objects
        if not path.exists( self.allInfoLoc ) \
        or not path.exists( self.baseInfoLoc ) \
        or not path.exists( self.scoreLoc ) \
        or not path.exists( self.baseScoreLoc ):
            if self.printBase:
                print("IM: Target_init: Needed information files not found.")
                tabprint('Main Info: %s' % gm.validPath(self.allInfoLoc))
                tabprint('Base Info: %s' % gm.validPath(self.baseInfoLoc))
                tabprint('Main Score: %s' % gm.validPath(self.scoreLoc))
                tabprint('Base Score: %s' % gm.validPath(self.baseScoreLoc))
                tabprint("Consider using -newInfo command")
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
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.maskDir ): mkdir( self.maskDir )

        if not path.exists( self.zooMergerDir ):
            print("IM: WARNING: This message should not be seen.")
        
        # Check for base file
        newBase = tArg.get('newBase',False)
        if not path.exists( self.baseInfoLoc ) and not newBase:
            if self.printBase: 
                print("IM: WARNING: newTargetSetup:")
                tabprint("No base info file!")
                tabprint("Consider '-newBase' command")
            return False
        
        if newBase:
            if self.printBase:
                createGood = self.createBaseInfo()
                if not createGood: return False
        
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

        self.gatherRunInfos( tArg=tArg, rArg=rArg )
        
        # Collect info 
        if newBase: self.createBaseScore()
        self.updateScores()
        self.saveInfoFile( )

    # End new target info dictionary
    
    
    def createBaseInfo( self, ):
        
        from os import getcwd, listdir
        from shutil import copyfile 
        from copy import deepcopy
        
        # For basic scores later on
        
        # Get parent directory this code suite is located in.
        simrDir = __file__.split('Support_Code/info_module.py')[0]
         
        # Assume directory name of target is target name
        tName = self.targetDir.split('/')[-2]
        
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
        elif metaLoc2 != None:
            mFile = open(metaLoc2,'r')
        else:
            if self.printBase: print("WARNING: IM: Meta data not found:")
            return False

        # Copy starting target zoo image param
        pLoc = simrDir + 'param/zoo_blank.json'
        blank_param = gm.readJson(pLoc)
        if blank_param == None:        
            if self.printBase: 
                print("WARNING: IM: Start zoo image param not found: %s"%pLoc)
            return False

        # Copy blank parameter
        new_params = {}
        new_name = 'zoo_0'
        new_params[new_name] = deepcopy(blank_param['zoo_blank'])

        # Make name and comments initial comments.
        new_params[new_name]['name'] = new_name
        new_params[new_name]['comment'] = 'Starting score parameters file for %s'%tName
        new_params[new_name]['imgArg']['comment'] = "Starting image parameters for %s"%tName

        # Grab information
        for l in mFile:
            l = l.strip()
            
            
            
            if 'height' in l:
                h = l.split('=')[1]
                new_params[new_name]['imgArg']['image_size']['width'] = int(h)
                
            if 'width' in l:
                w = l.split('=')[1]
                new_params[new_name]['imgArg']['image_size']['width'] = int(w)
            if 'px' in l:
                px = l.split('=')[1]
                new_params[new_name]['imgArg']['galaxy_centers']['px'] = int(px)
            if 'py' in l:
                py = l.split('=')[1]
                new_params[new_name]['imgArg']['galaxy_centers']['py'] = int(py)
            if 'sx' in l:
                sx = l.split('=')[1]
                new_params[new_name]['imgArg']['galaxy_centers']['sx'] = int(sx)
            if 'sy' in l:
                sy = l.split('=')[1]
                new_params[new_name]['imgArg']['galaxy_centers']['sy'] = int(sy)

        if self.printAll: gm.pprint(new_params)

        # Save new target image parameter
        newParamLoc = self.imgParamLoc
        gm.saveJson( new_params, newParamLoc, pretty=True )
        
        # Create basic scoring parameters
        self.createDirectScoreParameters( new_params['zoo_0'] )

        # Find pair file 
        pairPath1 = gm.validPath( inputDir + 'sdss%s.pair'%tName )
        pairPath2 = gm.validPath( inputDir + '%s.pair'%tName )
        if self.printAll: 
            gm.tabprint('pairPath1: ',pairPath1)
            gm.tabprint('pairPath2: ',pairPath2)
            
        if pairPath1 != None:            
            pairFile = open(pairPath1, 'r' )
            
        elif pairPath2 != None:            
            pairFile = open(pairPath2, 'r' )
            
        else:
            if self.printBase: print("WARNING: IM: Pair data not found: %s"%pairPath)
            return False

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
        
        # If None, initialize target images and try again.
        if img_params == None:
            getTargetInputData( self )
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
            gm.saveJson( in_params, self.imgParamLoc )
            return
        
        # Else file does exsit
        old_params = gm.readJson( self.imgParamLoc )
        
        # Loop through new image parameters
        for sKey in in_params:
            
            # Add to old if overwrite or not found
            if overWrite or sKey not in old_params:
                old_params[sKey] = in_params[sKey]            
        
        gm.saveJson( old_params, self.imgParamLoc )
    
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
        
        
# End target info class
    
def tabprint( inprint, begin = '\t - ', end = '\n' ):
    print('%s%s' % (begin,inprint), end=end )


    
if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main

