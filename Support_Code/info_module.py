'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    02 Apr 2020
Description:    For all things misc information related
'''

from os import path, listdir
import json
from copy import deepcopy

import pandas as pd
import numpy as np


from pprint import PrettyPrinter
pp = PrettyPrinter( indent = 2 )
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

    if arg.targetDir != None:

        tInfo = target_info_class( \
                targetDir = arg.targetDir, \
                printAll = arg.printAll, \
                newInfo = arg.get('newInfo', False ), \
            )

        if tInfo.status:
            print("IM: target_info_class good")

        if getattr( arg, 'updateProgress', False ):
            tInfo.updateProgress()


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


class score_parameter_class:

    pDict = None
    status = False

    baseDict = {
            'name' : None,
            'simArg' : {
                    'name' : '100k',
                    'nPts' : '100k',
                },
            'imgArg' : {
                    'name' : 'default',
                    'pType' : 'default',
                },
            'tgtArg' : "zoo",
            'scrArg' : {
                    'cmpMethod' : 'correlation',
                },
        }

    def __init__( self, paramLoc = None, new=False, printBase = True, printAll = False ):

        self.printBase = printBase
        self.printAll = printAll
        if self.printAll: self.printBase == True

        if self.printBase:
            print("IM: score_param_class.__init__")
            print("\t - paramLoc: ", paramLoc)

        # If creating param new from scratch
        if new:
            self.pDict = self.baseDict
            self.status = True

        # Else reading a param file
        else:
            self.readParam( paramLoc )

        if self.printAll:
            pprint( self.pDict )

    def get( self, inVal, defaultVal = None ):

        cVal = getattr( self, inVal, defaultVal )

        if cVal != defaultVal:
            return cVal

        dVal = self.pDict.get( inVal, defaultVal )
        if dVal != defaultVal:
            return dVal

        return defaultVal


    def readParam( self, paramLoc ):
        
        if self.printAll:
            print("IM: score_param_class.readParam")
            print("\t - paramLoc: ", paramLoc)
        
        # Check if param File is valid
        if paramLoc == None:
            if self.printBase: 
                print('IM: WARNING: Please give a param File Location')
                print('\t -paramLoc /path/to/file.txt')
            return

        elif type( paramLoc) != type('String'):
            if self.printBase: 
                print('IM: WARNING: paramLoc variable not string')
                print('\t -paramLoc: %s ' % type(paramLoc), paramLoc)
            return

        elif not path.exists( paramLoc ):
            if self.printBase: 
                print('IM: WARNING: Param File location not found')
                print('\t -paramLoc: %s' % paramLoc)
            return

        # Read file Contents
        with open( paramLoc ) as iFile:
            self.pDict = json.load( iFile )

        if self.pDict == None:
            if self.printBase: 
                print('IM: WARNING: Failed to read param file')
                print('\t -paramLoc: %s' % paramLoc)
            return
        
        self.status = True

    # End reading param file

    def printParam( self, ):
        pprint( self.pDict )
    # end print

# End score parameter class
        


class target_info_class:

    tDict = None        # Primary dictionary of info for all data 
    initDict = {}       # Copy of dictionary when read
    baseDict = None     # Simple dictionary of info for basic info.
    sFrame = None       # Score dataframe
    runClassDict = None # List of run classes

    status = False      # State of this class.  For initiating.
    printBase = True
    printAll = False

    targetDir = None
    zooMergerDir = None
    plotDir = None

    baseInfoLoc = None
    allInfoLoc = None
    scoreLoc = None
    zooMergerLoc = None



    # For user to see headers.
    targetHeaders = ( 'target_identifier', 'target_images', 'model_image_parameters', 'model_sets',)

    progHeaders = ( 'initial_creation', 'galaxy_zoo_models', '100k_particle_files', 'zoo_default_model_images', 'machine_scores' )

    def __init__( self, targetDir = None, \
            printBase = True, printAll = False, \
            newInfo=False, newRunInfos = False, \
            gatherRuns=False):

        # Tell class what to print
        self.printBase = printBase
        self.printAll = printAll
        if self.printAll:  self.printBase = True

        if self.printAll: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', targetDir)
            print('\t - printBase: ', printBase)
            print('\t - printAll: ', printAll)
            print('\t - newInfo: ', newInfo)
            print('\t - gatherRuns: ', gatherRuns)

        # if nothing given, complain
        if type(targetDir) == type(None):
            if self.printBase:
                print("IM: WARNING: ")
                print(" - No target Dir given.  No other option at this time.")
            return

        # Double check if target directory is valid
        elif type( targetDir ) != type( "String" ):
            if self.printBase:
                print("IM: WARNING: target_info_class.__init__")
                print("\t - Target Directory Not a String!")
                print("\t - %s" % type( self.targetDir ) )
            self.status = False
            return

        # If path not found, complain
        elif not path.exists( targetDir ):
            print("IM: WARNING: Target:")
            print("\t - Target dir not found")
            print('\t - targetDir: ', targetDir )
            return

        dirGood = self.initTargetDir( targetDir )

        if not dirGood:
            print("IM: Target.__init__(): ")
            print("\t - WARNING: Something went wrong initializing directory.")
            return
    
        # Remove info file if condition given
        if newInfo:

            if self.printAll: print("IM: Target.__init__. Removing previous info file.")
            from os import remove

            if path.exists( self.allInfoLoc ):
                remove( self.allInfoLoc )

            if path.exists( self.baseInfoLoc ):
                remove( self.baseInfoLoc )

            if path.exists( self.scoreLoc ):
                remove( self.scoreLoc )

        # end if

        if self.printAll: 
            print("IM: Target: Opening target info json")

        if path.exists( self.allInfoLoc ):

            with open( self.allInfoLoc ) as iFile:
                self.tDict = json.load( iFile )
            self.initDict = deepcopy( self.tDict )
            self.status = True

        # Copy base info file if all not found
        elif path.exists( self.baseInfoLoc ):

            if self.printAll:
                print("\t - target_info.json not found.")
                print("\t - Copying base_target_info.json")

            from os import system
            cpCmd = 'cp %s %s' % ( self.baseInfoLoc, self.allInfoLoc ) 
            system( cpCmd )

            with open( self.allInfoLoc ) as iFile:
                self.tDict = json.load( iFile )
            self.initDict = deepcopy( self.tDict )
            self.status = True

        # If target info does not exist, create
        else:
            if self.printAll: print("\t - Creating new info file" )
            
            self.newTargetInfo()
            self.saveInfoFile( baseFile=True )

        if gatherRuns or newInfo:
            self.gatherRunInfos( newInfo = newRunInfos )
            self.saveInfoFile( )

        if type(self.sFrame) == type(None) and path.exists( self.scoreLoc ):
            self.sFrame = pd.read_csv( self.scoreLoc )

        self.status = True
        return


    # End target init 

    # initialize variables in target directory
    def initTargetDir( self, targetDir ):

        if self.printAll:
            print( 'IM: Target.initTargetDir():' )
            print( '\t - targetDir: %s' % targetDir )

        # Define paths for all useful things in target structure
        self.targetDir = path.abspath( targetDir )
        if self.targetDir[-1] != '/': self.targetDir += '/'

        self.infoDir = self.targetDir + 'information/'
        self.allInfoLoc = self.infoDir + 'target_info.json'
        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.zooMergerLoc = self.infoDir + 'galaxy_zoo_models.txt'

        self.zooMergerDir = self.targetDir + 'gen000/'

        self.plotDir = self.targetDir + 'plots/'

        status = True

        # Check if everything needed is found
        if not path.exists( self.infoDir ):
            print("IM: WARNING: info directory not found!")
            status = False

        if not path.exists( self.zooMergerDir ):
            print("IM: WARNING: zoo models directory not found!")
            status = False

        if status and not path.exists( self.plotDir ):
            from os import mkdir
            mkdir( self.plotDir )
        
        if self.printAll:
            print( '\t - targetDir: (%s) %s' % ( path.exists( self.targetDir ), self.targetDir ) )
            print( '\t - infoDir: (%s) %s' % ( path.exists( self.infoDir ), self.infoDir ) )
            print( '\t - baseInfoLoc: (%s) %s' % ( path.exists( self.baseInfoLoc ), self.baseInfoLoc ) )
            print( '\t - allInfoLoc: (%s) %s' % ( path.exists( self.allInfoLoc ), self.allInfoLoc ) )
            print( '\t - zooMergerDir: (%s) %s' % ( path.exists( self.zooMergerDir ), self.zooMergerDir ) )
            print( '\t - plotDir: (%s) %s' % ( path.exists( self.plotDir ), self.plotDir ) )

        return status


    # Deconstructor
    def __del__( self ):
        if self.tDict != self.initDict:
            print("IM: WARNING: Target: target_info.json updated but not saved!")
    # End deconstructor

    def get( self, inVal, default = None ):
        return getattr( self, inVal, None )

    def printInfo( self, printAll=False ):
        # INCOMPLETE
        pprint( self.baseDict )
    # End print info

    # Create blank dictionary
    def createBlank( self, ):

        if self.printAll: print("IM: Target: Creating blank target_info_class")
        
        for h in targetHeaders:
            self.tDict[h] = None

        self.initDict = deepcopy( self.tDict )

        if self.printAll: print('IM: Target: info:\n' % self.rDict)
    # End creating blank target info

    def newTargetInfo( self, ):

        if self.printBase: 
            print("IM: newtargetInfo: Creating new target info file")

        # Create blank dict
        self.tDict = {}
        for key in self.targetHeaders:
            self.tDict[key] = {}

        # Find target/sdss name from progress file...
        progLoc = self.infoDir + 'progress.txt'
        pFile = gm.readFile( progLoc )
        tId = pFile[0].split()[-1]
        self.tDict['target_identifier'] = tId
        if self.printAll: print('\t- target_id: %s' % self.tDict['target_identifier'])

        # Search for target images and image parameter files
        iFiles = listdir( self.infoDir )

        for fName in iFiles:

            # if target image
            if 'target' in fName and '.png' in fName:
                self.tDict['target_images'][fName] = fName

            # if image parameter
            if 'param' in fName and '.txt' in fName and not 'parameters.txt' in fName:
                self.tDict['model_image_parameters'][fName] = fName

        # Set place for original galaxy zoo mergers models
        self.tDict['model_sets'] = { 'galaxy_zoo_mergers' : {} }

        # Make progress note
        self.tDict['progress'] = {}
        for key in self.progHeaders:
            self.tDict['progress'][key] = None
        self.tDict['progress']['initial_creation'] = True  

        if self.printAll: 
            print('\t - Created new info file')
            self.printInfo()

    # End new target info dictionary

    def getScores( self, scrName = None ):

        if type(self.sFrame) == type(None):
            if self.printAll: print( "IM: WARNING: Target.getScores:" )
            return None

        if scrName in self.sFrame.columns:
            return self.sFrame[ scrName ]
        
        return self.sFrame

    def getTargetImage( self, tName = None ):

        if tName == None:
            print("target is none")
            return

        tLoc = self.infoDir + 'target_%s.png' % tName
        if path.exists( tLoc ):
            return tLoc
        else:
            return None

    def getRunClass( self, runID ):

        if type( runID ) != type( 'string' ):
            runID = str( runID ).zfill(5)

        if self.get( 'runCLassDict', None ) == None:
            self.runClassDict = {}

        if self.runClassDict.get( runID ) != None:
            return self.runClassDict[runID]

        runDir = self.zooMergerDir + 'run_%s/' % runID

        rInfo = run_info_class( runDir = runDir, printBase=False)

        if rInfo.status:
            self.runClassDict[ runID ] = rInfo
            return self.runClassDict[ runID ]

        else:
            return None

        


    # Read in classes for run infos
    def readRunInfos( self, newInfo = False, model_key = 'galaxy_zoo_models', n = np.inf ):

        if model_key != 'galaxy_zoo_models':
            print("IM: WARNING: Target.readRunInfos")
            print('\t - gathering non zoo merger models not yet implemented')
            return

        self.runClassDict = {}

        # Get list of run directories
        self.runDirs = listdir( self.zooMergerDir )
        self.runDirs.sort()
        nRuns = len( self.runDirs )

        for i,run in enumerate(self.runDirs):

            rInfo = run_info_class( runDir = self.zooMergerDir + run, newInfo = newInfo, printBase=False)

            if rInfo.status == False:
                continue

            rId = rInfo.rDict['run_identifier']
            self.runClassDict[rId] = rInfo

            if self.printBase: print("IM: readRunInfos: %d/%d" % ( i, nRuns ), end='\r' )

            # used for troubleshooting quickly
            if i >= n: break

    # End reading run info classes

    #def updateRunInfo( self, rId, newInfo = False ):



    # Gather information from classes
    def gatherRunInfos( self, newInfo = False, model_key = 'galaxy_zoo_models' ):

        if self.printAll: print( "IM: Target.gatherRunInfos." )

        if model_key != 'galaxy_zoo_models':
            print("IM: WARNING: Target.gatherRunInfos")
            print('\t - gathering non zoo merger models not yet implemented')
            return

        if self.runClassDict == None:
            self.readRunInfos( newInfo = newInfo )

        nRuns = len( self.runClassDict )

        # shortened link to model set
        self.tDict['model_sets'][model_key] = {}
        modelSet = self.tDict['model_sets'][model_key]

        # For getting scores
        scoreHeaders = set( [ 'run_id', 'zoo_merger_score' ] )

        # Loop through collecting run dictionaries and score headers
        for i,rId in enumerate( self.runClassDict ):

            rInfo = self.runClassDict[rId]
            modelSet[rId] = rInfo.rDict

            # prep score headers 
            scrKeys = list( modelSet[rId].get('machine_scores').keys() ) 
            for key in scrKeys:
                scoreHeaders.add( key )

            if self.printBase: print("IM: gatherRunInfos: %d/%d" % ( i, nRuns ), end='\r' )

        self.tDict['model_sets']['galaxy_zoo_mergers'] = modelSet
        self.saveInfoFile()

        if self.printAll: 
            print( 'SIMR: GatherInfoRuns: Gathered %d run info files' % len( self.tDict['model_sets'][model_key] ) )

        # Gather scores

        nRuns = len( modelSet )

        self.sFrame = pd.DataFrame( \
                index = np.arange( nRuns ), \
                columns = list( scoreHeaders ) 
                )

        # Go back through and gather scores
        for i,rId in enumerate (modelSet ):
            rDict = modelSet[rId]

            # go through columns of frame
            for h in scoreHeaders:

                # Get run id
                if h == 'run_id':
                    self.sFrame.iloc[i]['run_id'] = rDict.get('run_identifier')

                # Get human score
                elif h == 'zoo_merger_score':
                    self.sFrame.iloc[i]['zoo_merger_score'] = rDict.get('zoo_merger_score')

                # get machine scores
                else:
                    self.sFrame.iloc[i][h] = rDict['machine_scores'].get( h )

        self.sFrame.to_csv( self.scoreLoc, index = False )
        if self.printAll:
            print( self.sFrame ) 

    # end gather Run Infos

    def saveInfoFile( self, saveLoc = None, baseFile=False ):

        if self.printAll: 
            print("IM: Target.saveInfoFile():")
            print("\t - Saving target info file...")

        if self.allInfoLoc == None and saveLoc == None:
            print("ERROR: IM: No target info location given...")
            return False

        if baseFile != False and self.baseInfoLoc == None:
            print("ERROR: IM: No base target info location given...")
            return False

        if self.tDict == self.initDict:
            if self.printAll: print("\t - No changes detected...")
            return True

        retVal = False

        with open( self.allInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
            retVal = True
            self.initDict = deepcopy( self.tDict )

        if baseFile:

            if self.printAll: print("IM: Target: Saving base info file")

            self.baseDict = {}
            self.baseDict['target_identifier'] = self.tDict['target_identifier']
            self.baseDict['target_images'] = self.tDict['target_images']
            self.baseDict['model_image_parameters'] = self.tDict['model_image_parameters']
            self.baseDict['zoo_merger_models'] = {}

            with open( self.baseInfoLoc, 'w' ) as infoFile:
                json.dump( self.baseDict, infoFile, indent=4 )
                if self.printAll: print("\t - Saved base info file")

        return retVal
   # End Target info class



class run_info_class: 

    rDict = None    # Primary dict of information contained in info.json
    initDict = None    # State of json upon initial reading.
    baseDict = None

    status = False  # State of this class.  For initiating.

    runDir = None
    infoLoc = None
    baseLoc = None

    ptsDir = None
    imgDir = None
    miscDir = None


    runHeaders = ( 'run_identifier', 'model_data', \
            'zoo_merger_score', 'machine_scores',)

    def __init__( self, runDir=None, printBase=True, printAll=False, newInfo=False, newRun=False
           ):

        # print 
        self.printAll = printAll
        self.printBase = printBase

        # Avoiding confusing things
        if self.printAll: self.printBase = True

        if self.printBase: 
            print("IM: run_info_class.__init__")
            print("\t - runDir: " , runDir )
            print("\t - printBase: ", printBase )
            print("\t - printAll: ", printAll )
            print("\t - newInfo: ", newInfo )
            print("\t - newRun: ", newRun )

        # Double check if run directory is valid
        if type( runDir ) != type( "String" ):
            if self.printBase:
                print("IM: WARNING: run_info_class.__init__")
                print("\t - Run Directory Not a String!")
                print("\t - %s" % type( self.runDir ) )
            self.status = False
            return

        # initialize directory structure for run
        dirGood = self.initRunDir( runDir )

        if not dirGood:
            return

        if self.printAll: print("IM: Run.__init__")

        # Remove info file if condition given
        if newInfo:

            if self.printAll: print('\t- Removing Info file.')
            from os import remove

            if path.exists( self.infoLoc ):
                remove( self.infoLoc )

            if path.exists( self.baseLoc ):
                remove( self.baseLoc )

        # Read info file
        if path.exists( self.infoLoc ):

            if self.printAll: print('\t - Reading Info file.')

            with open( self.infoLoc, 'r' ) as iFile:

                self.rDict = json.load( iFile )
                self.initDict = deepcopy( self.rDict )

            self.status = True
        # end read info file
        else:

            if self.printAll: print('\t - No info.json file.')
            # Create new run json from info file if it does not exist

            if path.exists( self.baseLoc ):
                if self.printAll: print('\t - Copying base_info.json file.')
                from os import system
                cpCmd = 'cp %s %s' % ( self.baseLoc, self.infoLoc )
                system( cpCmd )

                with open( self.infoLoc, 'r' ) as iFile:
                    self.rDict = json.load( iFile )

            else:
                self.txt2Json( )
 
            if type(self.rDict) == type(None):
                print("IM: Run.__init__ Error: Failed to initialize info file..." )
                return

               
            self.initDict = deepcopy( self.rDict )

            if self.printAll: 
                print("\t - Initialized run score file")
                self.printInfo()

            self.saveInfoFile()
            self.status = True

            return
        # End if not path exists
       

        if self.printAll: print("\t - Initalized info module.")

    # end __init__

    def findPtsFile( self, nPts ):

        
        # Check for a letter
        if 'k' in nPts:
            nPts = str( int( nPts.strip('k') ) * 1000 )

        elif 'K' in nPts:
            nPts = str( int( nPts.strip('K') ) * 1000 )


        ptsLoc = self.ptsDir + nPts + '_pts.zip'

        if path.exists( ptsLoc ):
            return ptsLoc
        else:
            return None

    # End findPtsFile

    def findImgFile( self, pName, initImg = False ):

        imgLoc = self.imgDir + pName + '_model.png'

        if not initImg:
            if path.exists( imgLoc ):
                return imgLoc
            else:
                return None

        else: 

            initLoc = self.miscDir + pName + '_init.png'

            if path.exists( imgLoc ) and path.exists( initLoc ):
                return imgLoc, initLoc
            else:
                return None, None
   
    # End findImgFile

    def initRunDir( self, runDir, newDir = False, ):

        # Print stuff
        if self.printAll:
            print("IM: run.initRunDir")
            print("\t - runDir: %s" % runDir )
            print("\t - newDir: %s" % str( newDir ) )

        # Check if path exists
        if not path.exists( runDir ):
            print("IM: WARNING: initRunDir")
            print("\t - runDir: '%s'" % runDir )
            print("\t - Non-Valid Directory")
            print("\t - Considering implementing newDir")
            return False

        # Save base directory
        self.runDir = path.abspath( runDir )
        if self.runDir[-1] != '/': self.runDir += '/'

        # Hard code location of main objects
        self.ptsDir = self.runDir + 'particle_files/'
        self.imgDir = self.runDir + 'model_images/'
        self.miscDir = self.runDir + 'misc_images/'
        self.infoLoc = self.runDir + 'info.json'
        self.baseLoc = self.runDir + 'base_info.json'

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

        if not path.exists( self.ptsDir ):
            print("IM: Run. WARNING!  Particle directory not found!")
            dirGood = False

        if not path.exists( self.imgDir ):
            print("IM: Run. WARNING!  Model Image directory not found!")
            dirGood = False

        if not path.exists( self.miscDir ):
            print("IM: Run. WARNING!  Misc Image directory not found!")
            dirGood = False

        # If you made it this far.  
        return dirGood

    # End initialize run directory structure

    # deconstructor
    def __del__( self ):

        if self.rDict != self.initDict:
            print("IM: WARNING: info.json updated but not save!")
    # End deconstructor


    def printInfo( self, allInfo = False):
        from pprint import PrettyPrinter
        pp = PrettyPrinter( indent = 2 )

        print( "IM: run_info_class.printInfo():")

        if allInfo:
            pprint( self.rDict )

    def getScore( self, sName,  ):

        score = self.rDict['machine_scores'].get( sName, None )
        
        return score

    def addScore( self, name=None, score=None ):

        if name == None or score == None:
            if self.printBase:
                print("IM: WARNING: run.addScore. name or score not given")
                print('\t - name: ', name)
                print('\t - score: ', score)
            return None

        print( self.rDict['machine_scores'] )
        self.rDict['machine_scores'][name] = score
        print( self.rDict['machine_scores'] )
        print( self.getScore( name ) )

        return self.rDict['machine_scores'][name]

   
    def createBlank( self, ):
        
        # Create blank dictionary
        tempDict = {}
        for key in self.runHeaders:
            tempDict[key] = {}

        return tempDict
    # End create Blank
       

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
            print("\t - oldLoc: %s" % infoLoc)
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
       
        # build lower structures first

        self.rDict['run_identifier'] = rNum
        self.rDict['model_data'] = mData
        self.rDict['zoo_merger_score'] = hScore

        # created initial info.json file from info.txt

        # Save results
        self.saveInfoFile( saveBase = True )

    # End txt2Json


    # end info2Dict

    def saveInfoFile( self, saveBase=False, ):

        if self.printAll: print("IM: Run.saveInfoFile: Saving info data file...")

        if self.rDict == self.initDict:
            if self.printAll: print("\t - No changes detected.")
            return True

        with open( self.infoLoc, 'w' ) as infoFile:
            json.dump( self.rDict, infoFile, indent=4 )
            retVal = True

        if saveBase:

            with open( self.baseLoc, 'w' ) as bFile:
                json.dump( self.baseDict, bFile, indent=4 )

        self.initDict = deepcopy( self.rDict )
    # End save info file

# End Run info class


if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main

