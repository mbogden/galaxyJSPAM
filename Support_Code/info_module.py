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

class group_score_parameter_class:

    group = {}
    status = False

    def __init__( self, pLoc = None ):

        if pLoc != None:			
            with open( pLoc, 'r' ) as iFile:
                self.group = json.load( iFile )
        
        if self.group != {}:
            self.status = True

    def addGroupParam( self, inDict ):
        self.group = inDict

    def addParamClass( self, paramIn ):		
        if paramIn.status:
            self.group[paramIn.get('name')] = paramIn.pDict

    def addParam( self, pDict ):
        self.group[ pDict['name'] ] = pDict

    def rmParam( self, pKey ):
        if pKey in self.group:
            del self.group[ pKey ]

    def printGroup( self, ):		
        print(self.group)

    def get( self, inVal, default = None, pName = None ):

        # Grab from class first
        cVal = getattr( self, inVal, None )
        if cVal != None:
            return cVal

        # Grab from param dicts next
        if pName == None:
            dVal = self.group.get( inVal, default )
        else:
            dVal = self.group[pName].get( inVal, default )

        return dVal	

    def saveParam(self, saveLoc=None):

        if saveLoc == None:
            return

        with open( saveLoc, 'w' ) as pFile:
            json.dump( self.group, pFile, indent=4 )

    def readParam(self, pLoc):
        with open( pLoc, 'r' ) as iFile:
            self.group = json.load( iFile )


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

    def __init__( self, paramLoc = None, paramDict = None, new=False, printBase = True, printAll = False ):

        self.printBase = printBase
        self.printAll = printAll
        if self.printAll: self.printBase = True

        if self.printBase:
            print("IM: score_param_class.__init__")
            print("\t - paramLoc: ", paramLoc)
            print("\t - paramDict: ", type(paramDict))

        # If creating param new from scratch
        if new:
            self.pDict = self.baseDict
            self.status = True

        elif paramDict != None:
            self.pDict = paramDict
            self.status = True

        # Else reading a param file
        elif paramLoc != None:
            self.readParam( paramLoc )

        if self.printAll:
            pprint( self.pDict )

    def setDict( self, paramDict):
        self.pDict = paramDict

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

        if self.printBase: 
            print("IM: run_info_class.__init__")
            print("\t - runDir: " , runDir )


        # initialize directory structure for run
        dirGood = self.initRunDir( rArg )

        if not dirGood:
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
    
    def getModelImg( self, imgName = 'default' ):
        
        imgLoc = self.findImgLoc( pName = imgName )
        
        if imgLoc != None:
            img = gm.readImg(imgLoc)
            return img
        
        else:
            return None

    def findImgLoc( self, pName, initImg = False, newImg=False, ):		 

        # Assume model image
        if not initImg:
            imgLoc = self.imgDir + pName + '_model.png'

        else:
            imgLoc = self.miscDir + pName + '_init.png'

        if newImg or path.exists( imgLoc ):
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

    def printScores( self, allScores=False):

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

        self.saveInfoFile()

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
        
        if targetDir == None: targetDir = tArg.targetDir
            
        if printAll != None:   
            self.printAll = printAll
        else:            
            self.printAll = tArg.printAll            
            
        # To avoid future confusion
        if self.printAll:  self.printBase = True

        if self.printAll: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', targetDir)
            
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

        # return if invalid request
        if tName == None:
            return
        
        # Create tmp target image dict if not found
        if self.get('targetImgs',None) == None:
            self.targetImgs = {}
        
        # Search if target image was previously loded. 
        if type( self.targetImgs.get(tName,None) ) != type( None ):
            return self.targetImgs[tName]
        
        # Else find and open target image
        tLoc = self.findTargetImage(tName)
        
        if not gm.validPath(tLoc,):
            return None
        
        else:
            self.targetImgs[tName] = gm.readImg(tLoc)
            return self.targetImgs[tName]
        
    # End getTargetImage()


    def findTargetImage( self, tName = None ):

        tLoc = self.infoDir + 'target_%s.png' % tName
        if path.exists( tLoc ):
            return tLoc
        else:
            return None

    def printParams( self, ):

        for pKey in self.tDict['score_parameters']:
            print( self.tDict['score_parameters'][pKey] )


    def addScoreParam( self, paramLoc = None, paramDict = None, overwrite = False):

        if paramDict == None and paramLoc == None:
            return None

        if paramLoc != None:			
            spClass = score_parameter_class( paramLoc = paramLoc, printBase =False )

        elif paramDict != None:
            spClass = score_parameter_class( paramDict = paramDict, printBase=False )

        if not spClass.status:
            return None

        # If score parameters already present
        if self.tDict['score_parameters'].get( spClass.get('name'), None) != None and not overwrite:
            return self.tDict['score_parameters'].get( spClass.get('name'), None)

        pName = spClass.get('name')
        self.tDict['score_parameters'][pName] = spClass.pDict
        self.saveInfoFile()
        del spClass
        return self.tDict['score_parameters'][pName]


    def getScores( self, scrName = None ):

        # Reading from file
        if type(self.sFrame) == type(None):
            if gm.validPath( self.scoreLoc, printWarning=False ):
                self.sFrame = pd.read_csv( self.scoreLoc )
            else:
                if self.printAll: print( "IM: WARNING: Target.getScores:" )
                self.createBaseScore()

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
            rDict = zDict[rKey]
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

    def getRunDict( self, rDir, modelSet, rArg=gm.inArgClass() ):

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

        modelSet[rID] = rInfo.rDict

        # update progress

        return modelSet[rID]

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
        
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.baseScoreLoc = self.infoDir + 'base_scores.csv'

        self.zooMergerLoc = self.infoDir + 'galaxy_zoo_models.txt'
        
        # If using old folder layout, rename        
        if gm.validPath( self.gen0 ):
            from os import rename
            print("Found old path   : %s" % self.gen0)
            print("Propose new path : %s" % self.zooMergerDir)
            rename(self.gen0,self.zooMergerDir)
            print("New Path? : %s" % gm.validPath(self.zooMergerDir) )
        
        elif gm.validPath( self.zooMergerDir ):
            if self.printBase:
                print("NEW PATH EXISTS: %s" % self.zooMergerDir)
        
        else: 
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        
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

        if not path.exists( self.zooMergerDir ):
            print("IM: WARNING: This message should not be seen.")
        
        # Check for base file
        newBase = tArg.get('newBase',False)
        if not path.exists( self.baseInfoLoc ) and not newBase:
            if self.printBase: 
                print("IM: WARNING: newTargetSetup:")
                tabprint("No base info file!")
                tabprint("Consider '-newInfo -newBase' command")
            return False
        
        if newBase:
            if self.printBase:
                print("IM: newTargetSetup:  Creating base files.")
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
        
        from os import getcwd
        from shutil import copyfile
                
        # Create blank base dict
        self.tDict = {}
        for key in self.baseHeaders:
            self.tDict[key] = {}
        
        # Find target id
        
        # Ask if parent directory name the target name
        tName = self.targetDir.split('/')[-2]
        '''
        print("Is this the target name? : %s" % tName)
        inVal = input('[y]es / [n]o?: ')
        print("input value: ",inVal)
        if inVal.lower() == 'y' or inVal.lower == 'yes':
            self.tDict['target_id'] = tName

        
        # Implement other methods if needed later
        else:
            print("Please implement new target naming method.")
            return False
        
        '''
        self.tDict['target_id'] = tName
        
        # Save
        with open( self.baseInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
        
        # Get starting target files
        cDir = getcwd()
        tImgDir = gm.validPath( cDir  + '/Input_Data/targets/' + tName + '/')
        tImgLoc = None

        if tImgDir == None:
            return False
        
        tFiles = listdir( tImgDir )
        for fName in tFiles:
            if '.png' in fName:
                tImgLoc = tImgDir + fName      
        
        if gm.validPath( tImgLoc ) != None:
            newImgLoc = self.infoDir + 'target_zoo.png'
            copyfile( tImgLoc, newImgLoc )
        
        if gm.validPath( newImgLoc ) == None:
            return False
        
        return True
        
    # end creating base info file
    
    
def tabprint( inprint, begin = '\t - ', end = '\n' ):
    print('%s%s' % (begin,inprint), end=end )


if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main

