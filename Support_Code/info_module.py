'''
    Author:	 Matthew Ogden
    Created:	21 Feb 2020
    Altered:	02 Apr 2020
Description:	For all things misc information related
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
        if printBase == None:
            self.printBase = rArg.printBase
        else:
            self.printBase = printBase
        
        if printAll == None:
            self.printAll = rArg.printAll
        else:
            self.printAll = printAll
        
        if rArg.get('tInfo',None) != None: 
            self.tInfo = rArg.tInfo

        # Avoiding confusing things
        if self.printAll: self.printBase = True

        if self.printBase: 
            print("IM: run_info_class.__init__")
            print("\t - runDir: " , runDir )
            
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

        # Remove info file(s) if condition given
        newInfo = rArg.get('newInfo',False)
        newBase = rArg.get('newBase',False)
        
        if newInfo or newBase:

            from os import remove

            if path.exists( self.infoLoc ):
                if self.printAll: print('\t- Removing Info file.')
                remove( self.infoLoc )
            
            if newBase and path.exists( self.baseLoc ):
                if self.printAll: print('\t- Removing base info file.')
                remove( self.baseLoc )               
                
        # Read info file
        if path.exists( self.infoLoc ):

            if self.printAll: print('\t - Reading Info file.')
            with open( self.infoLoc, 'r' ) as iFile:
                self.rDict = json.load( iFile )

            if self.rDict != None:
                self.status = True

        # end read info file
        else:

            if self.printAll: print('\t - No info.json file.')
            # Create new run json from info file if it does not exist

            if path.exists( self.baseLoc ):
                if self.printAll: print('\t - Copying base_info.json file.')
                    
                with open( self.baseLoc, 'r' ) as bFile:
                    self.baseDict = json.load( bFile )
                
                # Check if base info is valid
                baseGood = True
                if self.baseDict == None:
                    baseGood = False
                    
                else:
                    for rKey in self.runHeaders:
                        if self.baseDict.get(rKey,None) == None:
                            baseGood = False                
                
                # if not create new
                if not baseGood:
                    if self.printAll: tabprint('Base info file void. creating...')
                    self.txt2Json()
                
                # if good
                else:
                    self.rDict = deepcopy( self.baseDict )
                    self.saveInfoFile()                    

            else:
                self.txt2Json( )

            if type(self.rDict) == type(None) and self.printBase:
                print("IM: Run.__init__ Error: Failed to initialize info file..." )
                return

            if self.printAll: 
                print("\t - Initialized run score file")
                self.printInfo()

            self.saveInfoFile()
            self.status = True

            return
        # End if not path exists


        if self.printAll: print("\t - Initalized info module.")

    # end __init__

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
            return None

        # Save base directory
        self.runDir = path.abspath( runDir )
        if self.runDir[-1] != '/': self.runDir += '/'

        # Hard code location of main objects
        self.ptsDir = self.runDir + 'particle_files/'
        self.imgDir = self.runDir + 'model_images/'
        self.miscDir = self.runDir + 'misc_images/'
        self.tmpDir = self.runDir + 'tmp/'
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

    tDict = None		# Primary dictionary of info for all data 
    baseDict = None	 # Simple dictionary of info for basic info.
    progDict = None	 # For saving progress of current target
    status = False	  # State of this class.  For initiating.

    sFrame = None	   # Score dataframe
    rInfo = None		# run_info_class for accessing run directories

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

    progHeaders  = ( 'machine_scores' )

    baseHeaders = ( 'target_id', )


    def __init__( self, targetDir = None, tArg = gm.inArgClass(), \
            printBase = None, printAll = None, ):

        # Tell class what to print
        if printBase != None:
            self.printBase = printBase
        else:
            self.printBase = tArg.printBase
        
        if targetDir == None:
            targetDir = tArg.targetDir
            
        if printAll != None:   
            self.printAll = printAll
        else:            
            self.printAll = tArg.printAll            
            
        if self.printAll:  self.printBase = True

        if self.printAll: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', targetDir)
            print('\t - printBase: ', printBase)
            print('\t - printAll: ', printAll)
            
        # Check if directory has correct structure
        dirGood = self.initTargetDir( targetDir )

        # Complain if not
        if not dirGood:
            if self.printBase:
                print("IM: Target.__init__(): ")
                print("\t - WARNING: Something went wrong initializing directory.")
            return

        # Remove infos if asked for
        newInfo = tArg.get('newInfo',False)
        if newInfo:

            if self.printAll: print("IM: Target.__init__. Removing previous info file.")

            # remove old files
            from os import remove

            if path.exists( self.allInfoLoc ): remove( self.allInfoLoc )
            if path.exists( self.progInfoLoc ): remove( self.progInfoLoc )
            if path.exists( self.scoreLoc ): remove( self.scoreLoc )

            # Copy base files if present
            from shutil import copyfile

            if path.exists( self.baseInfoLoc ): copyfile( self.baseInfoLoc, self.allInfoLoc )
            if path.exists( self.baseScoreLoc ): copyfile( self.baseScoreLoc, self.scoreLoc )

        # end new info

        if self.printAll: 
            print("IM: Target: Opening target info json")

        if path.exists( self.allInfoLoc ):

            with open( self.allInfoLoc ) as iFile:
                self.tDict = json.load( iFile )
            self.initDict = deepcopy( self.tDict )
            self.status = True

        # If target info does not exist, create
        else:
            if self.printAll: print("\t - Creating new info file" )
            #self.newTargetInfo( )
            self.saveInfoFile( baseFile=True )

        if not path.exists( self.scoreLoc ) and path.exists( self.baseScoreLoc ):
            from shutil import copyfile
            copyfile( self.baseScoreLoc, self.scoreLoc )

        # Open score file
        if path.exists( self.scoreLoc ):
            # Read all as string
            self.sFrame = pd.read_csv( self.scoreLoc )


        if newInfo:
                        
            # Should run infos be modified? 
            rArg = gm.inArgClass()
            rArg.printBase = False
            if tArg.get('newRunInfos',True):
                rArg.newInfo = True
            if tArg.get('newRunBase',True):
                rArg.newBase = True
            
            self.gatherRunInfos(rArg=rArg)
            
            self.createBaseScore()
            self.updateScores()
            self.saveInfoFile( )

        self.status = True
        return

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
    def gatherRunInfos( self, rArg=gm.inArgClass() ):

        if self.printAll: print( "IM: Target.gatherRunInfos." )

        runDirList = self.iter_runs()
        nRuns = len(runDirList)

        # Prepare model Set
        for h in self.targetHeaders:
            if self.tDict.get(h,None) == None:
                self.tDict[h] = {}

        modelSet = self.tDict['zoo_merger_models']

        # Generate parellel processing class

        # Prepare parallel class
        ppClass = gm.ppClass( -1, printProg=True )
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

        rInfo = run_info_class( runDir = rDir, printBase=False, rArg=rArg )

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

    def getRunInfo( self, rID=None, runArg=None ):
        
        runDir = self.getRunDir(rID=rID)
        
        if runDir == None:
            return

        if runArg == None:
            rInfo = run_info_class( runDir = runDir, )
            
        else: 
            rInfo = run_info_class( runDir = runDir, args=runArg )

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

    def saveInfoFile( self, baseFile=False ):

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

        # Save progress seperate
        self.progDict = {}
        self.progDict['target_id'] = self.tDict['target_id']
        self.progDict['progress'] = self.tDict['progress']
        self.progDict['simulation_parameters'] = self.tDict['simulation_parameters']
        self.progDict['image_parameters'] = self.tDict['image_parameters']
        self.progDict['score_parameters'] = self.tDict['score_parameters']

        with open( self.progInfoLoc, 'w' ) as pFile:
            json.dump( self.progDict, pFile, indent=4 )

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
    def initTargetDir( self, targetDir ):
        
        
        if self.printAll:
            print( 'IM: Target.initTargetDir():' )
            print( '\t - targetDir: %s' % targetDir )

        self.targetDir = gm.validPath(targetDir)

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
        self.zooMergerDir = self.targetDir + 'gen000/'
        self.plotDir = self.targetDir + 'plots/'

        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        self.progInfoLoc = self.infoDir + 'prog_target_info.json'
        self.baseScoreLoc = self.infoDir + 'base_scores.csv'

        self.allInfoLoc = self.infoDir + 'target_info.json'
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.zooMergerLoc = self.infoDir + 'galaxy_zoo_models.txt'

        status = True

        # Check if everything needed is found
        if not path.exists( self.infoDir ):
            if self.printBase: print("IM: WARNING: info directory not found!")
            status = False

        if not path.exists( self.zooMergerDir ):
            if self.printBase: print("IM: WARNING: zoo models directory not found!")
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

    
def tabprint( inprint, begin = '\t - ', end = '\n' ):
    print('%s%s' % (begin,inprint), end=end )


if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main

