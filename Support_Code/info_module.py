'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    02 Apr 2020
Description:    For all things misc information related
'''

from os import path, listdir
import json
from copy import deepcopy


# For loading in Matt's general purpose python libraries
import general_module as gm

def test():
    print("IM: Hi!  You're in Matthew Ogden's information module for SPAM")
# End test print

def main(arg):
    print("IM: Hi!  You're in Matthew Information module's main function.")
    if arg.printAll:
        arg.printArg()

    if arg.targetDir != None:
        print("IM: Found SDSS dir!: %s" % arg.targetDir )
        procTarget( arg.targetDir, printAll=True )

    elif arg.dataDir != None:
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


# End main

class target_info_class:

    tDict = None      # Primary dictionary of info for all data 
    initDict = None     # Dictionary of info upon initial reading of file
    baseDict = None     # Simple dictionary of info for basic info.

    status = False      # State of this class.  For initiating.
    printAll = False

    targetDir = None
    zModelDir = None
    baseInfoLoc = None
    allInfoLoc = None
    scoreLoc = None
    plotDir = None

    # For user to see headers.
    targetHeaders = ( 'target_identifier', 'target_images', 'model_image_parameters', 'progress',  'model_sets',)

    def __init__( self, targetDir = None, printAll = False, rmInfo=False, ):

        # Tell class if it should print progress
        self.printAll = printAll
        if self.printAll: 
            print("IM: Initailizing target info class")
            print('\t - targetDir: ', targetDir)
            print('\t - rmInfo: ', rmInfo)
            print('\t - printAll: ', printAll)

        # if nothing given, complain
        if targetDir == None:
            print("IM: WARNING: No target Dir given.  No other option at this time.")
            return

        # If path not found, complain
        elif not path.exists( targetDir ):
            print("IM: WARNING: Target dir not found")
            print('\t - targetDir: ', targetDir )
            return

        good = self.initTargetDir( targetDir )
        if not good:
            print("IM: Target: WARNING: Something not right.")
            return
    
        # Remove info file if condition given
        if rmInfo:

            if self.printAll: print("IM: Removing previous info files.")
            from os import remove

            if path.exists( self.allInfoLoc ):
                remove( self.allInfoLoc )

            if path.exists( self.baseInfoLoc ):
                remove( self.baseInfoLoc )

        # If target info does not exist, create
        if not path.exists( self.baseInfoLoc ):
            if self.printAll: print("IM: WARNING: Base info json not found: %s" % self.baseInfoLoc )
            
            self.newTargetInfo()
            self.gatherRunInfos()
            self.saveInfoFile( baseFile=True )
            self.status = True
            return

        # Else, target info file found.
        else:

            if self.printAll: 
                print("IM: Target: Opening target info json")
                print("\t - allInfoLoc: (%s) %s", path.isfile( self.allInfoLoc ), self.allInfoLoc )

            with open( self.allInfoLoc ) as iFile:
                if self.printAll: print("IM: Target: Reading target info json")
                self.tDict = json.load( iFile )

            self.initDict = deepcopy( self.tDict )

            if self.printAll: print("IM: Target: Read target info json")
            self.status = True
            return

        # End reading target file
    # End target init 

    def initTargetDir( self, targetDir ):

        if self.printAll:
            print( 'IM: initTargetDir:' )
            print( '\t - targetDir: %s' % targetDir )

        # Define paths for all useful things in target structure
        self.targetDir = path.abspath( targetDir )
        if self.targetDir[-1] != '/': self.targetDir += '/'

        self.infoDir = self.targetDir + 'information/'
        self.allInfoLoc = self.infoDir + 'all_target_info.json'
        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        self.scoreLoc = self.infoDir + 'scores.csv'

        self.zooModelDir = self.targetDir + 'gen000/'

        self.plotDir = self.targetDir + 'plots/'

        status = True

        # Check if everything needed is found
        if not path.exists( self.infoDir ):
            print("IM: WARNING: info directory not found!")
            status = False

        if not path.exists( self.zooModelDir ):
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
            print( '\t - zooModelDir: (%s) %s' % ( path.exists( self.zooModelDir ), self.zooModelDir ) )
            print( '\t - plotDir: (%s) %s' % ( path.exists( self.plotDir ), self.plotDir ) )

        return status

    # For code I want to make in the future but isn't needed now
    def incomplete( self ):

        # if given pre-existing rDict
        if tDict != None:
            self.tDict = deepcopy( tDict )
            self.initDict = deepcopy( tDict )
            self.status = True

        else:
            print("IM: ERROR: Umm.... You shouldn't be here...")
            self.status = False
            return 

    # Deconstructor
    def __del__( self ):
        if self.tDict != self.initDict:
            print("IM: WARNING: Target: target_info.json updated but not saved!")
    # End deconstructor

    def printInfo( self ):
        from pprint import PrettyPrinter
        pp = PrettyPrinter( indent = 2 )
        pp.pprint( self.tDict )
    # End print info

    def printStatus( self ):

        print("IM: Printing target info status.")
        for h in self.tDict:
            item = self.tDict[h]

            if type( item ) == list:
                print("\tLIST!")
                for i in item:
                    print(i)


            print( '\t', h, len( item ) )

    # Gather human and machine scores into pandas files and save as csv
    def getScores( self, setId = 'galaxy_zoo_mergers', saveScores=True, newScores= False):

        import pandas as pd
        import numpy as np

        if newScores or not path.exists( self.scoreLoc ):
            if self.printAll: print("IM: Gathering scores")
            sFrame = self.gatherScores( setId=setId )
            if self.scoreLoc != None: sFrame.to_csv( self.scoreLoc, index = False )

        if self.printAll: print("IM: Getting scores from csv file")
        sFrame = pd.read_csv( self.scoreLoc )

        return sFrame


    def gatherScores( self, setId = None ):

        if self.printAll: print("IM: Gathering Scores")

        import pandas as pd
        import numpy as np

        setList = self.tDict[ 'model_sets' ]
        desiredSet = None

        # Find model set with name of setID
        for s in setList:
            setName = s.get( 'set_identifier', None)
            if setName == None: 
                print("IM: WARNING: A target model set without identifier found")
                continue

            if setName == setId:
                desiredSet = s
                break

        if desiredSet == None:
            print("IM: WARNING: Model set not found: ", setId)
            return None

        # Create Frame and populate scores
        models = desiredSet['models']
        if self.printAll: print("IM: getScores: \n\t - %d models" % len(models) )

        if len(models) == 0:
            if self.printAll: print("\t - WARNING: No scores found.")
            return None


        # Calculate DataDrame size based on first mode1
        m0 = models[0]
        m0_id = m0['run_identifier']
        m0_h = m0['human_scores']
        m0_m = m0['machine_scores']
        m0_p = m0['perturbedness']

        print( m0) 

        # run ID
        headerNames = [ 'run_id' ]

        # human Scores
        for h in m0_h:
            hN = h.get( 'score_name', None )
            headerNames.append( 'human_' + hN )

        # Machine Scores
        for m in m0_m:
            cN = m.get( 'comparison_name', None )
            headerNames.append( 'machine_' + cN )

        # Perturbedness Scores
        for p in m0_p:
            cN = p.get( 'comparison_name', None )
            print(cN)
            headerNames.append( 'perturbation_' + cN )

        if self.printAll: 
            print( '\t - %d headers' % len(headerNames) )
            print( '\t -', headerNames )


        # Create Empty dataframe and populate
        sFrame = pd.DataFrame( index = np.arange( len( models ) ), columns=headerNames)

        for i, m in enumerate(models):

            for h in headerNames:

                if h == 'run_id':
                    sFrame.loc[ i, h ] = m[ 'run_identifier' ]

                if 'galaxy_zoo' in h:
                    sFrame.loc[ i, h ] = m[ 'human_scores' ][0]['score']

                if 'machine_' in h:
                    for ms in m['machine_scores']:
                        if ms['comparison_name'] in h:
                            sFrame.loc[ i, h] =  ms['score'] 
                            break

                if 'perturbation_' in h:
                    for ps in m['perturbedness']:
                        if ps['comparison_name'] in h:
                            sFrame.loc[ i, h] = ps['score']
                            break 

            break

        # End going through models
        
        return sFrame

    # Create blank dictionary
    def createBlank( self, ):

        if self.printAll: print("IM: Target: Creating blank target_info_class")
        
        for h in targetHeaders:
            self.tDict[h] = None

        self.initDict = deepcopy( self.rDict )

        if self.printAll: print('IM: Target: info:\n' % self.rDict)
    # End creating blank target info


    # For appending new information to a list in target info module
    def appendList( self, keyName, aList ):

        if self.printAll: print("IM: Extending %s list" % keyName)

        if type( aList ) != type( [] ):
            print("ERROR: IM: Given non-list in appendList: %s" % type(aList))
            return None

        keyGood = keyName in self.tDict

        if not keyGood: 
            print("IM: WARNING: key %s does not exist in info module")
            return None

        # Check if key is list
        if type( self.tDict[keyName] ) != type( [] ):
            print("ERROR: IM: Key value not a list in appendList: %s" )
            print("\t- key: %s" % keyName)
            print("\t- Type: %s" % type( self.tDict[keyName] ) )
            return None

        # All is good. 
        self.tDict[keyName].extend( aList ) 
        return self.tDict[keyName]

    # end append list

    def newTargetInfo( self, ):

        if self.printAll: 
            print("IM: newtargetInfo: Creating new target info file")

        self.tDict = {}

        for key in self.targetHeaders:
            self.tDict[key] = None

        # Find target/sdss name from progress file...
        progLoc = self.infoDir + 'progress.txt'
        pFile = gm.readFile( progLoc )
        tId = pFile[0].split()[-1]
        self.tDict['target_identifier'] = tId
        if self.printAll: print('\t- target_id: %s' % self.tDict['target_identifier'])

        # Search for files
        iFiles = listdir( self.infoDir )
        tImgs = []
        pFiles = []

        for fName in iFiles:

            # if target image
            if 'target' in fName and '.png' in fName:
                tImgs.append( fName )

            # if image parameter
            if 'param' in fName and '.txt' in fName and not 'parameters.txt' in fName:
                pFiles.append( fName )

        # Save target images
        self.tDict['target_images'] = tImgs
        if self.printAll: print('\t- target images: ', self.tDict['target_images'])

        # Save image parameters
        self.tDict['model_image_parameters'] = pFiles
        if self.printAll: print('\t- image parameters: ', self.tDict['model_image_parameters'])

        # Make progress note
        self.tDict['progress'] = { 'Initial' : 'Initial creation of information file' } 

        # Add orginal runs from galaxy zoo
        self.tDict['model_sets'] = [ { 'set_identifier' : 'galaxy_zoo_mergers', 'models' : None } ]

    # End new target info dictionary


    def gatherRunInfos( self, ):

        self.gDir = self.targetDir + 'gen000/'

        self.runDirs = listdir( self.gDir )
        self.runDirs.sort()

        nRuns = len( self.runDirs )

        rInfoList = []

        for i,run in enumerate(self.runDirs):
            rInfoLoc = self.gDir + run + '/info.json'

            if not path.exists( rInfoLoc ):
                #print("IM: WARNING: No info file found")
                continue
            
            with open( rInfoLoc) as rInfo:
                rInfoList.append( json.load( rInfo ) )

            if self.printAll: print("\t- %d/%d" % ( i, nRuns ), end='\r' )
        
        self.tDict['model_sets'][0]['models'] = rInfoList
        if self.printAll: print( '\n\t- Gathered %d run info files' % len( self.tDict['model_sets'][0]['models'] ) )


    def saveInfoFile( self, saveLoc = None, baseFile=False ):

        if self.printAll: print("IM: Saving target info file...")

        if self.allInfoLoc == None and saveLoc == None:
            print("ERROR: IM: No target info location given...")
            return False

        if baseFile != False and self.baseInfoLoc == None:
            print("ERROR: IM: No base target info location given...")
            return False

        if self.tDict == self.initDict:
            if self.printAll: print("IM: No changes detected, not saving")
            return True

        retVal = False

        with open( self.allInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
            retVal = True

        if retVal:
            self.initDict = deepcopy( self.tDict )

        if baseFile:
            if self.printAll: print("IM: Target: Saving base info file")

            #targetHeaders = ( 'target_identifier', 'target_images', 'model_image_parameters', 'progress',  'model_sets',)
            self.baseDict = {}
            self.baseDict['target_identifier'] = self.tDict['target_identifier']
            self.baseDict['target_images'] = self.tDict['target_images']
            self.baseDict['model_image_parameters'] = self.tDict['model_image_parameters']
            self.baseDict['progress'] = self.tDict['progress']
            self.baseDict['model_sets'] = None

            with open( self.baseInfoLoc, 'w' ) as infoFile:
                json.dump( self.baseDict, infoFile )
                if self.printAll: print("\t - Saved base info file")

        return retVal
    

class run_info_class: 

    rDict = None    # Primary dict of information contained in info.json
    initDict = None    # State of json upon initial reading.
    status = False  # State of this class.  For initiating.



    runHeaders = ( 'run_identifier', 'model_data', 'human_scores', 
            'model_images', 'misc_images', 'perturbedness', 'machine_scores', )

    modelImgHeaders = ( 'image_name', 'image_parameter_name' )

    pHeaders = ( 'image_name', 'pertrubedness' )

    machineScoreHeaders = ( 'image_name', 'scores' )

    def __init__( self, printAll=False, rmInfo=False,
            runDir=None, infoLoc=None, rDict=None, ):

        # print settings
        self.printAll = printAll
        if self.printAll: print("IM: Initailizing run score class")
        
        # if nothing given, create blank rDict
        if runDir == None and infoLoc == None and rDict == None:
            self.createBlank()
            self.status = True
            return

        # if given pre-existing rDict
        elif rDict != None:
            self.rDict = deepcopy( rDict )
            self.initDict = deepcopy( rDict )
            self.status = True
            return

        # Else get data from file
        elif runDir != None or infoLoc != None:

            if runDir != None:
                if runDir[-1] != '/': runDir += '/'
                self.runDir = runDir
                self.infoLoc = runDir + 'info.json'

            elif infoLoc != None:
                self.infoLoc = infoLoc

            # Remove info file if condition given
            if rmInfo and path.exists( self.infoLoc ):
                from os import remove
                remove( self.infoLoc )

            # Create new run json from info file if it does not exist
            if not path.exists( self.infoLoc ):

                self.txt2Json( runDir )

                if type(self.rDict) == type(None):
                    print("Error: IM: run: Failed to initialize info file..." )

                if self.printAll: 
                    print("\t- Initialized run score file")
                    print( self.rDict )

                self.saveInfoFile( )

                self.status = True
                return
            # End if not path exists

            # Read run info file
            else:
                
                if self.printAll: print("IM: run: Reading score file: %s" % self.infoLoc)

                try: 
                    with open( self.infoLoc, 'r' ) as iFile:
                        self.rDict = json.load( iFile )
                        self.initDict = deepcopy( self.rDict )

                    if self.printAll: print("IM: run: Read score file:")
                    self.status = True

                # Print error if not opened for some reason
                except:
                    print("Error: Could not read score file.  Expecting .json")
                    print("\tinfoLoc: %s" % self.infoLoc )
                    self.status = False
                    return

            # end read info file

        if self.printAll: print("\t- Initalized info module.")

    # end __init__

    # deconstructor
    def __del__( self ):

        if self.rDict != self.initDict:
            print("IM: WARNING: info.json updated but not save!")
    # End deconstructor

    def printInfo( self ):
        from pprint import PrettyPrinter
        pp = PrettyPrinter( indent = 2 )
        pp.pprint( self.rDict )

    # Create blank dictionary
    def createBlank( self, ):

        if self.printAll: print("IM: run: Creating blank run_info_class")
        
        for h in runHeaders:
            self.rDict[h] = None

        self.initDict = deepcopy( self.rDict )

        if self.printAll: print('IM: run_info:\n' % self.rDict)


    # For appending new information to a list in run info module
    def appendList( self, keyName, aList ):

        if self.printAll: print("IM: Extending %s list" % keyName)

        if type( aList ) != type( [] ):
            print("ERROR: IM: Given non-list in appendList: %s" % type(aList))
            return None

        keyGood = keyName in self.rDict

        if not keyGood: 
            print("IM: WARNING: key %s does not exist in info module")
            return None

        # Check if key is list
        if type( self.rDict[keyName] ) != type( [] ):
            print("ERROR: IM: Key value not a list in appendList: %s" )
            print("\t- key: %s" % keyName)
            print("\t- Type: %s" % type( self.rDict[keyName] ) )
            return None

        # All is good. 
        self.rDict[keyName].extend( aList ) 
        return self.rDict[keyName]
        

    def txt2Json( self, runDir ):
        
        if self.printAll: print("IM: Initalizing new info.json file...")

        # Check if info locational given
        if runDir == None:
            print("Error: IM: Please give runDir when creating new info.json file")
            return None
        
        if runDir[-1] != '/':
            runDir += '/'

        infoLoc = runDir + 'info.txt'

        # Check if info location have a file
        if not path.exists( infoLoc ):
            print("Error: IM: info.txt file not found to create new info.json file")
            print("\t infoLoc: %s" % infoLoc)
            return None

        infoData = gm.readFile( infoLoc )

        # Check if retrieved info
        if infoData == None:
            print("Error: IM: Info.txt file not found to create new info.json file")
            print("\t infoLoc: %s" % infoLoc)
            return None
        
        # Go through info file and add appropriate information to json
        
        if self.printAll: print("\t- Reading info.txt")

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
            print("\t infoLoc: %s" % infoLoc)
            return None
       
        # build lower structures first

        self.rDict = {
                'run_identifier' : rNum,
                'model_data' : mData,
                'human_scores' : [
                    {
                        'score' : hScore,
                        'score_name' : 'galaxy_zoo',
                        'score_info' : 'wins/total: %s' % wins
                    } ,
                    
                ] ,

            }

        # created initial info.json file from info.txt

        # grab model images

        imgDir = runDir + 'model_images/'

        imgNames = [ img for img in listdir( imgDir ) if '.png' in img ]

        mImgList = []
        iImgList = []

        for name in imgNames:
            param = name.split('_')[0]

            iDict = { 
                    'image_name' : name,
                    'image_parameter_name' : param
                    }

            if 'model' in name:
                mImgList.append( iDict )

            elif 'init' in name:
                iImgList.append( iDict )

            else:
                print("WARNING: IM: Found unusual image in 'model_images'\n\t- %s" % name )

        # done creating img dicts

        self.rDict['model_images'] = mImgList
        self.rDict['misc_images'] = iImgList

        # Create blank arrays for perturbedness and fitness scores

        self.rDict['perturbedness'] = []
        self.rDict['machine_scores'] = []

        self.initDict = deepcopy( self.rDict )


    # end info2Dict

    def saveInfoFile( self, saveLoc = None ):

        if self.printAll: print("IM: Run: Saving info data file...")

        if self.infoLoc == None and saveLoc == None:
            print("ERROR: IM: Run: No score location given to save score file")
            return False

        if self.rDict == self.initDict:
            if self.printAll: print("No changes detected, not saving")
            return True

        retVal = False

        if self.infoLoc != None:
            with open( self.infoLoc, 'w' ) as infoFile:
                json.dump( self.rDict, infoFile, indent=4 )
                retVal = True

        if saveLoc != None:
            with open( saveLoc, 'w' ) as infoFile:
               json.dump( self.rDict, infoFile, indent=4 )
               retVal = True

        if retVal:
            self.initDict = deepcopy( self.rDict )

        return retVal

# End Run info class

def procTarget( tDir, printAll=False, gatherInfo=True ):

    if not path.exists( tDir ):
        print("IM: WARNING:\n\t- sdss directory not found: %s" % tDir)
        return False

    elif printAll:
        print("IM: Found targetDir.")

    iDir = tDir + 'information/'
    gDir = tDir + 'gen000/'
    pDir = tDir + 'plots/'

    if not path.exists( iDir ) or not path.exists( gDir ):
        print("IM: WARNING: Somethings wrong with sdss dir")
        print('\t- info Dir: %s' % iDir )
        print('\t- gen0 Dir: %s' % gDir )
        return False

    infoLoc = iDir + 'target_info.json'
    
    tInfo = target_info_class( targetDir = tDir, printAll = printAll )

    runDirs = listdir( gDir )
    runDirs.sort()

    for run in runDirs:
        rDir = gDir + run + '/'
        procRun( rDir, gatherInfo=True )

# End processing sdss dir


def procRun( rDir, printAll=False, gatherInfo=True ):
    
    if not path.exists( rDir ):
        print("IM: WARNING:\n\t- run directory not found: %s" % rDir)
        return False


    modelDir = rDir + 'model_images/'
    miscDir = rDir + 'misc_images/'
    ptsDir = rDir + 'particle_files/'
    infoLoc = rDir + 'info.json'

    if not path.exists( modelDir ) or not path.exists( ptsDir ) or not path.exists( infoLoc):
        print("Somethings wrong with run dir")



if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main
