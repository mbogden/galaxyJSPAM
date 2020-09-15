'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    02 Apr 2020
Description:    For all things misc information related
'''

from os import path, listdir
import json
from copy import deepcopy


from pprint import PrettyPrinter
pp = PrettyPrinter( indent = 2 )
pprint = pp.pprint

# For loading in Matt's general purpose python libraries
import general_module as gm


# Troubleshooting global variable
rmAll = True

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
                rmInfo = getattr( arg, 'rmInfo', None ), \
                newRun = getattr( arg, 'newRun', False ), \
            )

        if rInfo.status == False:
            return

        rInfo.printInfo()


    if arg.targetDir != None:

        tInfo = target_info_class( \
                targetDir = arg.targetDir, \
                printAll = arg.printAll, \
                rmInfo = getattr( arg, 'rmInfo', False ), \
                gatherRuns = getattr( arg, 'gatherRuns', False ), \
                gatherScores = getattr( arg, 'gatherScores', False ), \
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
        sp = pipeline_parameter_class( paramLoc = arg.param, printBase = arg.printBase, printAll = arg.printAll ) 

# End main


class pipeline_parameter_class:

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
                    'name' : 'correlation',
                },
        }

    def __init__( self, paramLoc = None, new=False, printBase = True, printAll = False ):

        self.printBase = printBase
        self.printAll = printAll
        if self.printAll: self.printBase == True

        if self.printBase:
            print("IM: pipeline_param_class.__init__")
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



    def readParam( self, paramLoc ):
        
        if self.printAll:
            print("IM: pipeline_param_class.readParam")
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

# End pipeline parameter class
        


class target_info_class:

    tDict = None        # Primary dictionary of info for all data 
    initDict = {}     # Copy of dictionary when read
    baseDict = None     # Simple dictionary of info for basic info.

    status = False      # State of this class.  For initiating.
    printAll = False

    targetDir = None
    zModelDir = None
    baseInfoLoc = None
    allInfoLoc = None
    scoreLoc = None
    zModelLoc = None
    plotDir = None

    # For user to see headers.
    targetHeaders = ( 'target_identifier', 'target_images', 'model_image_parameters', 'progress',  'model_sets',)

    progHeaders = ( 'initial_creation', 'galaxy_zoo_models', '100k_particle_files', 'zoo_default_model_images', 'machine_scores' )

    def __init__( self, targetDir = None, printBase = True, printAll = False, rmInfo=False, \
            gatherRuns=False, gatherScores = False ):

        # Tell class what to print
        self.printBase = printBase
        self.printAll = printAll
        if self.printAll:  self.printBase = True

        if self.printAll: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', targetDir)
            print('\t - printBase: ', printBase)
            print('\t - printAll: ', printAll)
            print('\t - rmInfo: ', rmInfo)
            print('\t - gatherRuns: ', gatherRuns)
            print('\t - gatherScores: ', gatherScores)

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
        if rmInfo:

            if self.printAll: print("IM: Target.__init__. Removing previous info file.")
            from os import remove

            if path.exists( self.allInfoLoc ):
                remove( self.allInfoLoc )

            if rmAll:
                if path.exists( self.baseInfoLoc ):
                    remove( self.baseInfoLoc )

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

        if gatherRuns:
            self.gatherRunInfos()
            self.saveInfoFile( )

        if gatherScores:
            self.getScores( newScores=True )
            self.updateProgress()
            self.saveInfoFile( )

        self.status = True
        return


    # End target init 

    def printProg( self, ):

        print( 'IM: Target: printing target progression.\n' )
        pprint( self.tDict['progress'] )
        print( '\nIM: Target: Done printing target progression.\n' )

    # End print info


    def updateProgress( self, model_set='galaxy_zoo_mergers', ):

        if self.printAll: print("IM: updateProgress")

        if self.tDict['progress']['initial_creation'] == True:
            pass
        else:
            print("IM: ERROR: You shouldn't be seeing this...")
            return None

        # If quantity exists, Good!
        if type( self.tDict['progress']['galaxy_zoo_models'] ) == type( int(1) ):
            pass

        # If path exists but no quantity, find it!
        elif path.exists( self.zModelLoc ):

            mFile = gm.readFile( self.zModelLoc, stripLine=True )

            if mFile != None:
                c = 0
                for l in mFile:
                    sN = len( l.split('\t') )
                    if sN == 2: c+= 1
                    else: break

                # If 1 or more models found
                if c != 0: 
                    self.tDict['progress']['galaxy_zoo_models'] = c
                else:
                    self.tDict['progress']['galaxy_zoo_models'] = None

            # Else readFile returns None
            else:
                print("IM: updateProgress: Error reading galaxy zoo model file.")
                self.tDict['progress']['galaxy_zoo_models'] = path.exists( self.zModelLoc )

        # If you can't count the quantity of galaxy zoo models, then just say file exists
        else:
            self.tDict['progress']['galaxy_zoo_models'] = path.exists( self.zModelLoc )

        print("IM: Target.updateProgress: Function still in progress!")

        # Check number of models
        self.tDict['progress']['100k_particle_files'] = None
        self.tDict['progress']['model_images'] = None
        self.tDict['progress']['machine_scores'] = None

        pass
    # End update proggress

    def loopRuns( self, model_set='galaxy_zoo_mergers', checkFunc = None ):

        if self.printAll: 
            print( 'IM: loopRuns' )
            print( '\t- model_set: %s' % model_set)
            print( '\t- checkFunc: ', checkFunc )

        if model_set != 'galaxy_zoo_mergers':
            print("IM: WARNING: Target.loopRuns() not implemented for non galaxy zoo mergers")
            return None

        # If nothing given, return error
        if type(checkFunc) == type(None):
            print("IM: WARNING: Please give something to do when looping through runs.")
            return None

        runList = self.zModelDir
        
        # If given list
        #if type( checkFunc ) == type( [] ): 


    def initTargetDir( self, targetDir ):

        if self.printAll:
            print( 'IM: Target.initTargetDir():' )
            print( '\t - targetDir: %s' % targetDir )

        # Define paths for all useful things in target structure
        self.targetDir = path.abspath( targetDir )
        if self.targetDir[-1] != '/': self.targetDir += '/'

        self.infoDir = self.targetDir + 'information/'
        self.allInfoLoc = self.infoDir + 'all_target_info.json'
        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.zModelLoc = self.infoDir + 'galaxy_zoo_models.txt'

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

        '''
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
        '''

    # End incomplete
    
    
    # Deconstructor
    def __del__( self ):
        if self.tDict != self.initDict:
            print("IM: WARNING: Target: target_info.json updated but not saved!")
    # End deconstructor

    def printInfo( self ):
        pprint( self.tDict )
    # End print info

    def printStatus( self ):

        print("IM: Printing target info status.")
        for h in self.tDict:
            item = self.tDict[h]

            if type( item ) == type([]):
                print("\tLIST!")
                for i in item:
                    print(i)


            print( '\t', h, len( item ) )

    # Gather human and machine scores into pandas files and save as csv
    def getScores( self, setId = 'galaxy_zoo_mergers', saveScores=True, newScores= False):

        if self.printAll:
            print("IM: Target.getScores():")
            print("\t - setId: %s" % setId)
            print("\t - saveScores: %s" % saveScores)
            print("\t - newScores: %s" % newScores)

        import pandas as pd
        import numpy as np

        if newScores or not path.exists( self.scoreLoc ):
            
            if self.printAll: 
                print("IM: Gathering scores")
            sFrame = self.gatherScores( setId=setId )
               
            pass

                
        if self.printAll: 
            print("IM: Getting scores from csv file")
            
        sFrame = pd.read_csv( self.scoreLoc )

        return sFrame


    def gatherScores( self, setId = None ):

        if self.printAll: print("IM: tareget.gatherScores():")

        import pandas as pd
        import numpy as np

        desiredSet = self.tDict[ 'model_sets' ][setId]

        # Create Frame and populate scores
        mKeys = list(desiredSet.keys())
        if self.printAll: print("IM: getScores: \n\t - %d models" % len(mKeys) )

        if len(mKeys) == 0:
            if self.printAll: print("\t - WARNING: No scores found.")
            return None


        # Calculate DataDrame size based on first mode1

        m0 = desiredSet[ mKeys[0] ]

        m0_id = m0['run_identifier']
        m0_h = m0['human_scores']
        m0_m = m0['machine_scores']
        m0_p = m0['perturbation']
        m0_i = m0['initial_bias']

        # Check if model has populated values
        if len( m0_m ) == 0 or  len( m0_p ) == 0 or len( m0_i ) == 0:
            print('\t - WARNING: model 0 has no scores')
            print('\t - Exiting gatherScores')
            return None

        headerNames = []

        # human Scores
        for hN in m0_h:
            headerNames.append( 'human_' + hN )

        # Machine Scores
        for m in m0_m:
            cN = m.get( 'comparison_name', None )
            headerNames.append( 'machine_' + cN )

        # Perturbedness Scores
        for p in m0_p:
            cN = p.get( 'comparison_name', None )
            headerNames.append( 'perturbation_' + cN )

        # Initial bias Scores
        for i in m0_i:
            cN = i.get( 'comparison_name', None )
            headerNames.append( 'initialBias_' + cN )

        if self.printAll: 
            print( '\t - %d headers' % len(headerNames) )
            print( '\t -', headerNames )


        # Create Empty dataframe and populate
        sFrame = pd.DataFrame( index = np.arange( len( mKeys ) ), columns=headerNames)

        nModels = len( mKeys )

        for i, k in enumerate(mKeys):

            print('%d / %d     ' % ( i, nModels ), end='\r' )

            
            m = desiredSet[k]

            for h in headerNames:

                if h == 'run_id':
                    sFrame.loc[ i, h ] = m[ 'run_identifier' ]

                if 'galaxy_zoo' in h:
                    sFrame.loc[ i, h ] = m[ 'human_scores' ]['galaxy_zoo_mergers']['score']

                if 'machine_' in h:
                    for ms in m['machine_scores']:
                        if ms['comparison_name'] in h:
                            sFrame.loc[ i, h] =  ms['score'] 
                            break

                if 'perturbation_' in h:
                    for ps in m['perturbation']:
                        if ps['comparison_name'] in h:
                            sFrame.loc[ i, h] = ps['score']
                            break 

                if 'initialBias_' in h:
                    for ps in m['initial_bias']:
                        if ps['comparison_name'] in h:
                            sFrame.loc[ i, h] = ps['score']
                            break 


        if self.scoreLoc != None: 
            sFrame.to_csv( self.scoreLoc, index = False )


        # End going through models
        
        return sFrame

    # Create blank dictionary
    def createBlank( self, ):

        if self.printAll: print("IM: Target: Creating blank target_info_class")
        
        for h in targetHeaders:
            self.tDict[h] = None

        self.initDict = deepcopy( self.tDict )

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
            print("\t - key: %s" % keyName)
            print("\t - Type: %s" % type( self.tDict[keyName] ) )
            return None

        # All is good. 
        self.tDict[keyName].extend( aList ) 
        return self.tDict[keyName]

    # end append list

    def newTargetInfo( self, ):

        if self.printAll: 
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


    def gatherRunInfos( self, model_key = 'galaxy_zoo_models' ):

        if self.printAll: print( "IM: Target.gatherRunInfos." )

        self.runDirs = listdir( self.zooModelDir )
        self.runDirs.sort()
        nRuns = len( self.runDirs )

        rInfoList = []

        for i,run in enumerate(self.runDirs):

            rInfo = run_info_class( runDir = self.zooModelDir + run, printBase=False)
            rInfo.updateInfo()
            rId = rInfo.rDict['run_identifier']
            self.tDict['model_sets']['galaxy_zoo_mergers'][rId] = rInfo.rDict

            if self.printBase: print("\t- %d/%d" % ( i, nRuns ), end='\r' )
        
        if self.printAll: print( '\n\t- Gathered %d run info files' % len( self.tDict['model_sets']['galaxy_zoo_mergers'] ) )


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
            #return True

        retVal = False

        with open( self.allInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
            retVal = True

        if retVal:
            self.initDict = deepcopy( self.tDict )

        if baseFile:

            if self.printAll: print("IM: Target: Saving base info file")

            self.baseDict = {}
            self.baseDict['target_identifier'] = self.tDict['target_identifier']
            self.baseDict['target_images'] = self.tDict['target_images']
            self.baseDict['model_image_parameters'] = self.tDict['model_image_parameters']
            self.baseDict['progress'] = self.tDict['progress']
            self.baseDict['model_sets'] = { 'galaxy_zoo_mergers' : {} }

            with open( self.baseInfoLoc, 'w' ) as infoFile:
                json.dump( self.baseDict, infoFile, indent=4 )
                if self.printAll: print("\t - Saved base info file")

        return retVal
    

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


    runHeaders = ( 'run_identifier', 'model_data', 'human_scores',  \
            'particle_files', 'model_images', 'misc_images', \
            'perturbation', 'initial_bias', 'machine_scores',)

    # To-Do
    baseHeaders = ( 'run_identifier', 'model_data', 'human_scores',  \
            'particle_files', 'model_images', 'misc_images', )

    modelImgHeaders = ( 'image_name', 'image_parameter_name' )

    pHeaders = ( 'image_name', 'pertrubedness' )

    machineScoreHeaders = ( 'image_name', 'scores' )


    def __init__( self, runDir=None, printBase=True, printAll=False, rmInfo=False, newRun=False
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
            print("\t - rmInfo: ", rmInfo )
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
        if rmInfo:

            if self.printAll: print('\t- Removing Info file.')
            from os import remove

            if path.exists( self.infoLoc ):
                remove( self.infoLoc )

            if rmAll and path.exists( self.baseLoc ):
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
            self.updateInfo()

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

    def getScore( self, sName,  ):

        score = self.rDict['machine_scores'].get( sName, None )
        
        return score


    def findImgFile( self, pName, initImg = True ):

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

        else: 
            if self.baseDict == None:
                self.createBase()
            
            pprint( self.baseDict )

    # Create dict of basic info for printing
    def createBase( self, ):

        self.baseDict = self.createBlank()
        for h in self.baseHeaders:
            self.baseDict[h] = self.rDict[h]
    # End create Base

    def addScore( self, name=None, score=None ):

        if name == None or score == None:
            if self.printBase:
                print("IM: WARNING: run.addScore. name or score not given")
                print('\t - name: ', name)
                print('\t - score: ', score)
            return None

        self.rDict['machine_scores'][name] = score

        return self.rDict['machine_scores'][name]

    # For appending new information to a list in run info module
    def appendScores( self, keyName, addList ):

        if self.printAll: print("IM: Extending list: '%s'" % keyName)

        if type( addList ) != type( [] ):
            print("ERROR: IM: Given non-list in appendList: %s" % type(addList))
            return None

        if keyName in self.rDict and keyName in ['machine_scores', 'perturbation', 'initial_bias']:
            keyGood = True
        else:
            keyGood = False

        if not keyGood: 
            print("IM: WARNING: key %s does not exist in info module")
            return None

        # Check if key is list
        if type( self.rDict[keyName] ) != type( [] ):
            self.rDict[keyName] = []
            '''
            print("ERROR: IM: Key value not a list in appendList: %s" )
            print("\t - key: %s" % keyName)
            print("\t - Type: %s" % type( self.rDict[keyName] ) )
            '''
            return None

        # All is good. 
        self.rDict[keyName].extend( addList ) 
        return self.rDict[keyName]
    
    def createBlank( self, ):
        
        # Create blank dictionary
        tempDict = {}
        for key in self.runHeaders:
            tempDict[key] = {}

        tempDict['initial_bias'] = {}
        tempDict['perturbation'] = {}
        tempDict['machine_scores'] = {}
        
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
        self.rDict['human_scores'] = { 
                    'galaxy_zoo_mergers' : {
                            'score' : hScore,
                            'score_info' : 'wins/total: %s' % wins
                        } 
                    }

        # created initial info.json file from info.txt

        # Save results
        self.saveInfoFile( saveBase = True )

    # End txt2Json

    def updateInfo( self, ):

        # If init images found in model_images, move to misc
        from os import system

        # grab model images
        for name in listdir( self.imgDir ):

            if type( self.rDict['model_images'].get( name, None ) ) == type( None ):

                if 'model' in name:
                    param = name.split('_')[0]
                    self.rDict['model_images'][name] = { 'image_parameter_name' : param } 

                elif 'init' in name:
                    mvCmd = "mv %s %s" % (self.imgDir + name, self.miscDir + name)
                    system(mvCmd)

                else:
                    print("IM: Run.updateInfo: WARNING: Found unusual image in 'model_images")
                    print("\t- %s" % name )


        # Grab misc images
        for name in listdir( self.miscDir):

            # If initial image
            if 'init' in name:
                param = name.split('_')[0]
                self.rDict['misc_images'][name] = { 'image_parameter_name' : param } 

            # Else any other image
            else:
                self.rDict['misc_images'][name] = name

        # Grab particle files
        for name in listdir( self.ptsDir):

            # If initial image
            if 'pts.zip' in name:
                nPts = name.split('_')[0]
                self.rDict['particle_files'][nPts] = nPts 

            # Else any other file?
            else:
                print("IM: Run.updateInfo: WARNING!")
                print("\t - You shouldn't be seeing me!")
                print("\t - runDir: %s" % self.runDir)
                print("\t - ptsFile: %s" % ('particle_files/' + name ))


        # Done looping through images. 

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

