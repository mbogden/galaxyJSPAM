'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
    Altered:    02 Apr 2020
Description:    For all things misc information related
'''

from os import path, listdir
from sys import argv
import json
from copy import deepcopy
from pprint import PrettyPrinter

pp = PrettyPrinter( indent = 2 )

# For loading in Matt's general purpose python libraries
import general_module as gm

def test():
    print("IM: Hi!  You're in Matthew Ogden's information module for SPAM")
# End test print


class target_info_class:

    tDict = None    # Primary dict of information contained in target_info.json
    sDict = None    # State of json upon initial reading.
    status = False  # State of this class.  For initiating.
    printAll = False

    # For user to see headers.
    targetHeaders = ( 'target_identifier', 'target_images', 'model_image_parameters', 'runs', 'progress', )

    def __init__( self, printAll = False, targetDataDir = None, targetDict=None ):

        # Tell class if it should print progress
        self.printAll = printAll

        if targetDict != None:
            self.tDict = deepcopy( targetDict )
            if self.printAll: print("IM: Target: copied input dict")

        if targetDataDir != None:
            print("IM: Target:")

        else:
            print("IM: Target: Target_info_class without input not implemented yet!")
            
    # deconstructor
    def __del__( self ):

        if self.tDict != self.sDict:
            print("IM: WARNING: target_info.json updated but not saved!")
    # End deconstructor

    def printMe( self ):
        pp.pprint( self.tDict )

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
        



class run_info_class: 

    rDict = None    # Primary dict of information contained in info.json
    sDict = None    # State of json upon initial reading.
    status = False  # State of this class.  For initiating.

    runHeaders = ( 'run_identifier', 'model_data', 'human_scores', 'model_images', 'misc_images', 'perturbedness', 'machine_scores', )

    modelImgHeaders = ( 'image_name', 'image_parameter_name' )

    pHeaders = ( 'image_name', 'pertrubedness' )

    machineScoreHeaders = ( 'image_parameter_name', 'scores' )

    def __init__( self, printAll=False, rDict=rDict, infoLoc=None, runDir=None ):

        self.printAll = printAll
        if self.printAll: print("IM: Initailizing run score class")

        # Get data
        if infoLoc != None:

            self.infoLoc = infoLoc

            # Read score data file if it exists
            if path.exists( self.infoLoc ):

                try: 
                    with open( self.infoLoc, 'r' ) as sFile:
                        self.rDict = json.load( sFile )
                        self.sDict = deepcopy( self.rDict )
                    if self.printAll: 
                        print("\t- Read score file")
                    self.status = True

                # Print error if not opened for some reason
                except:
                    print("Error: Could not read score file.  Expecting .json")
                    print("\tinfoLoc: %s" % self.infoLoc )
                    self.status = False
                    return

            # Create new run json from info file if it does not exist
            if not path.exists( self.infoLoc ):

                self.txt2Json( runDir )

                if type(self.rDict) == type(None):
                    print("Error: Failed to initialize score data file..." )
                    return

                if self.printAll: 
                    print("\t- Initialized run score file")
                    print( self.rDict )

                self.saveInfoFile( )

                self.status = True
            # end if not path exists

        # End if score loc not given 

        else: 
            print("ERROR: IM: info.json without info file not implemented")
            return
            
        if self.printAll: print("\t- Initalized info module.")

    # end __init__

    # deconstructor
    def __del__( self ):

        if self.rDict != self.sDict:
            print("IM: WARNING: info.json updated but not save!")
    # End deconstructor

    def printMe( self ):
        pp.pprint( self.rDict )

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

        self.sDict = deepcopy( self.rDict )


    # end info2Dict

    def saveInfoFile( self, saveLoc = None ):

        if self.printAll: print("IM: Saving score data file...")

        if self.infoLoc == None and saveLoc == None:
            print("ERROR: IM: No score location given to save score file")
            return False

        if self.rDict == self.sDict:
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
            self.sDict = deepcopy( self.rDict )

        return retVal
    
    def getModelImgNames( self ):

        modelImgDicts = self.rDict.get( 'model_images', None )

        if modelImgDicts == None:
            if self.printAll: print("WARNING: IM: No model images found in data")
            return None

        for md in modelImgDicts:
            print(md)
            pass
    # end getModelImgsNames


def main(argList):
    print("Hi!  In main python template")

# End main

# Run main after declaring functions and stuff
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
