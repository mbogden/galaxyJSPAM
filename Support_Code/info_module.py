'''
    Author:     Matthew Ogden
    Created:    21 Feb 2020
Description:    For all things scores related
'''

from os import path, listdir
from sys import exit, argv, path as sysPath
import json

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Suuport_Code/" ) )
sysPath.append( supportPath )
import general_module as gm

def test():
    print("SM: Hi!  You're in Matthew Ogden's Module for score files")
# End test print

class run_score_class: 

    rDict = None
    sDict = None
    status = False

    runHeaders = ( 'run_identifier', 'model_data', 'human_scores', 'model_images', 'perturbedness', 'machine_fitness_scores', )

    modelImgHeaders = ( 'image_name', 'image_parameter_name' )

    pHeaders = ( 'image_name', 'pertrubedness' )

    machineScoreHeaders = ( 'image_parameter_name', 'scores' )

    def __init__( self, scoreLoc=None, infoLoc=None, printAll=False ):

        self.printAll = printAll
        if self.printAll: print("SC: Initailizing run score class")

        # Get data
        if scoreLoc != None:

            self.scoreLoc = scoreLoc

            # Read score data file if it exists
            if path.exists( self.scoreLoc ):

                try: 
                    with open( self.scoreLoc, 'r' ) as sFile:
                        self.rDict = json.load( sFile )
                    if self.printAll: 
                        print("\t- Read score file")
                        print('\t- ', self.rDict )
                    self.status = True

                # Print error if not opened for some reason
                except:
                    print("Error: Could not read score file.  Expecting .json")
                    print("\tscoreLoc: %s" % self.scoreLoc )
                    self.status = False
                    return

            # Create new run json from info file if it does not exist
            if not path.exists( self.scoreLoc ):

                self.info2Dict( infoLoc )

                if type(self.rDict) == type(None):
                    print("Error: Failed to initialize score data file..." )
                    return

                if self.printAll: 
                    print("\t- Initialized run score file")
                    print( self.rDict )

                self.saveScoreData( )

                self.status = True
            # end if not path exists
            
        # End if score loc not given 
        
        else: 
            print("ERROR: SM: Score data without info or score file not implemented")
            return
            
        print("\t- Initalized score file.")

    # end __init__

    def info2Dict( self, infoLoc ):
        
        if self.printAll: print("SC: Initalizing new score data file...")

        # Check if info locational given
        if infoLoc == None:
            print("Error: SC: Please give info file when creating new score file")
            return None

        # Check if info location have a file
        elif not path.exists( infoLoc ):
            print("Error: SC: Info file not found to create new score file")
            print("\t infoLoc: %s" % infoLoc)
            return None

        infoData = gm.readFile( infoLoc )

        # Check if retrieved info
        if infoData == None:
            print("Error: SC: Info file not found to create new score file")
            print("\t infoLoc: %s" % infoLoc)
            return None
        
        # Go through info file and add appropriate information to json
        
        if self.printAll: print("SC: Creating new score json from info file")

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
            print("Error: SC: Needed information not found in info.txt")
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


    # end info2Dict

    def saveScoreData( self ):

        if self.printAll: print("SM: Saving score data file...")

        if self.scoreLoc == None:
            print("ERROR: SM: No score location given to save score file")
            return False

        else: 
            with open( self.scoreLoc, 'w' ) as sFile:
               json.dump( self.rDict, sFile, indent=4 )

        return True
    
    def getModelImgsNames():

        

        modelImgDicts = self.rDict.get( 'model_images', None )

        if modelImgDicts == None:
            if self.printAll: print("WARNING: SM: No model images found in data")
            return None

        for md in modelImgDicts:
            print(md)




class score_class:

    status = True
    scoreLoc = None
    printAll = False

    # Data
    allData = None
    sData = None

    t1Header = ( 'target_identifier', )

    t2Header = ( 'generation_identifier', )

    t3Header = ( 'run_identifier', )

def main(argList):
    print("Hi!  In main python template")

# End main

# Run main after declaring functions and stuff
if __name__ == '__main__':
    arg = gm.inArgClass( argv )
    main( arg )
