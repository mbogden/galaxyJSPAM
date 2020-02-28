'''
    Author:     Matthew Ogden
    Created:    28 Feb 2020
Description:    I have finally commited to creating a dedicated file for all things relating to the input arguments and processing.
'''

from sys import ( argv, path as sysPath )

from os import ( path, listdir )

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Suuport_Code/" ) )
sysPath.append( supportPath )
import general_module as gm

def main(argList):

    arg = inArgClass( argList )
    arg.checkBddool()

    if arg.printAll:  arg.printAllArg()

# End main


def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return None
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return None

    else:
        inList = list(inFile)
        inFile.close()
        return inList

# End simple read file


class inArgClass:

    def __init__( self, inArg=None ):
        self.printAll = False
        self.dataDir = None
        self.sdssDir = None
        self.runDir = None
        
        if inArg != None:
            self.updateArg( inArg )

    def updateArg( self, inArg ):

        n = len( inArg )

        # Loop through given arguments
        for i, arg in enumerate( inArg ):

            # Ignore unless handle provided
            if arg[0] != '-':
                continue

            # Grab string of everything except starting handle '-'
            argName = arg[1:]

            # Check if last argument in list
            if i+1 == n:
                argVal = True

            # Check if suplimentary info provided, aka no handle for next arg
            elif inArg[i+1][0] != '-':
                argVal = inArg[i+1]

            # If no supplimentary arg given, assume True
            else:
                argVal = True

            # Save argument handle name and value
            setattr( self, argName, argVal )

    # End update input arguments

    # For manual setting
    def setArg( self, inName, inArg ):
        setattr( self, inName, inArg )

    # For checking if input strings are meant to be a boolean
    def checkBool(self):

        allAttrs = vars( self )

        for argName in allAttrs:
            
            argVal = getattr(self, argName )
            oldType = type( argVal )

            if oldType == str:
                print( 'Old: %s - %s' % (argName,argVal) )
                if   argVal == 'false': argVal = False
                elif argVal == 'False': argVal = False
                elif argVal == 'True': argVal = True
                elif argVal == 'true': argVal = True
                setattr( self, argName, argVal )
                print( 'New: %s -' % argName,argVal )

            # End if string
        # End looping through arguments
    # End check for booleans


    def printAllArg(self):

        allAttrs = vars( self )
        for a in allAttrs:
            print('\t- %s :' % a, getattr(self, a ), type(getattr(self,a)) )



# Run main after declaring functions
if __name__ == '__main__':
    argList = argv
    main(argList)
