'''
    Author:	 Matthew Ogden
    Created:	28 Feb 2020
Description:	I have finally commited to creating a dedicated file for all things relating to the input arguments and processing.
'''

# For loading in Matt's general purpose python libraries
import json
import numpy as np
import cv2
from os import path
from sys import path as sysPath
from mpi4py import MPI

supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
sysPath.append( supportPath )

from pprint import PrettyPrinter
pp = PrettyPrinter(width=41, compact=True)
pprint = pp.pprint

def test():
    print("GM: Hi!  You're in Matthew's module for generally useful functions and classes")

def validPath( inPath, printWarning = False, pathType = None ):

    # Check if valid string
    if type( inPath ) != type( 'string' ):
        if printWarning: print('GM: validPath: Input not a string')
        return None

    # Check if path exists
    if not path.exists( inPath ):
        if printWarning: print('GM: validPath: Input path does not exist')
        return None

    outPath = path.realpath( inPath )

    # Check if directory or something else
    if path.isdir( outPath ) and outPath[-1] != '/':
        outPath += '/'

    return outPath

# End valid path function

# converting between two common image types
def uint8_to_float32( in_img ):
    if in_img.dtype != np.uint8:
        print("WARNING: GM: uint8_to_float32: In image not type np.uint8.")
        return None
    
    f_img = in_img.astype(np.float32)
    f_img /= 255.
    return f_img
# End converting to float image between 0 and 1

def float32_to_uint8( f_img ):
    if f_img.dtype != np.float32:
        print("WARNING: GM: float32_to_uint8: In image not type np.float32.")
        return None
    to_img = np.copy(f_img)
    to_img *= 255.
    to_img = to_img.astype(np.uint8)
    return to_img
# End float32 to uint8

# Read json file
def readJson( jPath ):
    
    # Check for valid path/file
    jPath = validPath( jPath )
    if jPath == None:
        return
    
    # Read json file
    jDict = None
    with open( jPath, 'r' ) as jFile:
        jDict = json.load( jFile )
    return jDict

# Save json file
def saveJson( jDict, jPath, pretty=False ):    
    with open( jPath, 'w' ) as jFile:
        if pretty:
            json.dump( jDict, jFile, indent=4 )
        else:
            json.dump( jDict, jFile )
            
    
def getScores( scoreLoc ):
    
    import pandas as pd
    
    scores = None    
    
    sLoc = validPath( scoreLoc )
    if sLoc != None:
        scores = pd.read_csv(sLoc)
    
    return scores


def saveImg( img, imgLoc, printAll = False ):
    if img.dtype == np.float32:
        img = float32_to_uint8(img)
    
    cv2.imwrite(imgLoc,img)

    if not path.exists( imgLoc ) and printAll:
        print("GM: WARNING: image not saved")
        print("\t - %s: " %imgLoc )

def readFile( fileLoc, stripLine=False ):

    if not path.isfile( fileLoc ):
        print("Error: GM: File does not exist: %s" % fileLoc)
        return None

    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Error: GM: Failed to open/read file at \'%s\'' % fileLoc)
        return None

    else:

        inList = list(inFile)
        inFile.close()

        if stripLine:
            for i in range( len( inList ) ):
                inList[i] = inList[i].strip()

        return inList

# End simple read file into list

class inArgClass:

    def __init__( self, inArg=None, argFile=None ):

        self.printBase = True
        self.printAll = False

        self.simple = False
        self.runDir = None
        self.targetDir = None
        self.dataDir = None

        if inArg != None:
            self.updateArg( inArg )

        elif argFile != None:
            self.readArgFile( argFile )

        if self.printAll:
            self.printBase = True

    def get( self, inVal, default = None ):

        return getattr( self, inVal, default )

    def updateArg( self, inArg, printAll = False ):

        if printAll:
            print("GM: inArgClass.updateArg()")
            print("\t - Before:")
            self.printArg()

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

        self.checkBool()

        if printAll:
            print("GM: inArgClass.updateArg()")
            print("\t - Before:")
            self.printArg()

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
                if   argVal == 'false': argVal = False
                elif argVal == 'False': argVal = False
                elif argVal == 'True': argVal = True
                elif argVal == 'true': argVal = True
                setattr( self, argName, argVal )

            # End if string
        # End looping through arguments
    # End check for booleans


    def printArg(self):

        allAttrs = vars( self )
        print('GM: Printing Input arguments')
        for a in allAttrs:
            print('\t- %s - %s : ' % (a, str(type(getattr(self,a))) ), getattr(self, a ) )
    # End print all arguments

    def getArg( self, arg ):

        return getattr( self, arg, None )

# Global input arguments


class ppClass:

    import multiprocessing as mp
    from queue import Empty
    from time import sleep

    nCores = 1
    printProg = False

    jobQueue = mp.Queue()
    nQueue = 0

    funcPtr = None
    manager = mp.Manager()

    def __init__(self, nCore, printProg=False):
        
        if type(nCore) != type( 123 ):
            nCore = int(nCore)            

        # 0 for all, negative all minus negative
        if nCore < 1:
            self.nCores = self.mp.cpu_count() + nCore
            if self.nCores < 1: self.nCores = 1
        else:
            self.nCores = nCore

        self.printProg = printProg

    def printProgBar(self):
        self.printProg = True

    # Assuming you're only running a single function
    def loadQueue( self, funcIn, inList ):

        self.funcPtr = funcIn
        self.nQueue = len( inList )

        for args in inList:

            self.jobQueue.put(args)

    def runCores( self ):

        if self.nCores == 1:
            print("Why use parallel processing with 1 core?")

        self.coreList = []

        # Start all processes
        for i in range( self.nCores ):
            p = self.mp.Process( target=self.coreFunc, )
            self.coreList.append( p )
            p.start()

        # Wait until all processes are complete
        for p in self.coreList:
            self.coreList.pop()
            p.join()

    # Blah

    def __del__( self, ):
        # check if queue is still full
        pass


    def coreFunc( self ):

        n = int( self.nQueue )
        from sys import stdout

        # Keep core running until shared queue is empty
        while True:

            try:
                funcArgs = self.jobQueue.get( block=True, timeout=1 )

            # Will exist loop if queue is empty
            except self.Empty:
                #print('%s - queue empty' % self.mp.current_process().name)
                break

            if self.printProg:
                p = n - int( self.jobQueue.qsize() )
                perc = ( p / n ) * 100
                
                stdout.write( "%.1f%% - %d / %d	  \r" % ( perc, p, n ) )

            # Run desired function on core
            self.funcPtr(**funcArgs)

    # End exectute function

    def testPrint():
        print("Inside parallel processing python Module.  Written by Matthew Ogden")

# End parallel processing class

def printVal( n1, n2=1 ):
    from time import sleep
    #print("Val: %d %d" % ( n1, n2) )
    sleep(n1)
    


def checkPP(arg):

    import multiprocessing as mp

    print("Hi!  You're in Matt's parallel processing module.")
    print("\t- requested cores: %s" % arg.nProc)
    print("\t- available cores: %d" % mp.cpu_count() )

    nCores = 2
    pHolder = ppClass( nCores )

    pHolder.printProgBar()

    argList = []

    for i in range( 4 ): 
        argList.append( dict( n1=i, n2=i ) )
        argList.append( dict( n1=i ) )

    pHolder.loadQueue( printVal, argList )

    pHolder.runCores()
    

def readImg( imgLoc, printAll = False, toSize=None, toType=np.float32 ):

    # Check if path is valid
    if not path.exists( imgLoc ):
        if printAll:
            print("MC: WARNING: image not found at path.")
            print("\t- %s" % imgLoc)
        return None

    # Read image from disk
    img = cv2.imread( imgLoc, 0 ) 
    
    # Convert to floating point with values between 0 and 1
    if img.dtype != np.float32 and toType == np.float32:
        img = uint8_to_float32(img)

    # Resize if requested
    if toSize != None:
        img = cv2.resize(img, toSize, interpolation = cv2.INTER_AREA)

    # Return image
    return img

# End get image

def saveImg( imgLoc, img ):
    
    # Convert to floating point with values between 0 and 1
    if img.dtype == np.float32:
        img = float32_to_uint8(img)
        
    # Read image from disk
    img = cv2.imwrite( imgLoc, img ) 

# End save image


def tabprint( inprint, begin = '\t - ', end = '\n' ):
    print('%s%s' % (begin,inprint), end=end )

class mpi_class:
    status = "HI from MPI"
    


# Run main after declaring functions
if __name__ == '__main__':

    # For testing main input arguments
    from sys import argv
    arg = inArgClass( argv )
    arg.printArg()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print( 'MPI: ', rank, size )

    checkPP(arg)
    
    
