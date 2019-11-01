'''
    Author:     Matthew Ogden
    Created:    27 Oct 2019
    Altered:    
Description:    This program is designed to run image_creator_pl for an entire sdss Galaxy pipeline version 1.
'''

from sys import \
        exit, \
        argv, \
        path as sysPath

from os import \
        path, \
        listdir, \
        system

import multiprocessing as mp
import image_creator_pl_v1 as imgCreator

# For loading in Matt's general purpose python libraries

print( path.abspath( path.join( __file__ , "../../../Useful_Bin/" ) ) )
sysPath.append( path.abspath( path.join( __file__ , "../../../Useful_Bin/" ) ) )
import ppModule as pm

printAll = True
nProc = 1

sdssDir = ''

def sdssImageCreator(argList):

    endEarly = readArg(argList)

    if printAll:
        print("sdssDir: %s" % sdssDir)

    if endEarly:
        print('Exiting...')
        exit(-1)

    sdssFolders = listdir( sdssDir )

    paramFolder = sdssDir + 'sdssParameters/'

    imgParam = readSdssParams( paramFolder )
    imgParam.printVal()

    genFolder = sdssDir + 'gen000/'

    runDirList = listdir( genFolder )
    runDirList.sort()

    argList = []


    for rDir in runDirList:
        runDir = genFolder + rDir + '/'

        if not path.exists( runDir ):
            print("runDir doesn't exist: %s" % runDir)
            continue

        if 'run' not in runDir:
            print("run not in dir: %s" % runDir)
            continue
        
        toArg = [ '-runDir', runDir ]

        if nProc == 1:
            imgCreator.image_creator_pl_v1( toArg )
        else:
            argList.append( ( toArg ) )

    if nProc > 1:
        print("About to start parallel")
        print('len: %d' % len( argList) )

        imgCreator.printAll = False
        imgCreator.paramGiven = True
        imgCreator.wrapperImgParam = imgParam

        pClass = pm.ppClass(nProc)
        pClass.printProgBar()
        pClass.loadQueue( imgCreator.image_creator_pl_v1, argList )
        pClass.runCores()


# End sdss_dir

def readSdssParams( paramFolder ):

    dirContents = listdir( paramFolder)
    sdssParams = ''

    imgParam = imgCreator.imageParameterClass_v3(paramFolder + 'param_v3_default.txt')

    return imgParam

 
def readArg(argList):
    global printAll, sdssDir, nProc
    endEarly = False


    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]
            if sdssDir[-1] != '/':
                sdssDir += '/'
 
        elif arg == '-pp':
            parProc = True
            nProc = argList[i+1]

    # Check if input arguments were valid

    if not path.exists(sdssDir):
        print('\tsdssDir not found: \'%s\'' % sdssDir)
        endEarly = True

    try:
        nProc = int( nProc )

    except:
        print("# of cores given not an int: %s" % nProc )
        endEarly = True

    else:
        max_CPU_cores = int(mp.cpu_count())

        if nProc > max_CPU_cores:
            print('WARNING:  Number of cores requested is greater than the number of cores available.')
            print('Requested: %d' % nProc)
            print('Available: %d' % max_CPU_cores)
            exitEarly = True

    return endEarly

# End reading command line arguments

def readFile( fileLoc ):

    if not path.exists( fileLoc ):
        print("File not found: %s" % fileLoc)
        return []

    try:
        fileIn = open(fileLoc,'r')

    except:
        print("File failed to open: %s" % fileLoc)
        return []

    else:
        fileContents = list(fileIn)
        fileIn.close()
        return fileContents


# Run main after declaring functions

if __name__=='__main__':

    argList = argv
    sdssImageCreator(argList)

