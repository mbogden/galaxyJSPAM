'''
    Author:     Matthew Ogden
    Created:    20 Sep 2019
    Altered:    28 Oct 2019
Description:    Get target img, info, and pair file from jspamstuff zip file
'''

from sys import \
        exit, \
        argv

from os import \
        path, \
        listdir, \
        system

from re import \
        search, \
        findall

import pandas as pd

# Global Variables
printAll = True
fromDir = ''
toDir = ''

def main():

    global fromDir, toDir

    fromDir = 'zippy/'
    toDir = 'targets/'

    if printAll:
        print("\nGetting Targets and info from jspamstuff")
        print("\tfromDir : %s" % fromDir)
        print("\ttoDir   : %s" % toDir )

    endEarly = False
    if endEarly:
        print('')
        exit(-1)

    print(path.exists( fromDir ))
    print(path.exists( toDir ))

    zipFiles = listdir( fromDir )

    for z in zipFiles:
        zFile = fromDir + z

        sName = zFile.split('/')[-1]
        sName = sName.split('.')[0]
        if 'sdss' in sName:
            sName = sName.split('sdss')[1]
        tDir = toDir + '%s/' % sName


        #mvZipStuff(zFile, tDir)
        checkContents(tDir)

    #moveFiles()
    #findThree()

def checkContents( tDir ):

    sFiles = listdir( tDir )
    nFiles = len( sFiles )

    imgF = False
    metaF = False
    pairF  = False

    for f in sFiles:

        if '.png' in f: imgF = True
        elif '.meta' in f: metaF = True
        elif '.pair' in f: pairF = True

    if not ( imgF and metaF and pairF ):
        print("Bad")




    

def mvZipStuff( zFile, tDir):

    if path.exists(zFile):
        uzCmd = 'unzip %s -d %s' % ( zFile, tDir )
        system(uzCmd)

    else:
        print("BAD!: %s" % zFile)


def findThree():

    toFolders = listdir(toDir)

    for folder in toFolders:
        fs = listdir( toDir + folder )
        if ( len(fs) == 3 ):
            print(fs)

        else:
            mvCmd = 'mv %s %s' % ( toDir + folder, toDir + 'incomplete/' + folder )
            print(mvCmd)


def moveFiles():
 
    fromFiles = listdir(fromDir)

    tFiles = [ f for f in fromFiles if search('png',f) ]
    pairCond = lambda s : (search('pair',s) and not search('pair\.pair',s) and not search('zip\.pair',s) )
    tFiles.extend( [ f for f in fromFiles if pairCond(f) ] )
    tFiles.extend( [ f for f in fromFiles if search('meta',f) ] )

    for f in tFiles:
        fromPath = fromDir + f

        sdss = findall('\d+',f)[0]

        d = toDir + sdss + '/'
        toPath = d + f

        print('%s   ---->   %s' % (fromPath, toPath))

        if not path.exists( toDir + sdss ):
            system('mkdir %s' % (d))
        
        system('cp %s %s' % ( fromPath, toPath ))
# End move files
   

    

# End main

def readArg():

    global printAll, fromDir, toDir

    argList = argv
    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-fromDir':
            fromDir = argList[i+1]
            if fromDir[-1] != '/':
                fromDir += '/'

        elif arg == '-toDir':
            toDir = argList[i+1]
            if toDir[-1] != '/':
                toDir += '/'

    # Check if input arguments were valid

    if not path.exists(fromDir):
        if fromDir == '':
            print("Please specify target directory with 'python3 getTargets.py -fromDir path/to/target'")
        else:
            print("Target directory not found: %s" % fromDir)
        endEarly = True

    if not path.exists(toDir):
        if toDir == '':
            print("Please specify To directory with 'python3 getTargets.py -toDir path/to/target'")
        else:
            print("Target directory not found: %s" % toDir)
        endEarly = True

    return endEarly

# End reading command line arguments


def readArgFile(argList, argFileLoc):

    try:
        argFile = open( argFileLoc, 'r')
    except:
        print("Failed to open/read argument file '%s'" % argFileLoc)
    else:

        for l in argFile:
            l = l.strip()

            # Skip line if comment
            if l[0] == '#':
                continue

            # Skip line if empty
            if len(l) == 0:
                continue

            lineItems = l.split()
            for item in lineItems:
                argList.append(item)
        # End going through file
    return argList
# end read argument file

def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return False, []
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return False, []

    else:
        inList = list(inFile)
        inFile.close()
        return True, inList

# End simple read file

# Run main after declaring functions
main()
print('')
