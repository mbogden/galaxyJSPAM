'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    
Description:    This is my python version 2 for creating images from spam particle files.
'''

from sys import \
        exit, \
        argv

from os import \
        path,\
        listdir, \
        system

# Global input variables
printAll = True
makeMask = False
overwriteImage = True
imageLoc = ''

partLoc1 = ''
partLoc2 = ''

infoFile = ''
paramFileLoc = ''
paramName = ''
runDir = ''


def main():

    print('In image creator 2')

    endEarly = readArg()

    if printAll:
        print('runDir: %s' % runDir)
        print('partLoc1: %s' % partLoc1)
        print('partLoc2: %s' % partLoc2)

    if endEarly:
        exit(-1)



# End main

def readArg():

    global printAll, maekMask, imageLoc, overwrite, runDir
    global infoFileLoc, paramFileLoc, partLoc1, partLoc2

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-mask':
            makeMask = True

        elif arg == '-overwrite':
            overwrite = True

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir = runDir + '/'
            readRunDir(runDir)
            

    # Check if input arguments were valid
    endEarly = False
    
    if runDir == '':
        print('No run directory given')
    elif not path.exists(runDir):
        print('Run directory \'%s\' not found')
        endEarly = True

# End reading command line arguments

def readRunDir( runDir ):
    global partLoc1, partLoc2, infoLoc

    zipFile1 = ''
    zipFile2 = ''

    try:
        dirList = listdir( runDir)

    except:
        print("Run directory '%s' not found" % runDir)
        return

    else:
        
        for f in dirList:
            fPath = runDir + f

            if 'info' in f:
                infoLoc = fPath

            elif '.i.' in f:
                partLoc1 = fPath

            elif '.f.' in f:
                partLoc2 = fPath

            elif '000.zip' in f:
                zipFile1 = fPath

            elif '101.zip' in f:
                zipFile2 = fPath
        
        if partLoc1 == '' and zipFile1 != '':
            unzip = 'unzip -d %s -o %s' % (runDir, zipFile1)
            system(unzip)
            unzip = 'unzip -d %s -o %s' % (runDir, zipFile2)
            system(unzip)

            lDir = listdir(runDir)
            sDir = ''
            for f in lDir:
                if '588' in f:
                    sDir = runDir + f

            rDir = sDir + '/' + listdir(sDir)[0]
            lDir = listdir(rDir)

            p1Loc = ''
            p2Loc = ''
            for f in lDir:
                if '.i.' in f:
                    p1Loc = rDir + '/' + f
                if '.f.' in f:
                    p2Loc = rDir + '/' + f

            mvCmd1 = 'mv %s %sa.000' % (p1Loc, runDir)
            mvCmd2 = 'mv %s %sa.101' % (p2Loc, runDir)

            system(mvCmd1)
            system(mvCmd2)

            print(mvCmd1)

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
# end read argument file

# Run main after declaring functions
main()
