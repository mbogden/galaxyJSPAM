'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    
Description:    This is my python version 2 for creating images from spam particle files.
'''

from sys import \
        exit, \
        argv

# Global input variables
printAll = True
makeMask = False
overwriteImage = True
imageLoc = ''

partFileLoc1 = ''
partFileLoc2 = ''

infoFile = ''
paramFileLoc = ''
paramName = ''
runDir = ''


def main():

    endEarly = readArg()

    if endEarly:
        exit(-1)

# End main

def readArg():

    global printAll, maekMask, imageLoc, overwrite
    global infoFileLoc, paramFileLoc, partFileLoc1, partFileLoc2

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

        elif arg == '-imageLoc':
            imageLoc = argList[i+1]

        elif arg == '-partFile1':
            partFileLoc1 = argList[i+1]

        elif arg == '-partFile2':
            partFileLoc2 = argList[i+1]
            


    # Check if input arguments were valid
    endEarly = False

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
# end read argument file

# Run main after declaring functions
main()
