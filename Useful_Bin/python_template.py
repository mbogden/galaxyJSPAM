'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    29 Oct 2019
Description:    This is my python template for how I've been making my python programs
'''

from sys import \
        exit, \
        argv

from os import \
        path, \
        listdir

printAll = True

def main(argList):

    endEarly = readArg(argList)

    if endEarly:
        exit(-1)

# End main

def readArg(argList):

    global printAll

    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

    # Check if input arguments are valid

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

            # Skip line if empty
            if len(l) == 0:
                continue

            # Skip line if comment
            if l[0] == '#':
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
if __name__ == '__main__':
    argList = argv
    main(argList)
