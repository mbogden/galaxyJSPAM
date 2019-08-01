'''
    Author:     Matthew Ogden
    Created:    19 July 2019
    Altered:    
Description:    This is my python template for how I've been making my python programs
'''

from sys import \
        exit, \
        argv


printAll = True


def main():

    endEarly = readArg()

    if endEarly:
        exit(-1)

# End main

def readArg():

    global printAll

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

    # Check if input arguments were valid

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
