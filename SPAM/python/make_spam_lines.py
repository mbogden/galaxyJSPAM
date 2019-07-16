'''
    Author:     Matthew Ogden
    Created:    16 July 2019
    Altered:    16 July 2019
Description:    This is a quick experiment, it's entire purpose is to make a list
                    within a file that is meant to be read and each line is treated
                    as an executable command on linux by batch_single_machine.py 
                    or other batch scripts.  This creates the executables 
                    I want for zooRun.py.  Whenever I need a custom batch
                    of spam particles this will be the code I edit.
'''

import os
import sys


# Global List of input arguments

sdssTargetLoc = ''
outLoc = ''
bigRunList = ''

nPart = 100000  # 100k
dataName = '100k'



def main():

    endEarly = readArg()

    if endEarly:
        print('Exiting...')
        sys.exit(-1)

    if sdssTargetLoc != '' and bigRunList != '':
        sdssList = readFile(sdssTargetLoc)
        bigList = readFile(bigRunList)

        makeTargetList( sdssList, bigList )

# End main

def makeTargetList( sList, bigList ):

    oFile = open(outLoc, 'w')


    for i, l in enumerate( bigList ):

        sdssName, genName, runName, hScore, zModel = l.split()
        
        if sdssName in sList:

            cmd = 'python3 SPAM/python/zooRun.py -compress \
-spam SPAM/bin/basic_run -name 100k -uID -n 100000 \
-o ~/spam_data/100k/%s/run_%s_%s/ -noDir -m %s'\
            % ( sdssName, genName, runName, zModel )

            oFile.write(cmd+'\n')




def readFile(inFileLoc):
    try:
        inFile = open(inFileLoc, 'r')
    except:
        print('Failed to open \'%s\'' % inFileLoc)
        print('Exiting...')
        sys.exit(-1)
    else:

        retList = []
        for l in inFile:
            retList.append( l.strip() )

        inFile.close()

        return retList
# End read SDSS targets


def readArg():

    global sdssTargetLoc, outLoc, bigRunList

    for i, arg in enumerate( sys.argv ):

        # Ignore arguments unless beginning with a specifier. 
        if arg[0] != '-':
            continue

        elif arg == '-sdssTargets':
            sdssTargetLoc = sys.argv[i+1]

        elif arg == '-o':
            outLoc = sys.argv[i+1]

        elif arg == '-bigList':
            bigRunList = sys.argv[i+1]

    # End looping through arguments


    endEarly = False


    # Check if arguments are valid

    if outLoc == '':
        print('Please specify output file name and location')
        endEarly = True

    if ( sdssTargetLoc != '' ) and ( not os.path.isfile( sdssTargetLoc ) ):
        print('SDSS List \'%s\' not found' % sdssTargetLoc)
        endEarly = True

    if ( bigRunList != '' ) and ( not os.path.isfile( bigRunList ) ):
        print('Big run list \'%s\' not found' % bigRunList )
        endEarly = True


    return endEarly

# End read arguments


# Start main after declaring functions
main()
