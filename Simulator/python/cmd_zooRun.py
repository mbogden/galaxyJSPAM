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

outLoc = ''     # File of cmd I'm writing to

sdssTargetLoc = ''
zooModelsLoc = ''
bigRunList = ''
nLines = 0

nPart = 100000              # 100k
partFileName = 'pts'        # name given to particle files
dataDirName = '~/spam_data' 

printAll = True

def main():

    endEarly = readArg()

    if printAll:
        print('Writing to %s'%outLoc)
        print('points per gal: %d' % nPart)
        print('Using zoo model file %s'% zooModelLoc)
        print('Printing %d lines' % nLines)
        print('To data directory %s' % dataDirName)

    if endEarly:
        print('Exiting...')
        sys.exit(-1)

    if True:

        modelList = readFile(zooModelLoc)
        createCmdFileCluster(modelList, outLoc)


    # Ignore the following
    if False and os.path.exists( outLoc ):
        justTrimList()
        sys.exit(-1)

    if False and sdssTargetLoc != '' and bigRunList != '':
        sdssList = readFile(sdssTargetLoc)
        bigList = readFile(bigRunList)

        makeTargetList( sdssList, bigList )

# End main

def createCmdFileCluster( mList, oLoc ):

    oFile = open(oLoc, 'w')

    for i,l in enumerate(mList):
        
        l = l.strip()
        sName, gName, rName, hScore, mName, mData = l.split()

        cmd = 'python galaxyJSPAM/SPAM/python/zooRun.py'
        cmd += ' -spam galaxyJSPAM/SPAM/bin/basic_run'
        #cmd += ' -print'
        cmd += ' -compress'
        #cmd += ' -overwrite'
        cmd += ' -sdss %s' % sName
        cmd += ' -gen %s' % gName
        cmd += ' -run %s' % rName
        cmd += ' -name %s' % partFileName
        cmd += ' -n %d' % nPart
        cmd += ' -o %s' % ( '%s/%s/run_%s_%s/' % (dataDirName,sName,gName,rName)  )
        cmd += ' -noDir'
        cmd += ' -modelName %s' % mName
        cmd += ' -modelData %s' % mData
        cmd += ' -uID'
        cmd += '\n'

        if i < 3:
            print(cmd)
        oFile.write(cmd)

        if nLines != 0 and i >= nLines:
            break

# End create cmd list




def justTrimList():

    oFile = open( outLoc, 'r' )

    oldFile = list( oFile )

    oFile.close()

    newFile = []

    for lNum, l in enumerate(oldFile):

        lineItems = l.strip().split()

        outDir = ''

        # find output directory in exec line
        for i,item in enumerate(lineItems):
            if item == '-o':
                outDir = lineItems[i+1]
                outDir = os.path.expanduser( outDir )

        if outDir == '':
            print("Failed to trim line %d is file \'%s\'" % ( lNum, outLoc) )
            continue

        if not os.path.exists( outDir ):
            newFile.append(l)

        if lNum == 0:
            print('%s %s ' % ( outDir, l.strip() ), os.path.exists( outDir ))

    print('Trimming %d out of %d execLines' % ( len(oldFile) - len(newFile) , len(oldFile)))

    nFile = open( outLoc, 'w' )

    for l in newFile:
        nFile.write(l)

    nFile.close()


def makeTargetList( sList, bigList ):

    oFile = open(outLoc, 'w')

    for i, l in enumerate( bigList ):

        sdssName, genName, runName, hScore, zModel = l.split()
        
        if sdssName in sList:

            cmd = 'python3 SPAM/python/zooRun.py'
            cmd += ' -spam SPAM/bin/basic_run' 
            cmd += ' -compress'
            cmd += ' -name 100k'
            cmd += ' -n 100000'
            cmd += ' -o ~/spam_data/100k/%s/run_%s_%s/' % ( sdssName, genName, runName )
            cmd += ' -noDir'
            cmd += ' -m %s' % zModel
            cmd += ' -uID'
            cmd += '\n'

            if printAll:
                print(cmd)

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

    global sdssTargetLoc, outLoc, bigRunList, zooModelLoc
    global nLines, nPart, dataDirName, partFileName

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

        elif arg == '-noprint':
            printAll = False

        elif arg == '-zooLoc':
            zooModelLoc = sys.argv[i+1]

        elif arg == '-n':
            nLines = int(sys.argv[i+1])

        elif arg == '-nPart':
            nPart = int( sys.argv[i+1] )

        elif arg == '-dataDirName':
            dataDirName = sys.argv[i+1]

        elif arg == '-partFileName':
            partFileName = sys.argv[i+1]


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

    if dataDirName[-1] == '/':
        dataDirName = dataDirName[:-1]


    return endEarly

# End read arguments


# Start main after declaring functions
main()
