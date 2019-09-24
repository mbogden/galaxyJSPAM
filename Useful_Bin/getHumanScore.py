'''
    Author:     Matthew Ogden
    Created:    20 July 2019
    Altered:    19 Sep 2019
Description:    This program is temporary get images
'''

from sys import \
        exit, \
        argv, \
        stdout

from os import \
        path, \
        listdir

import pandas as pd

printAll = True

mainDir = ''
zooDir = ''


def main():

    endEarly = readArg()

    if printAll:
        print('mainDir     : %s' % mainDir)
        print('zooModelDir : %s' % zooDir)

    if endEarly:
        print('Exiting...')
        exit(-1)


    i = 0
    zooFiles = listdir(zooDir)
    sdssFolders = listdir(mainDir)

    for sf in sdssFolders:
        
        # find zoo model file
        for zf in zooFiles:
            
            if sf in zf:

                sdss_dir( mainDir + sf + '/' , zooDir + zf )



# End main

def sdss_dir( sdssDir, zooPath ):


    runDirList = listdir( sdssDir )

    toFrame = []

    # Create list run Paths
    for runDir in runDirList:
        if not path.exists( sdssDir + runDir ):
            continue
        if 'run' not in runDir:
            continue
        parse = runDir.split('_')
        try:
            toFrame.append( [ int(parse[2]), sdssDir + runDir + '/' ] )
        except:
            print("ERROR:")
    # End loop through runDir

    runFrame = pd.DataFrame( toFrame, columns=['runNum', 'runPath'] )
    runFrame = runFrame.sort_values(by=['runNum'])
    runFrame.reset_index()

    zooFile = readFile( zooPath )

    for i,run in runFrame.iterrows():
        
        j = run.runNum
        zooLine = zooFile[j].strip()

        stuff, mData = zooLine.split('\t')
        hScore = stuff.split(',')[1]
        win = stuff.split(',')[2]
        total = stuff.split(',')[3]
        winTotal = "%s/%s" % ( win, total )

        infoPath = run.runPath + 'info.txt'

        oldInfo = readFile( infoPath )

        for k, il in enumerate( oldInfo ):
            if mData in il:
                print(" YES! ", k)
                oldInfo.insert( k+1, "human_score %s\n" % hScore )
                oldInfo.insert( k+2, "wins_total %s\n" % winTotal )
                break

        for l in oldInfo:
            print(l.strip())

        newInfo = open( infoPath , 'w' )

        for l in oldInfo:
            newInfo.write(l)

        newInfo.close()

    return False

# End sdss_dir

 
def readArg():

    global printAll, mainDir, zooDir
    endEarly = False

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-mainDir':
            mainDir = argList[i+1]

        elif arg == '-zooDir':
            zooDir = argList[i+1]


    # Check if input arguments were valid

    if not path.exists(mainDir):
        print('mainDir not found: \'%s\'' % mainDir)
        endEarly = True

    if not path.exists(zooDir):
        print('zooDir not found: \'%s\'' % zooDir)
        endEarly = True

    return endEarly

# End reading command line arguments

def readFile( fPath ):

    try:
        fFile = open( fPath, 'r')
    except:
        print("Failed to open: %s" % fPath)
    else:
        fList = list(fFile)
        fFile.close()
        return fList

# Run main after declaring functions
main()
