'''
    Author:     Matthew Ogden
    Created:    20 July 2019
    Altered:    
Description:    This program is for goign through run directories, reading score files, and analyzing the data.
'''

from sys import \
        exit, \
        argv, \
        stdout

from os import \
        path, \
        listdir, \
        system

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


printAll = True

basicPlots = False
samplePlots = False
sampleRuns = [ 0,2,5, 10,20,50, 100, 200, 500 ]
sampleImgLoc = []
scorePlots = False

sdssDir = ''

sdssName = ''
genName = ''
plotDir = ''



def main():

    endEarly = readArg()

    if endEarly:
        exit(-1)

    allFrame = readAllScoreFiles( sdssDir )
    nAll = len( allFrame.index )
    print(allFrame.columns)


    # Go through compairons methods
    methodList = allFrame.comparison_method.unique()
    print(methodList)

    for method in methodList:

        methodFrame = allFrame[ allFrame.comparison_method == method ]
        methodFrame = methodFrame.reset_index()

        print('%d / %d - %s' % ( len(methodFrame.index), nAll, method ))


        if scorePlots:
            createMethodPlot( method, methodFrame )

    # End going through comparison methods


    if samplePlots:
        createSamplePlots()

# End main

def createSamplePlots():


    for run in sampleRuns:

        runDir = sdssDir + "run_0_%d/" % int(run)

        try:
            runFiles = listdir( runDir )
        except:
            print("Failed to find run dir: %s" % runDir )
        else:

            for f in runFiles:
                if 'model.png' in f:
                    imgLoc = runDir + f
                    system("cp %s %s" % ( imgLoc, plotDir + "%d_%s" % (int( run), f) ) )

                if 'diff' in f:
                    diffLoc = runDir + f
                    system("cp %s %s" % ( diffLoc, plotDir + "%d_diff.png" % int( run) ) )




def createMethodPlot(method, mFrame ):

    global sampleRuns

    # Sort by Machine Score
    mFrame = mFrame.sort_values( by='machine_score', ascending=False)

    hScore = mFrame.human_score.values 
    mScore = mFrame.machine_score.values
    iScore = mFrame.index.values

    # Get Trend Line
    tLine = np.polyfit( hScore, mScore, 1 )
    trendLine = np.poly1d( tLine )

    # Try to identify top x machine scores
    x = 5
    topIndex = mFrame[0:x].index
    runName = mFrame[0:x].run
    print(topIndex[0:x])

    # Create Plot
    plotLoc = plotDir + '%s.png' % method

    # Add points and trendline to plot
    plt.scatter( hScore, mScore )
    plt.plot( hScore, trendLine(hScore), 'r--')


    # Add top x machine score labels
    for run, index in zip( runName,topIndex ):

        row = mFrame[ mFrame['run'] == run ]
        x = row['human_score'].values
        y = row['machine_score'].values

        plt.annotate( run, (x, y) )

        # Add run numbers and neighbors to sampleRuns
        sampleRuns.append(int(run)-1)
        sampleRuns.append(int(run))
        sampleRuns.append(int(run)+1)

    # End adding lables for top x


    # Setup plot
    print( mFrame.comparison_method.values[0])
    plt.title( 'Human Score vs Machine Score\n%s' % mFrame.comparison_method.values[0])
    plt.xlabel( 'Human Score' )
    plt.ylabel( 'Machine Score' )
    #plt.ylim( np.amin( mScore) , np.amax( mScore) )
    plt.tight_layout()

    if printAll:
        print('Saving plot %s' % plotLoc)
    plt.savefig( plotLoc )
    plt.clf()



def readAllScoreFiles( sdssDir ):

    global sdssName, genName, sampleImgLoc

    runDirList = listdir( sdssDir )
    
    if printAll:
        print('Found %d directories in runDir list'%len(runDirList))


    scoreLoc = sdssDir + runDirList[0] + '/scores.csv'
    try:
        initFrame = pd.read_csv( scoreLoc )
    except:
        print('Failed to open "%s"' % scoreLoc)
        exit(-1)

    
    if len( initFrame.sdss.unique() ) != 1:
        print('Found more than 1 sdss name in score files')
        exit(-1)

    if len( initFrame.generation.unique() ) != 1:
        print('Found more than 1 generation in score files')
        exit(-1)

    initCol = initFrame.columns

    sdssName = initFrame.sdss[0]
    genName = initFrame.generation[0]

    if printAll:
        print('Gathering results for generation %s of sdss "%s"' % ( genName, sdssName))

    for i, runDir in enumerate(runDirList): 

        # If first, skip
        if not 'run' in runDir or i == 0:
            continue

        scoreLoc = sdssDir + runDir + '/scores.csv'

        if not path.isfile( scoreLoc ):
            print('Didn\'t find score file in %s' % scoreLoc)
            continue
        
        scoreFrame = pd.read_csv( scoreLoc )
        scoreCol = scoreFrame.columns

        # Check that headers match
        headersGood = True
        for j in range( len( scoreCol )):
            if initCol[j] != scoreCol[j]:
                print('runDir %s - header %s is not %s' % ( runDir, initCol[j], scoreCol[j] ) )
                headersGood = False
                break
        if not headersGood:
           return

        initFrame = initFrame.append(scoreFrame)


        if printAll:
            #pass
            stdout.write('Score files read: %d / %d\r' % ( i, len( runDirList ) ) )
        
        '''
        # Check to add for sample plots
        if samplePlots and scoreFrame.run[0] in sampleRuns: 

            listDir = listdir( sdssDir + runDir )
            imageLoc = ''
            for f in listDir:
                if 'model.png' in f:
                    imageLoc = sdssDir + runDir +'/'+ f
            sampleImgLoc.append(imageLoc)
        '''

    if printAll:
        print('\nLoaded all dataframes')

    # Drop unneeded columns
    initFrame = initFrame.drop( columns=[ 'sdss','generation',\
            'zoo_model_data','target_image','model_image'] )


    return initFrame.reset_index()

# End read score files

def readArg():

    global printAll, sdssDir, plotDir 
    global scorePlots, samplePlots

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]
            if sdssDir[-1] != '/':
                sdssDir = sdssDir + '/'

        elif arg == '-plotDir':
            plotDir = argList[i+1]
            if plotDir[-1] != '/':
                plotDir = plotDir + '/'

        elif arg == '-basicPlots':
            scorePlots = True
            samplePlots = True

    # Check if input arguments were valid
    endEarly = False

    if sdssDir == '':
        print("Please enter an sdss directory")
        endEarly = True

    if plotDir != '' and not path.exists(plotDir):
        try:
            if printAll:
                print('Creating directory for plots: %s' % plotDir )
            system( 'mkdir -p %s' % plotDir )
        except:
            print('Failed to make plot directory: %s' % plotDir )
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

# end read argument file

# Run main after declaring all functions
main()
