'''
    Author:     Matthew Ogden
    Created:    22 Apr 2020
Description:    *** Inital creation of code ***
                For analyzing scores and models and stuff. 
'''

from os import path, listdir
from sys import path as sysPath

# For loading in Matt's general purpose python libraries
sysPath.append( path.abspath( "Support_Code/" ) )
sysPath.append( path.abspath( path.join( __file__ , "../../Support_Code/" ) ) )
import general_module as gm
import info_module as im

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test():
    print("SA: Hi!  You're in Matthew's Main program for score analysis!")

def main(arg):

    if arg.printAll:

        arg.printArg()
        gm.test()
        im.test()

    # end main print
    
    if arg.simple:
        if arg.printBase: 
            print("PT: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.targetDir != None:
        procTarget( arg.targetDir, printAll=arg.printAll )

    elif arg.dataDir != None:
        procDataDir( arg.dataDir, arg )

    else:
        print("PT: Nothing selected!")
        print("PT: Recommended options")
        print("\t - simple")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

    return

    # OLD
    if newSdss:
        initSdss()

    else:
        newMain()

# End main
 

def procDataDir( dataDir, arg = gm.inArgClass() ):

    printBase=arg.printBase
    printAll=arg.printAll 
    pltPath = arg.get( 'pltDir', None )
    
    if printBase: 
        print("SA.procAllData")
        print("\t - dataDir: %s" % arg.dataDir )

    dataDir = gm.validPath( dataDir, pathType='dir' )

    # exit if invalid directory
    if dataDir == None:
        if printBase: 
            print("SA.procAllData: WARNING")
            print("\t - dataDir not found" )
            print("\t - dataDir: %s - '%s'" % (type(arg.dataDir), arg.dataDir) )
        return

    allScores = gatherAllScores( dataDir, )    
    
    if printBase:
        print("SA: Printing Return Scores")

    for tInfo, tScores in allScores:
        print( '%s : %s' % ( tInfo.get('target_identifier'), list(tScores.columns) ) )
    
    pLoc = 'param/t.json'
    path.exists( pLoc )
    param = im.score_parameter_class( paramLoc = pLoc )
    
    plotAll( allScores, param = param, pltPath = pltPath )    
    

def plotAll( allScores, param = None, pltPath = None ):
    
    if param == None:
        print("SA.plotAll: WARNING: Please give param")
        return 
    
    pName = param.get('name')
    
    newPath = gm.validPath( pltPath, pathType = 'dir' )
    if newPath == None:
        print("SA.plotAll: WARNING: Plots path not found: %s" % pltPath)
        return
    else:
        pltPath = newPath
    
    for tInfo, tScores in allScores:
        
        if pName not in tScores.columns:
            print("not found")
            continue
        
        tName = tInfo.get('target_identifier')
        tImgName = param.get('tgtArg')
        
        hScores = { 'name': 'zoo_merger_score' , 'scores': tScores['zoo_merger_score'].values }
        mScores = { 'name': pName, 'scores': tScores[pName].values }
        
        tLoc = tInfo.getTargetImage( tName = tImgName )
        #print('tLoc: ',tLoc)
        tImg = gm.getImg( tLoc )
        
        plotName = '%s:\n%s' % (tName, pName)
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        plt.title( plotName)
        
        basicSubPlot( ax2, hScores, mScores )
        ax1.imshow( tImg, cmap=plt.gray())
        plt.show()
        
        pltLoc = pltPath + '/%s_%s.png' % ( tName, pName )
        fig.savefig( pltLoc )
        print(pltLoc)


def basicSubPlot( ax, s1, s2, titleName=None, plotLoc = None ):


    # Add points and trendline to plot
    ax.scatter( s1['scores'], s2['scores'] )

    # Setup plot
    ax.set( xlabel = s1['name'], ylabel = s2['name'] )

    if titleName != None:
        ax.set_title( '%s' % titleName)

    return ax


def gatherAllScores( dataDir, printBase = True):

    tScoreList = []

    dataList = listdir( dataDir )
    if dataDir[-1] != '/': dataDir += '/'

    # Find target directories
    for folder in dataList:
        tDir = dataDir + folder

        tReturn = gatherTargetScores( tDir = tDir, \
                    printBase = printBase, )

        if tReturn != None:
            tScoreList.append( tReturn )


    return tScoreList


# Process target directory
def gatherTargetScores( tDir, printBase = True, printAll=False ):

    if printBase:
        print("SA: procTarget:")
        print("\t - tDir: " , tDir)

    tInfo = im.target_info_class( targetDir=tDir, printBase = printBase, printAll=printAll )

    if printBase:
        print("SA: procTarget:")
        print("\t - tInfo.status: %s" % tInfo.status )

    # Check if target is valid
    if tInfo.status == False:
        print("SA: WARNING: procTarget:")
        print("\t - Target not good.")
        return

    if printAll:
        tInfo.printInfo( )

    scores = tInfo.getScores()

    return tInfo, scores

   
# Process target directory
def old_procTarget( tDir, printAll=False ):

    print("Only using correlation for now")

    h = ( 'galaxy_zoo_mergers', allFrame['human_galaxy_zoo_mergers'].values )
    m = ( 'machine', allFrame['machine_correlation'].values )
    p = ( 'perturbation', allFrame['perturbation_correlation'].values )
    i = ( 'initial_bias', allFrame['initialBias_correlation'].values )
 
    filter_me = p[1] < .75
    fFrame = allFrame[ filter_me ]

    fh = ( 'galaxy_zoo_mergers', fFrame['human_galaxy_zoo_mergers'].values )
    fm = ( 'machine', fFrame['machine_correlation'].values )
    fp = ( 'perturbation', fFrame['perturbation_correlation'].values )
    fi = ( 'initial_bias', fFrame['initialBias_correlation'].values )
   

    pSubi = ( '%s - %s' % ( p[0], i[0] ), p[1] - i[1] ) 
    mSubi = ( '%s - %s' % ( m[0], i[0] ), m[1] - i[1] ) 


    newM = ( 'new score thing', m[1] * np.exp( - ( p[1] - i[1] )**2 / ( 2 * .15**2 ) ) )


    createHeatPlot_v2( h, newM, p, saveLoc = tInfo.plotDir + 'new.png' ) 

    '''
    createHeatPlot_v2( m, p, h, saveLoc = tInfo.plotDir + '0_1m2.png' ) 

    createHeatPlot_v2( m, p, h, saveLoc = tInfo.plotDir + '0_1m2.png' ) 

    createHeatPlot_v2( h, m, p, saveLoc = tInfo.plotDir + '0_0h1.png' ) 
    createHeatPlot_v2( h, m, i, saveLoc = tInfo.plotDir + '0_0h2.png' ) 
    createHeatPlot_v2( h, i, p, saveLoc = tInfo.plotDir + '0_0h3.png' ) 
    createHeatPlot_v2( h, i, m, saveLoc = tInfo.plotDir + '0_0h4.png' ) 
    createHeatPlot_v2( h, p, i, saveLoc = tInfo.plotDir + '0_0h5.png' ) 
    createHeatPlot_v2( h, p, m, saveLoc = tInfo.plotDir + '0_0h6.png' ) 

    createHeatPlot_v2( m, i, h, saveLoc = tInfo.plotDir + '0_1m0.png' ) 
    createHeatPlot_v2( m, i, p, saveLoc = tInfo.plotDir + '0_1m1.png' ) 
    createHeatPlot_v2( m, p, h, saveLoc = tInfo.plotDir + '0_1m2.png' ) 
    createHeatPlot_v2( m, p, i, saveLoc = tInfo.plotDir + '0_1m3.png' ) 

    createHeatPlot_v2( p, i, h, saveLoc = tInfo.plotDir + '0_2i0.png' ) 
    createHeatPlot_v2( p, i, m, saveLoc = tInfo.plotDir + '0_2i1.png' ) 

    createHeatPlot_v2( fh, fm, fp, saveLoc = tInfo.plotDir + '0_0h1f.png' ) 
    createHeatPlot_v2( fh, fm, fp, saveLoc = tInfo.plotDir + '1_1f.png' ) 
    createHeatPlot_v2( fh, fm, fi, saveLoc = tInfo.plotDir + '1_2f.png' ) 
    createHeatPlot_v2( fm, fp, fh, saveLoc = tInfo.plotDir + '1_3f.png' ) 
    createHeatPlot_v2( fm, fp, fi, saveLoc = tInfo.plotDir + '1_4f.png' ) 
    createHeatPlot_v2( fp, fi, fh, saveLoc = tInfo.plotDir + '1_5f.png' ) 
    createHeatPlot_v2( fp, fi, fm, saveLoc = tInfo.plotDir + '1_6f.png' ) 


    createHeatPlot_v2( h, mSubi, m, saveLoc = tInfo.plotDir + '2_mSubi_1.png', mH = False  ) 
    createHeatPlot_v2( h, mSubi, p, saveLoc = tInfo.plotDir + '2_mSubi_2.png', mH = False  ) 
    createHeatPlot_v2( h, mSubi, i, saveLoc = tInfo.plotDir + '2_mSubi_3.png', mH = False  ) 

    createHeatPlot_v2( mSubi, p, i, saveLoc = tInfo.plotDir + '2_mSubi_4.png', mH = False  ) 
    createHeatPlot_v2( mSubi, i, p, saveLoc = tInfo.plotDir + '2_mSubi_5.png', mH = False  ) 
    createHeatPlot_v2( p, i, mSubi, saveLoc = tInfo.plotDir + '2_mSubi_6.png', mH = False  ) 

    createHeatPlot_v2( h, m, pSubi, saveLoc = tInfo.plotDir + '3_pSubi_1.png', mH = False  ) 
    createHeatPlot_v2( h, pSubi, m, saveLoc = tInfo.plotDir + '3_pSubi_2.png', mH = False  ) 
    createHeatPlot_v2( h, pSubi, p, saveLoc = tInfo.plotDir + '3_pSubi_3.png', mH = False  ) 
    createHeatPlot_v2( h, pSubi, i, saveLoc = tInfo.plotDir + '3_pSubi_4.png', mH = False  ) 

    createHeatPlot_v2( m, pSubi, i, saveLoc = tInfo.plotDir + '3_mSubi_5.png', mH = False  ) 
    createHeatPlot_v2( m, i, pSubi, saveLoc = tInfo.plotDir + '3_mSubi_6.png', mH = False  ) 
    createHeatPlot_v2( pSubi, i, m, saveLoc = tInfo.plotDir + '3_mSubi_7.png', mH = False  ) 

    createHeatPlot_v2( h, mSubi, pSubi, saveLoc = tInfo.plotDir + '4_pmSubi_1.png', mH = False  ) 
    createHeatPlot_v2( h, pSubi, mSubi, saveLoc = tInfo.plotDir + '4_pmSubi_2.png', mH = False  ) 

    createHeatPlot_v2( mSubi, pSubi, i, saveLoc = tInfo.plotDir + '4_mSubi_5.png', mH = False  ) 
    createHeatPlot_v2( mSubi, i, pSubi, saveLoc = tInfo.plotDir + '4_mSubi_6.png', mH = False  ) 
    createHeatPlot_v2( pSubi, i, mSubi, saveLoc = tInfo.plotDir + '4_mSubi_7.png', mH = False  ) 
    '''

def createHeatPlot_v2( s1, s2, s3, saveLoc = None, titleName=None, mH=True ):

    plt.clf()

    c1 = ( s1[0] +' v '+ s2[0], np.corrcoef( s1[1], s2[1])[0,1] )
    c2 = ( s1[0] +' v '+ s3[0], np.corrcoef( s1[1], s3[1])[0,1] )
    c3 = ( s2[0] +' v '+ s3[0], np.corrcoef( s2[1], s3[1])[0,1] )

    #hMin = np.amin( s3[1] )
    #hMax = np.amax( s3[1] )

    cMap = plt.cm.get_cmap('RdYlBu_r')

    if mH:
        plot = plt.scatter( s1[1], s2[1], c=s3[1], s=5, cmap=cMap, vmax=1.0)
    else:
        plot = plt.scatter( s1[1], s2[1], c=s3[1], s=5, cmap=cMap)

    cBar = plt.colorbar(plot)
    
    #plt.xlim(0,1)
    #plt.ylim(0,1)

    if titleName != None:
        plt.title( titleName)


    plt.xlabel(s1[0])
    plt.xlim(right=1.0)
    plt.ylabel(s2[0])
    plt.ylim(top=1.0)
    cBar.set_label( s3[0] )

    ax = plt.gca()
    ax.set_facecolor("xkcd:grey")

    if saveLoc != None:
        plt.savefig( saveLoc )

# End plot creation




def incomplete(): 
    # Sort values
    humanList = [ name for name in allFrame.columns if 'human_' in name ]
    machineList = [ name for name in allFrame.columns if 'machine_' in name ]
    perturbList = [ name for name in allFrame.columns if 'perturbation_' in name ]

    # Hard coded for single human scores
    hScores = allFrame[ humanList[0] ].values
    pScores = allFrame[ perturbList[0] ] .values

    # Go through compairons methods
    for mMethod in machineList:
        mScores = allFrame[ mMethod ].values
        for pMethod in perturbList:
            pScores = allFrame[ pMethod ].values

            title = mMethod + '\n' + pMethod
            saveLoc = tInfo.plotDir + mMethod + '-' + pMethod + '.png'

            createHeatPlot( hScores, mScores, pScores = pScores, titleName=title, saveLoc = saveLoc )


    # End going through comparison methods

# End processing target dir

def createMethodPlot( hScores, mScores, titleName=None, plotLoc = None ):

    plt.clf()

    # Add points and trendline to plot
    r = plt.scatter( hScores, mScores )

    # Setup plot
    plt.xlabel( 'Human Score' )
    plt.ylabel( 'Machine Score' )
    plt.tight_layout()

    if titleName != None:
        plt.title( '%s' % titleName)

    if plotLoc != None:
        print( plotLoc )
        plt.savefig( plotLoc )

    return r



def createHeatPlot( hScores, mScores, pScores = None, saveLoc = None, titleName=None ):

    plt.clf()
    corrVal = np.corrcoef(hScores,mScores)[0,1]

    if type( pScores ) == type( None ):
        plot = plt.scatter( hScores, mScores, s=5 )

    else:
        pMin = np.amin( pScores )
        pMax = np.amax( pScores )
        cMap = plt.cm.get_cmap('RdYlBu_r')

        plot = plt.scatter( hScores, mScores, c=pScores, vmin = pMin, vmax = 1.0, s=5, cmap=cMap)

        cBar = plt.colorbar(plot)
        cBar.set_label("Correlation with Unperturbed model image")
        
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.title( titleName + "\nCorrelation: %f" % corrVal )
    plt.xlabel("Human Scores")
    plt.ylabel("Machine Score")

    ax = plt.gca()
    ax.set_facecolor("xkcd:grey")

    if saveLoc != None:
        plt.savefig( saveLoc )

# End plot creation


def newMain():

    if imgDir != '':
        prepImgDir()

    elif sdssDir != '':
        prepSdss()

    '''
    getTrainingImages( 0.3 )
    exit()
    '''

    #newSdss = True

    if newSdss:
        initSdss()

    loadImgs = False
    loadImgs = True

    if loadImgs:

        tImg = cv2.imread( targetLoc, 0 )

        if imgDir != '':
            mImgs, uImgs = getImgs( imgDir )

        else:
            mImgs = getImgsSdss( 'model.png' )
            uImgs = getImgsSdss( 'init.png' )

    pModel = ft.loadFilterDir( filterDir )

    mImgs = ft.convertCV2Img( mImgs )
    mPred = ft.predictImg( mImgs, pModel )

    allScores = getScores()
    hScores = allScores.human_scores
    #mScores = allScores.target_pixel_difference
    mScores = allScores.target_correlation
    printCol( allScores )


    h = 0.3

    q1File = open( 'q1.txt' , 'w' )
    q4File = open( 'q4.txt' , 'w' )

    n = len( hScores )
    
    for i in range( n ):

        if hScores[i] < 0.3 and mPred[i] == 0:
            q1File.write( '%d\n' % i )

        if hScores[i] > 0.3 and mPred[i] == 1:
            q4File.write( '%d\n' % i )

    q1File.close()
    q4File.close()
    

    '''
    classPlot( hScores, mScores, mPred, plotDir + 'Neural_Disk_plot_1.png', 'Neural Network Disk Classification' )
    classFilterPlot( hScores, mScores, mPred, plotDir + 'Neural_Disk_plot_2.png', 'Neural Network Disk Classification' )
    '''

def printCol( allScores ):

    colNames = allScores.columns
    for c in colNames:
        print(c)


def exampleFilter():


    imgsLoc = 'Comparison_Methods/filters/'
    gImg = cv2.imread( imgsLoc + 'good.png', 0 )
    bImg = cv2.imread( imgsLoc + 'bad.png', 0 )

    imgs = ft.convertCV2Img( [ gImg, bImg ] )

    pModel = ft.loadFilterDir( filterDir )

    print( "Predictions!: ", ft.predictImg( imgs, pModel ) )



def getTrainingImages(h):

    allScores = getScores()
    hScores = allScores.human_scores
    pScores = allScores.perturbedness_correlation

    for i,row in allScores.iterrows():

        rNum = int ( row.run )

        fromLoc = imgDir + '%04d_model.png' % rNum

        if row.human_scores < h:
            toLoc = path.abspath( filterDir + 'badImgs/' ) + '/.'
        else:
            toLoc = path.abspath( filterDir + 'goodImgs/' ) + '/.'

        cmd = 'cp %s %s' % ( fromLoc, toLoc )
        system( cmd ) 
        

def old():

    #createScoreCSV()
    #makeAllPlots()


    allScores = getScores()
    hScores = allScores.human_scores
    pScores = allScores.perturbedness_correlation
    colNames = allScores.columns

    h = 0.90
    uVal = allScores[ allScores['perturbedness_correlation'] > h ]
    pVal = allScores[ allScores['perturbedness_correlation'] <= h ]

    print( len( uVal ) , len( pVal ) )


    '''

    for i,row in allScores.iterrows():

        print(i, row)

        rNum = int ( row.run )
        print(i, row.perturbedness_correlation)

        fromLoc = imgDir + '%04d_model.png' % rNum

        if row.perturbedness_correlation > h:
            toLoc = path.abspath( imgDir + '../badImgs/' ) + '/.'
        else:
            toLoc = path.abspath( imgDir + '../goodImgs/' ) + '/.'

        cmd = 'cp %s %s' % ( fromLoc, toLoc )
        print( cmd )
        system( cmd ) 
        
    '''


    for c in colNames:
        #print(c)
        pass


    plt.clf()
    plt.hist( pScores, 25 )

    plt.title( "Perturbedness" )
    plt.savefig( plotDir + 'perturbedness_histogram.png' )

   

    print("About to filter")
    #newScores = filterScores3( allScores, 'target_overlap', 0.7, 0.85 )
    print("Filtered")

    newScores = allScores[ allScores[ 'perturbedness_correlation' ] < 0.85 ] 
    newScores = newScores[ newScores[ 'perturbedness_correlation' ] > 0.7 ] 


    '''
    hScores = newScores.human_scores
    pScores = newScores.perturbedness_correlation
    mScores = newScores.target_overlap
    col = 'overlap'

    createHeatPlot( hScores, mScores, pScores , plotDir + '%s_%s_filtered2_plot.png'% ( sdssName, col ), '%s_%s_Filtered' % ( sdssName, col ) )
    '''
    


def prepImgDir():
    global plotDir, scoreDir, targetLoc, sdssName

    scoreDir = path.abspath( imgDir + '../newScores/' ) + '/'
    plotDir = scoreDir
    targetLoc = imgDir + 'target.png'
    sdssName = 'sdss'

def prepSdss():
    global plotDir, scoreDir, targetLoc, sdssName
    
    sdssName = sdssDir.split('/')[-2]

    if plotDir == '':
        plotDir = sdssDir + 'plots/'
        if not path.exists( plotDir ):
            system("mkdir %s" % plotDir )

    if scoreDir == '':
        scoreDir = sdssDir + 'scores/'
        if not path.exists( scoreDir ):
            system("mkdir %s" % scoreDir )

    if targetLoc == '':
        targetLoc = sdssDir + 'sdssParameters/target_zoo.png'


def binTargets(tImg):

    tbImg = ms.binImg( tImg, 55 )
    cv2.imwrite( sdssDir + 'sdssParameters/target_zoo_binary.png', tbImg )
    binDir = sdssDir + 'sdssParameters/binTargets/'
    system( 'mkdir %s' % binDir )




def initSdss():

    if sdssDir != '':
        prepSdss()
    else:
        prepImgDir()

    global scoreDir, plotDir
    scoreDir = path.abspath( imgDir + '../newScores/' ) + '/'
    plotDir = scoreDir

    tImg = cv2.imread( imgDir + 'target.png', 0 )
    mImgs, uImgs = getImgs( imgDir )


    '''
    tImg = cv2.imread( targetLoc, 0 )
    #binTargets(tImg)

    mImgs = getImgsSdss( 'v3_test_param_model.png' )
    uImgs = getImgsSdss( 'v3_test_param_init.png' )
    '''

    mImgs = adjModelImgs( tImg, mImgs )
    uImgs = adjModelImgs( tImg, uImgs )
    tB = np.sum( tImg )

    # Gather Human Scores
    '''
    hScores = gatherHumanScores()
    comment = 'human_scores'
    saveScores2( hScores, comment, scoreDir + 'humanScores.txt' )
    '''

    # Make score for perturbedness between model image and unperturbed image
    pScores = makePerturbedness( ms.scoreCorrelation, mImgs, uImgs )
    comment = 'perturbedness_correlation' 
    saveScores2( pScores, comment, scoreDir + 'perturbedScores.txt' )


    # Make pixel difference scores for target and model images
    cScores = makeScores2( ms.scoreAbsDiff, tImg, mImgs )
    comment = 'target_pixel_difference'
    saveScores2( cScores, comment, scoreDir + 'absPixelDifferenceScores.txt' )

    # Make correlation scores for target and model images
    cScores = makeScores2( ms.scoreCorrelation, tImg, mImgs )
    comment = 'target_correlation'
    saveScores2( cScores, comment, scoreDir + 'correlationScores.txt' )

    # Make Binary correlation scores for target and model images
    cScores = makeScores3( ms.scoreBinaryCorrelation, tImg, mImgs, 1 )
    comment = 'target_correlation_binary'
    saveScores2( cScores, comment, scoreDir + 'correlationBinaryScores.txt' )

    # Make Binary correlation scores for target and model images
    cScores = makeScores3( ms.scoreOverLap, tImg, mImgs, 1 )
    comment = 'target_overlap'
    saveScores2( cScores, comment, scoreDir + 'overlapScores.txt' )


    createScoreCSV()

    makeAllPlots()


def makeAllPlots():

    print("Making all Plots")

    allScores = getScores()

    hScores = allScores.human_scores
    pScores = allScores.perturbedness_correlation

    colNames = allScores.columns
   
    for col in colNames:
        # skip if not a machine score
        if 'target' not in col: continue

        mScores = allScores[col]
        createHeatPlot( hScores, mScores, pScores, plotDir + '%s_%s_plot.png'% ( sdssName, col ), '%s_%s' % ( sdssName, col ) )




def adjModelImgs( tImg, mImgs ):

    tB = np.sum( tImg )
    n = len( mImgs )

    print("Adjusting image brightness")
    for i,m in enumerate(mImgs):
        mB = np.sum( m )
        nImg = mImgs[i] * tB/mB
        nImg[ nImg >= 255 ] = 255
        mImgs[i] = np.uint8( nImg )


        print( '%.1f - %d / %d' % ( 100*((i+1)/n), i, n ), end='\r' )

    print("Adjusting image brightness - Done")
    return mImgs

def adjInitImgs( tImg, mImgs ):

    tB = np.sum( tImg )
    n = len( mImgs )

    print("Adjusting image brightness")
    for i,m in enumerate(mImgs):
        mB = np.sum( m )
        nImg = mImgs[i] * tB/mB
        nImg[ nImg >= 255 ] = 255
        mImgs[i] = np.uint8( nImg )


        print( '%.1f - %d / %d' % ( 100*((i+1)/n), i, n ), end='\r' )

    print("Adjusting image brightness - Done")
    return mImgs

def findExpFuncMax( mScores, pScores, hScores ):

    eVal = np.linspace( 1, 100, 100 )
    fVal = np.linspace( 0, 1.0, 100 )
    
    eImg = np.zeros((100,100))
    fImg = np.zeros((100,100))
    cImg = np.zeros((100,100))

    for i,e in enumerate( eVal ):

        for j, f in enumerate( fVal ):

            nScores = expFunc( mScores, pScores, e, f )

            cImg[i,j] = np.corrcoef( hScores, nScores)[0,1]
            eImg[i,j] = e
            fImg[i,j] = f

    cPlot = cImg.flatten()
    ePlot = eImg.flatten()
    fPlot = fImg.flatten()

    cMax = np.amax( cPlot )
    iMax = np.argmax( cPlot )
    fMax = fPlot[iMax]
    eMax = ePlot[iMax]

    return eMax, fMax


def gatherHumanScores( ):

    gDir = sdssDir + 'gen000/'

    runDirList = listdir( gDir )
    runDirList.sort()
    n = len( runDirList )

    scores = np.zeros(n)

    print("Getting Human Scores")

    for i,rDir in enumerate(runDirList):
        runDir = gDir + rDir + '/'

        if not path.exists( runDir ):
            print("runDir doesn't exist: %s" % runDir)
            continue

        if 'run' not in runDir:
            print("Not a run dir: %s" % runDir)
            continue

        iLoc = runDir + 'info.txt'

        iFile = readFile( iLoc )
        for l in iFile:
            if 'human_score' in l:
                scores[i] = l.strip().split()[1]

        print( '%.1f - %d / %d - %s' % ( 100*((i+1)/n), i, n, rDir ), end='\r' )

    print("Gathered Human Scores")
    return scores


def getScores():
    try:
        return pd.read_csv( scoreDir + 'scores.csv' )
    except:
        print("No scores.csv found at %s" % scoreDir + 'scores.csv')

def createScoreCSV():
    
    allFiles = listdir( scoreDir )

    scoreFiles = [ f for f in allFiles if 'Scores.txt' in f ]

    allScores = pd.DataFrame()

    for sName in scoreFiles:
        sLoc = scoreDir + sName
        sFrame = pd.read_csv( sLoc )

        print(sFrame.columns[0])
        allScores[sFrame.columns[0]] = sFrame

    allScores.insert( 0, 'run', allScores.index.values )
    saveLoc = scoreDir + 'scores.csv'
    allScores.to_csv(r'%s'%saveLoc, index = None, header=True)


def makeScores3( scorePtr, tImg, mImgs, inVal ):

    n = len( mImgs )
    scores = np.zeros( n )

    print("Getting Scores" )
    for i in range(n):

        scores[i] = scorePtr( tImg, mImgs[i], inVal )
        print( '%.1f - %d / %d' % ( 100*((i+1)/n), i, n ), end='\r' )

    print('')

    return scores




def makeScores2( scorePtr, tImg, mImgs ):

    n = len( mImgs )
    scores = np.zeros( n )

    print("Getting Scores" )
    for i in range(n):

        scores[i] = scorePtr( tImg, mImgs[i] )
        print( '%.1f - %d / %d' % ( 100*((i+1)/n), i, n ), end='\r' )

    print('')

    return scores




def makeScores2pp( scorePtr, tImg, mImgs ):

    n = len( mImgs )
    scores = np.zeros( n )

    print("Getting Scores Parallel" )

    argList = []
    for i in mImgs:
        argList.append( ( tImg, i ) )

    scores = pool.map( scorePtr, argList )

    '''
    for i in range(n):

        scores[i] = scorePtr( tAr, mImgs[i] )
        print( '%.1f - %d / %d' % ( 100*((i+1)/n), i, n ), end='\r' )

    print('')
    '''
    return scores



def saveScores2( scores, comment, scoreLoc ):

    oFile = open( scoreLoc, 'w' )

    oFile.write( '%s\n' % comment )

    for s in scores:
        oFile.write( '%f\n' % s )

    oFile.close()



def makePerturbedness( scorePtr, mImgs, uImgs ):

    n = len( mImgs )
    scores = np.zeros( n )

    print("Getting Perturbedness" )
    for i in range(n):

        scores[i] = scorePtr( mImgs[i], uImgs[i] )
        print( '%.1f - %d / %d' % ( 100*((i+1)/n), i, n ), end='\r' )

    print('')

    return scores



def getImgsSdss(keyWord ):

    gDir = sdssDir + 'gen000/'
    runDirList = listdir( gDir )
    runDirList.sort()
    n = len( runDirList )

    imgs = []

    print("Getting %s images" % keyWord)

    for i,rDir in enumerate(runDirList):
        runDir = gDir + rDir + '/'

        if not path.exists( runDir ):
            print("runDir doesn't exist: %s" % runDir)
            continue

        if 'run' not in runDir:
            print("Not a run dir: %s" % runDir)
            continue

        rNum = int(rDir.split('_')[1])

        imgDir = runDir + 'model_images/'
        imgLoc = ''

        imgFiles = listdir( imgDir )

        for f in imgFiles:
            if keyWord in f:
                imgLoc = imgDir + f

        imgs.append( cv2.imread( imgLoc, 0 ) )

        print( '%.1f - %d / %d - %s' % ( 100*((i+1)/n), i, n, rDir ), end='\r' )

    print("Gathered %s images" % keyWord)
    return imgs
        
def expFunc( inScores, fScores, eVal, fVal ):
    nScores = inScores * ( np.exp( - eVal * np.abs( fVal - fScores )**2 ) )
    return nScores


def createHeatPlot( hScores, mScores, pScores, saveLoc, titleName ):

    plt.clf()
    corrVal = np.corrcoef(hScores,mScores)[0,1]

    pMin = np.amin( pScores )
    pMax = np.amax( pScores )

    cMap = plt.cm.get_cmap('RdYlBu_r')
    plot = plt.scatter( hScores, mScores, c=pScores, vmin = pMin, vmax = 1.0, s=5, cmap=cMap)

    cBar = plt.colorbar(plot)
    cBar.set_label("Correlation with Unperturbed model image")
    
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.title( titleName + "\nCorrelation: %f" % corrVal )
    plt.xlabel("Human Scores")
    plt.ylabel("Machine Score")

    ax = plt.gca()
    ax.set_facecolor("xkcd:grey")

    plt.savefig( saveLoc )

# End plot creation



def classPlot( hScores, mScores, classList, saveLoc, titleName ):

    corrVal = np.corrcoef(hScores,mScores)[0,1]

    h1 = []
    m1 = []

    h2 = []
    m2 = []

    q = [0, 0, 0, 0]

    for i in range( len( classList ) ):
        
        if classList[i] == 1:
            h2.append( hScores[ i ] )
            m2.append( mScores[ i ] )

        else:
            h1.append( hScores[ i ] )
            m1.append( mScores[ i ] )

        if hscores[i] < 0.3 and classlist[i] == 0:
            q[0] += 1

        if hscores[i] > 0.3 and classlist[i] == 0:
            q[1] += 1

        if hscores[i] < 0.3 and classlist[i] == 1:
            q[2] += 1

        if hscores[i] > 0.3 and classlist[i] == 1:
            q[3] += 1

    print("LOOK AT ME##################")
    print(q)
    print("LOOK AT ME##################")


    h1 = np.array( h1 )
    m1 = np.array( m1 )
    h2 = np.array( h2 )
    m2 = np.array( m2 )

    plt.clf()

    plt.title( titleName + "\nCorrelation: %f" % corrVal )

    plot = plt.scatter( h1, m1, s=5, marker='o', color='blue' )
    plot = plt.scatter( h2, m2, s=5, marker='d', color='red' )

    n = len( hScores )
    n1 = len( h1 )
    n2 = len( h2 )

    l1 = '%2.1f' % float(n1/n*100) + '%: Non-Disk'
    l2 = '%2.1f' % float(n2/n*100) + '%: Disk'

    plt.legend( [ l1, l2 ] )

    plt.xlabel("Human Scores")
    plt.ylabel("Machine Score")
    plt.xlim(0,1)
    plt.ylim(0.35,1)

    ax = plt.gca()
    ax.set_facecolor("xkcd:grey")

    plt.savefig( saveLoc )

# End plot creation



def classFilterPlot( hScores, mScores, classList, saveLoc, titleName ):


    h1 = []
    m1 = []


    for i in range( len( classList ) ):
        
        if classList[i] == 0:

            h1.append( hScores[ i ] )
            m1.append( mScores[ i ] )

    h1 = np.array( h1 )
    m1 = np.array( m1 )

    corrVal = np.corrcoef( h1, m1)[0,1]

    plt.clf()

    plt.title( titleName + "\nCorrelation: %f" % corrVal )

    plot = plt.scatter( h1, m1, s=5, marker='o', color='blue' )

    plt.legend( [ 'Non-Disk' ] )

    plt.xlabel("Human Scores")
    plt.ylabel("Machine Score")
    plt.xlim(0,1)
    plt.ylim(0.35,1)

    ax = plt.gca()
    ax.set_facecolor("xkcd:grey")

    plt.savefig( saveLoc )

# End plot creation


def main_old():

    endEarly = readArg()

    if printAll: print("ImgDir: %s" % imgDir )


    tImg = cv2.imread( imgDir + 'target.png' )

    '''
    tImgs = []
    nImg = np.zeros( tImg.shape)

    bLines = np.linspace( 0, 200, 7)

    for i,threshold in enumerate(bLines):
        tImgs.append( binImg( tImg, threshold ) )
        #cv2.imwrite( 'target_binary_%d.png' % i, tImgs[-1] )

        nImg += tImgs[-1]


    bMax = np.amax( nImg )

    nImg = nImg * 255 / bMax

    #tImg = binImg( tImg, 45 )

    cv2.imwrite( 'target_3.png', nImg )

    return
    '''

    if False:
        recreateData(tImg)

    if False:
        createCorrelationData( tImg )

    if False:
        createBinaryCorrelationData( tImg )

    # read all files
    hScores = readScores( 'scoreFile.txt' )
    pScores = readScores( 'pCorrScores.txt' )
    cScores = readScores( 'mCorrScores.txt' )
    bScores = readScores( 'mBinaryCorrScores.txt' )
    dScores = readScores( 'pixelDiffScores.txt' )

    allScores = []
    allScores.append( hScores )
    allScores.append( pScores )
    allScores.append( cScores )
    allScores.append( bScores )
    allScores.append( dScores )

    newScores = filterScores2( allScores, pScores, 0, 0.75 )

    hScores = newScores[0]
    pScores = newScores[1]
    cScores = newScores[2]
    bScores = newScores[3]
    dScores = newScores[4]

    print("Checking shapes")
    for l in newScores:
        print(l.shape)

    pltList = findBestWeights2( hScores, cScores, bScores, dScores )


    corrVal = pltList[:,:,0].flatten()
    w1Val = pltList[:,:,1].flatten()
    w2Val = pltList[:,:,2].flatten()

    maxCorr = np.amax( corrVal )
    iMax = np.argmax( corrVal )
    w1Max = w1Val[iMax]
    w2Max = w2Val[iMax]
    w3Max = 1 - w1Max - w2Max

    createHeatPlot( w1Val, w2Val, corrVal, 'plot_weights.png', 'Max Corr: %f \nw1: %0.2f w2: %0.2f w3 %0.2f'%(maxCorr,w1Max,w2Max, w3Max) )

    eVal, eVectors = getEigenStuff( hScores, cScores, bScores, dScores )

    wScores = w1Max*cScores + w2Max*bScores + w3Max*dScores

    createHeatPlot( hScores, wScores, pScores, 'plot_8.png', 'Weighted Methods' )

# End main

def filterScores3( allScores, colName, lowFilter, highFilter ):

    print( colName, lowFilter, highFilter )
    newScores = allScores[ allScores[ colName ] > lowFilter ]

    newScores = newScores[ newScores[ colName ] < highFilter]

    print( len( newScores) )

    print( 'Before: ', len( allScores ) )
    print( 'After : ', len( newScores ) )

    return newScores

# End filter scores



def filterScores2( allScores, fScores, lowFilter, highFilter ):

    newScores = []
    newFScores = []
    nLists = len(allScores)

    print(nLists)

    for i in allScores:
        newScores.append( [] )

    # Filter out lowly perturbed
    for i in range( len( fScores )):

        if fScores[i] > highFilter or fScores[i] < lowFilter:
            continue

        for j in range( nLists ):
            newScores[j].append( allScores[j][i] )

    finalScores = []
    for i in range( nLists ):
        finalScores.append( np.array( newScores[i] ) )

    nBefore = len( allScores[0] )
    nAfter = len( newScores[0] )

    print("Filtering...\n\tBefore: %d   \n\tAfter: %d" % ( nBefore, nAfter )) 

    pVal = ( nBefore - nAfter) / nBefore * 100

    return pVal, finalScores
# End filter scores


def binImg( imgIn, threshold ):

    cpImg = np.copy( imgIn )
    cpImg[ cpImg > threshold ] = 255
    cpImg[ cpImg <= threshold] = 0

    return cpImg
# end binary image creator


def scaleImg( imgIn ):

    imgType = imgIn.dtype
    
    nImg = cv2.resize( imgIn, None, fx = imgScale, fy=imgScale )

    
    nImg = np.uint8( nImg )
    nType = nImg.dtype

    return nImg

# end scaleImg


def createBinaryCorrelationData( tImg ):

    mImgs, iImgs = getImgs()

    pScores = createBinaryCorrPScores( mImgs, iImgs )
    mScores = createBinaryCorrMScores( mImgs, tImg )

    saveScores( pScores, 'pBinaryCorrScores.txt' )
    saveScores( mScores, 'mBinaryCorrScores.txt' )

def createBinaryCorrMScores( mImgs, tImg ):

    mScores = np.zeros( len( mImgs ))

    tImg = scaleImg( tImg )
    tB = np.sum( tImg )
    tAr = tImg.flatten()

    for i,mImg in enumerate(mImgs):

        mImg = binImg( mImg, 45 )

        # Resize image to quicken computation speed
        mImg = scaleImg( mImg )

        # flatten to 1d for correlation
        mAr = mImg.flatten()

        mVal = np.corrcoef( mAr, tAr )[0,1]

        mScores[i] = mVal

    return mScores

# end createMScores()

def createBinaryCorrPScores( mImgs, iImgs ):

    pScores = np.zeros( len( mImgs ))

    for i in range( len( mImgs )):

        mImg = mImgs[i]
        iImg = iImgs[i]

        mImg = binImg( mImg, 45 )
        iImg = binImg( iImg, 45 )

        mAr = mImg.flatten()
        iAr = iImg.flatten()

        pVal = np.corrcoef( mAr, iAr )[0,1]

        pScores[i] = pVal

    return pScores

# end create

def createCorrelationData( tImg ):

    mImgs, iImgs = getImgs()

    pScores = createCorrPScores( mImgs, iImgs )
    mScores = createCorrMScores( mImgs, tImg )

    saveScores( pScores, 'pCorrScores.txt' )
    saveScores( mScores, 'mCorrScores.txt' )


def createCorrMScores( mImgs, tImg ):

    mScores = np.zeros( len( mImgs ))

    tImg = scaleImg( tImg )
    tB = np.sum( tImg )
    tAr = tImg.flatten()

    for i,mImgList in enumerate(mImgs):

        # adjust so mean brightness matches
        mB = np.sum( mImgList )
        mImg = mImgList * ( tB / mB )

        # Resize image to quicken computation speed
        mImg = scaleImg( mImg )

        # flatten to 1d for correlation
        mAr = mImg.flatten()

        mVal = np.corrcoef( mAr, tAr )[0,1]

        mScores[i] = mVal

    return mScores

# end createMScores()





def createCorrPScores( mImgs, iImgs ):

    pScores = np.zeros( len( mImgs ))

    for i in range( len( mImgs )):

        mImg = scaleImg( mImgs[i] )
        iImg = scaleImg( iImgs[i] )

        mB = np.sum( mImg )        
        iB = np.sum( iImg )        

        iImg = iImg * (mB/iB)

        mAr = mImg.flatten()
        iAr = iImg.flatten()

        pVal = np.corrcoef( mAr, iAr )[0,1]

        pScores[i] = pVal

    return pScores

# end createPScores()
def recreateData(tImg):

    mImgs, iImgs = getImgs()

    pScores = createPScores( mImgs, iImgs )
    mScores = createMScores( mImgs, tImg )

    saveScores( pScores, 'pScores.txt' )
    saveScores( mScores, 'pixelDiffScores.txt' )



def readScores( fileName ):

    inFile = readFile( imgDir + fileName )

    nLines = len( inFile )

    scores = np.zeros( nLines )

    for i, val in enumerate(inFile):
        val = float(val.strip())
        scores[i] = val

    if everyN == 1:
        return scores

    else:

        nScores = []

        for i, val in enumerate( scores ):

            if i % everyN == 0:
                nScores.append( val )

        return np.array( nScores )


def saveScores( scores, fileName ):
    
    oFile = open( imgDir +  fileName , 'w' )

    for val in scores:
        oFile.write( "%f\n" % val )

    oFile.close()


def createMScores( mImgs, tImg ):

    mScores = np.zeros( len( mImgs ))

    tB = np.sum( tImg )

    for i,mImg in enumerate(mImgs):

        # adjust so mean brightness matches
        mB = np.sum( mImg )
        mImg = mImg * ( tB / mB )

        dImg = np.abs( mImg - tImg )
        mVal = np.sum( dImg ) / dImg.size / 255
        mVal = 1 - mVal

        # arbitrary adjustment to make score more human readable
        mVal = 10* (mVal - 0.9)

        if mVal < 0.0: mVal = 0.0

        mScores[i] = mVal

    return mScores

# end createMScores()


def createPScores( mImgs, iImgs ):

    pScores = np.zeros( len( mImgs ))

    for i in range( len( mImgs )):

        dImg = np.abs( mImgs[i] - iImgs[i] )
        pVal = np.sum( dImg ) / dImg.size / 255
        pScores[i] = pVal

    return pScores

# end createPScores()

def getImgs( iDir ):
    mImgs = []
    iImgs = []

    allFiles = listdir( iDir )
    allFiles.sort()

    n = len( allFiles )

    for i,f in enumerate(allFiles):

        try:
            rNum = int( f.split('_')[0] )
        except:
            continue
        
        if rNum % everyN == 0:

            f = imgDir + f

            if 'model' in f:
                mImgs.append( cv2.imread( f, 0 ) )

            '''
            if 'init' in f:
                iImgs.append( cv2.imread( f, 0 ) )
            '''

        print( '%.1f - %d / %d ' % ( 100*((i+1)/n), i, n ), end='\r' )


    '''
    if len(mImgs) == 0 or len(iImgs) == 0:
        print("# of model and init images 0")

    if len(mImgs) != len(iImgs):
        print("# of model and init images different")
    '''

    return mImgs, iImgs

# End getImgs

def getHumanScores():

    allFiles = listdir( imgDir )
    nRuns = int( ( len(allFiles) - 2 ) / 3 )
    hScores = np.zeros(nRuns)

    sFile = readFile( imgDir + 'scoreFile.txt' )

    for i, score in enumerate(sFile):
        hScores[i] = float(score.strip())

    if everyN == 1:
        return hScores
    else:

        nhScores = []
        for i,s in enumerate(hScores):
            if i % everyN == 0:
                nhScores.append(s)
        hScores = np.array( nhScores )
        return hScores

    '''

    for f in allFiles:

        # Skip if no human score
        if not 'info' in f: continue

        iFile = readFile( imgDir + f )
        rNum = -1
        hScore = -1.0

        for l in iFile:
            l = l.strip()

            if 'run_number' in l:
                rNum = int(l.split()[1])

            if 'human_score' in l:
                hScore = float(l.split()[1])

        if rNum == -1 or hScore == -1.0:
            print("ERROR")
            continue

        hScores[rNum] = hScore

    for i, s in enumerate(hScores):
        sFile.write("%f\n" % s )

    '''

# end getHumanScores


def readArg():

    global printAll, imgDir, everyN, plotDir, scoreDir, sdssDir, newSdss, nProc, targetLoc, filterDir

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

        elif arg == '-sdssDir':
            sdssDir = argList[i+1]
            if sdssDir[-1] != '/': sdssDir += '/'

        elif arg == '-imgDir':
            imgDir = argList[i+1]
            if imgDir[-1] != '/': imgDir += '/'

        elif arg == '-n':
            everyN = int( argList[i+1] )

        elif arg == '-plotDir':
            plotDir = argList[i+1]
            if plotDir[-1] != '/': plotDir += '/'

        elif arg == '-scoreDir':
            scoreDir = argList[i+1]
            if scoreDir[-1] != '/': scoreDir += '/'

        elif arg == '-filterDir':
            filterDir = argList[i+1]
            if filterDir[-1] != '/': filterDir += '/'

        elif arg == '-new':
            newSdss = True

        elif arg == '-pp':
            nProc = int( argList[i+1] )

    # Check if input arguments were valid
    if sdssDir == '' and imgDir == '':
        print("Please specify sdss directory")
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
        return []
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return []

    else:
        inList = list(inFile)
        inFile.close()
        return inList

# End simple read file

# Run main after declaring functions
if __name__ == '__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )


   
