'''
    Author:     Matthew Ogden
    Created:    04 Jun 2021
Description:    Implementing standardized way to extract features from images. 
'''

from os import path, listdir
from sys import exit, path as sysPath
import shutil

import numpy as np
import cv2
import pandas as pd


# For importing my useful support module
supportPath = path.abspath( path.join( __file__, "../../Support_Code/" ) )
sysPath.append( supportPath )
sysPath.append( __file__[:-21] )  # Note: Getting path this code is located. (21 comes from the length of this file name "main_machine_score.py")

import general_module as gm
import info_module as im


def test():
    print("FE: Hi!  You're in Matthew's module for extracting feature values from images.")

# global variables
test_func = None

# When creating and testing new functions from outside of module (Ex. Jupyter)
def set_test_func( inFuncLink ):
    global test_func
    test_func = inFuncLink
    
    
def main( arg ):

    if arg.printAll:

        test()
        gm.test()
        im.test()
        arg.printArg()
        
    # End printAll
    

def main_fe_run( rInfo = None, arg = gm.inArgClass() ):
    
    # extract variables
    printBase=arg.printBase
    printAll=arg.printAll
    
    if rInfo == None:
        rInfo = arg.get('rInfo')
        
    scoreParams = arg.get('scoreParams',None)

    if printBase:
        print("FE: image_creator_run")
        #arg.printArg()
    
    if rInfo == None:
        rDir = arg.get('runDir',None)
        rInfo = im.run_info_class( runDir=rDir, printBase = False, printAll=printAll )

    if printBase:
        print('FE: rInfo.status: ', rInfo.status )

    if rInfo.status == False:
        if printBase: print('FE: WARNING:\n\t - rInfo status not good. Exiting...' )
        return
    
    if scoreParams == None:
        if printBase: print("FE: WARNING: Please provide score parameters")
        return
    
    elif printAll:
        if printBase: print("FE: given parameters: %d"%len(scoreParams))
    
    # Extract types of features being extracted
    featParams = {}
    featTypes = []
    
    for sKey in scoreParams:
        
        featArg = scoreParams[sKey].get('featArg',None)
        if featArg == None:    continue
            
        fType = featArg.get('type',None)
        if fType == None:    continue
            
        # wndchrm extraction is only done once for many images.  Only queue once.
        if 'wndchrm' in fType and fType in featTypes:    continue
            
        if sKey not in featParams:
            featParams[sKey] = scoreParams[sKey]
            featTypes.append(fType)
            
    # Loop through types of features
    n = len(featParams)
    
    if n == 0 and printBase: print("FE: WARNING: No Features to extract.")
    
    for i, pKey in enumerate(featParams):
        if printBase: print( 'FE_LOOP: %3d / %3d: %s' % (i,n, pKey), end='\r' )
        
        featType = featParams[pKey]['featArg'].get('type',None)
        
        if printAll: print( "FE: name/type: %s/%s"% (pKey,featType) )
        
        if featType == 'wndchrm_all':
            wndchrm_run_all( rInfo, arg )
           
        else:
            if printBase: print("FE: featType '%s' unavailable"%str(featType))
            continue
                
        
        
    if printBase: print( 'FE_LOOP: %3d / %3d COMPLETE!' % (n,n), )
            
# End main image creator run

def wndchrm_run_all( rInfo, arg ):
    from shutil import which
    from os import remove, listdir
    import subprocess 
    
    # extract variables
    printBase=arg.printBase
    printAll=arg.printAll
    
    if printAll: print("FE: wndchrm_run_all: Notebook development")
            
    wndDir = rInfo.get('wndDir')
    allLoc = wndDir + 'wndchrm_all.fit'
    wndchrm_command = [ 'wndchrm', 'train', '-s', wndDir, allLoc ]
    
    if arg.get('overWrite',False):
        wndchrm_command.insert(3, '-o')    
    
    # If None, assumed wndchrm not installed/in public executable path.
    wndchrm_path = which('wndchrm')
    if wndchrm_path == None:
        print("FE: WARNING: WNDCHRM executable not found")
        return 
    
    # If overwriting, remove current files
    if arg.get('overWrite',False):
        if printAll: gm.tabprint("Removing .sig files  WNDCHRM dir")
        
        if gm.validPath( allLoc ): remove( allLoc )
        
        wndFiles = listdir( wndDir )
        for name in wndFiles:
            if '.sig' in name:
                fLoc = wndDir + name
                remove( fLoc )
    
    if printAll:
        print("FE: wndchrm_run_all: Executing:")
        gm.tabprint( '$ %s' % ' '.join(wndchrm_command))
        
    cmd_results = subprocess.run(wndchrm_command)
    
    if printAll: 
        print("FE: wndchrm_run_all: Execution Done")
        gm.tabprint("Collecting wndchrm all data")
    
    if gm.validPath( allLoc ) == None:
        if printAll: 
            print("WARNING: FE: Failed to create wndchrm_run_all.fit")
        return
    
    else:
        reorganize_wndchrm_run_data( rInfo, arg )   
    

def reorganize_wndchrm_run_data( rInfo, arg ):
    
    printBase = arg.printBase
    printAll = arg.printAll
    allLoc = rInfo.get('wndFitLoc')
    csvLoc = rInfo.get('wndAllLoc')
    
    if printAll: print("FE: reorganize_wndchrm_run_data")
    
    # Read file to get prelimary data
    fitFileList = gm.readFile( allLoc, stripLine = True )
    
    if type(fitFileList) == type(None) or len(fitFileList) < 3:
        if printAll: 
            print("WARNING: FE: reorganize_wndchrm_run_data: Failed to read wndchrm .fit file")
            gm.tabprint('%s'%allLoc)
        return
                
    nFeat = int( fitFileList[1] )
    nImg = int( fitFileList[2] )
    headerOffset = 3
    imageOffset = headerOffset + nFeat + 1
    
    # Grab header names 
    wndHeaders = [ fitFileList[i] for i in range( headerOffset, nFeat + headerOffset ) ]
        
    if printAll: 
        gm.tabprint( "number of features: %d" % nFeat )
        gm.tabprint( "number of images: %d" % nImg )
        gm.tabprint( "feature labels head: [ %s, ... ]" % ', '.join(wndHeaders[0:2]) )
        gm.tabprint( "feature labels tail: [ ..., %s ]" % ', '.join(wndHeaders[-2:]) )
    
    # Use pandas to read feature values and image paths at end of file
    fitDF = pd.read_csv( allLoc, skiprows=list(range(0,1062)), sep='\s+', index_col=False, header=None, names=wndHeaders )
    
    # Insert columns I want added
    fitDF.insert( 0,'run_id', rInfo.get('run_id') )
    fitDF.insert( 1,'image_name', None )
    fitDF.insert( 2,'zoo_merger_score', rInfo.get('zoo_merger_score') )
    
    # Loop through rows, extract image name and place in new image_name column
    for i in np.arange(0,nImg*2,2):
        imgPath = fitDF.loc[ i+1, wndHeaders[0] ] 
        imgName = imgPath.split('/')[-1].split('.')[0] 
        fitDF.at[i,'image_name'] = imgName
    
    # Extract rows with feat values
    featDF = fitDF[ fitDF.index %2 == 0 ]
    
    if printAll:
        gm.tabprint("Extracted feature values dataframe")
        headers = list( featDF.columns )
        
        gm.tabprint( " head: [ %s, ... ]" % ', '.join(headers[0:4]) )
        gm.tabprint( "feature labels tail: [ ..., %s ]" % ', '.join(headers[-2:]) )
        gm.tabprint("Saving wndchrm_all data at: %s" % csvLoc )
    
    # Save extracted and organized data
    featDF.to_csv( csvLoc, index=False )
    

def target_collect_wndchrm_all_raw( args, tInfo ):

    printAll = args.printAll
    printBase = args.printBase 
    
    if printAll: print("FE: target_collect_wndchrm_all_raw")
        
    if tInfo == None:
        print("WARNING: FE: target_collect_wndchrm_all_raw: target info not given")
        return
    
    # Run directory list to iterate through
    #runDirList = tInfo.iter_runs( n = 3 )
    runDirList = tInfo.iter_runs( )
    n = len(runDirList)
    
    # Initiate a single run to get path to wndchrm_all.csv file in run directory
    rDir = runDirList[0]
    rInfo = im.run_info_class( runDir = rDir )
    rwLoc = rInfo.wndAllLoc
    rwPath = rwLoc.replace( rDir, '' )
    
    # Go through directories and collect wndchrm DataFrames.
    runDFList = []
    
    for i, rDir in enumerate(runDirList):
        if printBase: print('FE: target_collect_wndchrm_all_raw: Loop: %4d / %4d        ' % (i,n), end='\r' )
            
        rwLoc = gm.validPath( rDir + rwPath )
        if rwLoc != None:
            runDFList.append( pd.read_csv( rwLoc ) )
    
    if printBase:    print('FE: target_collect_wndchrm_all_raw: Loop: %4d / %4d: Complete!' % (n,n) )
    if printAll:    gm.tabprint( "Read %d of %d wndchrm run files." % ( len(runDFList),  len(runDirList), ) )

    # Combine all the run DataFrames into one.
    targetDF = pd.concat( runDFList, )
    
    if printAll:
        gm.tabprint( "Final WNDCHRM DataFrame")
        gm.tabprint( "Shape: %s" % str( targetDF.shape ) )
        gm.tabprint( "Unique Runs: %d" % len(targetDF['run_id'].unique()))
        gm.tabprint( "Unique Image Names: %d" % len(targetDF['image_name'].unique()))
        gm.tabprint( "Header Length: %d" % len( targetDF.columns ) )
        gm.tabprint( "Headers head: [ %s, ... ]" % ', '.join( list( targetDF.columns )[0:3] ) )
        gm.tabprint( "Headers tail: [ ... , %s ]" % ', '.join( list( targetDF.columns )[-2:] ) )
        
    #print( targetDF )
    
    twLoc = tInfo.get( 'wndRunRawDFLoc', None )
    if twLoc != None:
        targetDF.to_pickle( twLoc )


def wndchrm_target_all( arg, tInfo ):
    
    from shutil import which
    from os import remove, listdir
    import subprocess 
    
    # extract variables
    printBase=arg.printBase
    printAll=arg.printAll
    
    if printAll: print("FE: wndchrm_target_all: Notebook development")
            
    # Get useful target variables
    wndDir = tInfo.get('wndDir')
    fitLoc = tInfo.get('wndTargetRawFitLoc')
    
    if wndDir == None or fitLoc == None:
        if printBase:
            print("WARNING: FE: wndchrm_target_all: target info variables not found")
            gm.tabprint('wndDir: %s' % wndDir )
            gm.tabprint('fitLoc: %s' % fitLoc )
        return
    
    # Build executable command
    wndchrm_command = [ 'wndchrm', 'train', '-s', wndDir, fitLoc ]
    
    if arg.get('overWrite',False):
        wndchrm_command.insert(3, '-o')
    
    # Check if WNDCHRM is installed or available
    # If None, assumed wndchrm not installed/in public executable path.
    wndchrm_path = which('wndchrm')
    if wndchrm_path == None:
        print("WARNING: FE: wndchrm_target_all: WNDCHRM executable not found")
        return 
    
    # Execute WNDCHRM 
    if printAll: 
        print("FE: wndchrm_target_all: Executing:") 
        gm.tabprint( '$ %s' % ' '.join(wndchrm_command)) 
        
    cmd_results = subprocess.run(wndchrm_command) 
    
    if printAll: 
        print("FE: wndchrm_target_all: Execution Done") 
    
    if gm.validPath( fitLoc ) == None:
        if printAll: 
            print("WARNING: FE: wndchrm_target_all: Failed to create wndchrm_target_all.fit")
        return
    
    else:   
        if printBase: gm.tabprint("Collecting wndchrm all data")
        reorganize_wndchrm_target_data( arg, tInfo )   


def reorganize_wndchrm_target_data( arg, tInfo,  ):

    printBase = arg.printBase
    printAll = arg.printAll
    fitLoc = tInfo.get('wndTargetRawFitLoc')
    csvLoc = tInfo.get('wndTargetRawCSVLoc')
    dfLoc = tInfo.get('wndTargetRawDFLoc')
    
    if printAll: print("FE: reorganize_wndchrm_target_data")
    
    # Read file to get prelimary data
    fitFileList = gm.readFile( fitLoc, stripLine = True )
    
    if type(fitFileList) == type(None) or len(fitFileList) < 3:
        if printAll: 
            print("WARNING: FE: reorganize_wndchrm_target_data: Failed to read wndchrm .fit file" )
            gm.tabprint('%s'%fitLoc)
        return
                
    nFeat = int( fitFileList[1] )
    nImg = int( fitFileList[2] )
    headerOffset = 3
    imageOffset = headerOffset + nFeat + 1
    
    # Grab header names 
    wndHeaders = [ fitFileList[i] for i in range( headerOffset, nFeat + headerOffset ) ]
        
    if printAll: 
        gm.tabprint( "number of features: %d" % nFeat )
        gm.tabprint( "number of images: %d" % nImg )
        gm.tabprint( "feature labels head: [ %s, ... ]" % ', '.join(wndHeaders[0:2]) )
        gm.tabprint( "feature labels tail: [ ..., %s ]" % ', '.join(wndHeaders[-2:]) )
    
    # Use pandas to read feature values and image paths at end of file
    fitDF = pd.read_csv( fitLoc, skiprows=list(range(0,1062)), sep='\s+', index_col=False, header=None, names=wndHeaders )
    
    # Insert columns I want added
    fitDF.insert( 0,'target_id', '"%s"' % str( tInfo.get('target_id') ) )
    fitDF.insert( 1,'image_name', None )
    fitDF.insert( 2,'zoo_merger_score', 1.0 )
    
    # Loop through rows, extract image name and place in new image_name column
    for i in np.arange(0,nImg*2,2):
        imgPath = fitDF.loc[ i+1, wndHeaders[0] ] 
        imgName = imgPath.split('/')[-1].split('.')[0]
        fitDF.at[i,'image_name'] = imgName
    
    # Extract rows with feat values
    featDF = fitDF[ fitDF.index %2 == 0 ]
    
    if printAll:
        gm.tabprint("Extracted feature values dataframe")
        headers = list( featDF.columns )
        
        gm.tabprint( " head: [ %s, ... ]" % ', '.join(headers[0:4]) )
        gm.tabprint( "feature labels tail: [ ..., %s ]" % ', '.join(headers[-2:]) )
        gm.tabprint( "Saving wndchrm data at: %s" % csvLoc )
        gm.tabprint( "... and at: %s" % dfLoc )
    
    # Save extracted and organized data
    featDF.to_csv( csvLoc, index=False )
    
    # Save in Pandas Dataframe form as well.
    tInfo.saveWndchrmDF( featDF, 'targets_raw' )

# Function to collect model wndchrm values and normalize them.
def target_wndchrm_create_norm_scaler( args, tInfo ):
    
    printAll = args.printAll
    printBase = args.printBase
    normName = args.get('normName', None)
    normLoc = args.get('normLoc', None)
    normDict = None
    
    if normName == None and normLoc == None:
        if printBase: 
            print("WARNING: FE: target_wndchrm_create_norm_scaler")
            gm.tabprint("Please provide `-normName file_name`")
            gm.tabprint("Or provice `-normLoc path/to/file.json`")
        return
    
    elif type(normName) == type('string'):
        normDict = tInfo.readWndchrmNorm( normName )
        if printAll:
            gm.tabprint("Read from target_info saved norms")
            gm.pprint(normDict)
    elif type(normLoc) == type('string'):
        normDict = gm.readJson( normLoc )
    
    if normDict == None:
        if printBase: 
            print("WARNING: FE: target_wndchrm_create_norm_scaler")
            gm.tabprint("Please provide a -normFeats name")
            gm.tabprint("Or provide a -normDict dictionary")
        return
        
    
    if printBase:
        print( "FE: normalize_target_wnchrm." )
        gm.tabprint( "tID: %s" % tInfo.get( 'target_id' ) )
        gm.tabprint( "Normalization Parameters")
        gm.pprint( normDict )
        
    # Useful variables
    
    infoHeaders =tInfo.wndchrmInfoHeaders
    runsRaw = tInfo.readWndchrmDF( 'all_runs_raw')
    targetRaw = tInfo.readWndchrmDF( 'targets_raw')
    
    
    allRawDF = pd.concat( [ runsRaw, targetRaw ] )
    allInfoDF = allRawDF[ infoHeaders ]
    
    if printAll:
        gm.tabprint( 'Target Shape: %s' % str( targetRaw.shape ) )
        gm.tabprint( 'Runs Shape: %s' % str( runsRaw.shape ) )
        gm.tabprint( 'All Raw Shape: %s' % str( allRawDF.shape ) )
    
    trainDF = None
    # Combine top N models and target 
    if normDict.get( 'top_models', None) != None:
        
        topN = int( normDict['top_models'] )
        
        if printAll:
            gm.tabprint( 'top_models: %d' % topN )
        
        # Grab names of top N models
        # assume run_id is listed in alphanumerical order of best.  This will likely change later
        runIDList = list(runsRaw['run_id'].unique())        
        if len( runIDList ) >= 500:
            runIDList = runIDList[0:topN]        
            
        topRunRaw = runsRaw[ runsRaw['run_id'].isin(runIDList) ]        
        trainDF = pd.concat( [ topRunRaw, targetRaw ] )
        
        if printAll: gm.tabprint("Shape top N: %s" % str( trainDF.shape ) )
    
    # Combine all models and target.
    else:
        trainDF = pd.concat( [ runsRaw, targetRaw ] )
    
    image_group_name = normDict.get('image_group',None)
    image_group = None
    if image_group_name != None:
        if printAll: gm.tabprint('Score Params for Filtering: %s' % image_group_name )
        image_group = tInfo.readScoreParam( image_group_name )
        if image_group != None:
            if printAll: gm.tabprint('Read filter params: %s' % image_group_name )
        else:
            if printBase: 
                print("WARNING: FE: target_wndchrm_create_norm_scaler")
                gm.tabprint('Failed to find score params: %s' % image_group_name )
            return
            
        
    elif printAll: gm.tabprint('Not reading score parameters for filtering' )
    
    # Extract only images from image_group
    if image_group != None:
        imgNameList = [ image_group[pKey]['imgArg']['name'] for pKey in image_group ]
        trainDF = trainDF[ trainDF['image_name'].isin(imgNameList) ]
        
        if printAll:    gm.tabprint("Filtered Images: %s" % str( trainDF.shape ) )
    
    if printAll:    gm.tabprint('All Raw Shape: %s'%str(trainDF.shape))
    
    # Headers not in info headers are assumed feature value header names
    featHeaders = [ h for h in trainDF if h not in infoHeaders ]

    # Remove any rows with nan values in feature headers
    trainDF = trainDF[ ~trainDF[ featHeaders ].isin([np.nan, np.inf, -np.inf]).any(1)]
    if printAll:    gm.tabprint('filtered out: %s' % str(trainDF.shape))
        
    # Seperate information columns from feature value columns being normalized.
    trainRaw = trainDF.drop( infoHeaders, axis=1 ).values
    
    if printAll:
        gm.tabprint('info Headers: [ %s ]' % ', '.join(infoHeaders))
        gm.tabprint('feat value Shape: %s' % str(trainRaw.shape ) )
    
    # Determine what method to use for normalizing data and create a scaler model
    normMethod = normDict.get( 'normalization_method', 'sklearn_StandardScaler' )
    
    if printAll:    gm.tabprint("Creating Scaler: %s" % normMethod )
        
    if normMethod == 'sklearn_StandardScaler':
        from sklearn.preprocessing import StandardScaler 
        scaler = StandardScaler()
        featScaled = scaler.fit_transform( trainRaw )
    
    else:
        print("WARNING: FE: normalize_target_wndchrm")
        gm.tabprint("Normalization Method Not Found: %s" % normMethod )
        return
    
    if printAll:    gm.tabprint("Scaler Complete. Saving...")

    # Have target info save scaler file
    tInfo.saveWndchrmScaler( scaler, normDict.get('name', 'default') )
    
    
    # Apply new scaler to all the feature data      
    if printAll: gm.tabprint("Applying scaler to all data")    
        
    featRawValues = allRawDF.drop( infoHeaders, axis=1 ).values  
    
    if printAll: 
        gm.tabprint("Raw Feat Values Shape: %s" % str( featRawValues.shape ) )
        gm.tabprint("Transforming Raw Feat Values..." )
    
    featScaledValues = scaler.transform( featRawValues )
    
    if printAll: 
        gm.tabprint("Transform Complete" )
        
    
    scaledDF = pd.DataFrame( featScaledValues, columns = featHeaders )
    
    #pd.concat([df1, df4.reindex(df1.index)], axis=1)
    
    if printAll: 
        gm.tabprint("Scaled Feat Values Shape: %s" % str( scaledDF.shape ))
        gm.tabprint("Scaled Info Values Shape: %s" % str( allInfoDF.shape ))
    
    # In
    for i,h in enumerate( infoHeaders ):
        scaledDF.insert(i, h, allInfoDF[h].values )
    
    if printAll: 
        gm.tabprint("Scaled DF Shape: %s" % str( scaledDF.shape ))
        gm.tabprint("Saving scaled DF...")
    
    tInfo.saveWndchrmDF( scaledDF, normDict['name'] )
    
    if printBase:  print("FE: target_wndchrm_create_norm_scaler: Complete")
               

def variance_analysis( X, args, tInfo, ):
    
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import VarianceThreshold    
    
    printAll = args.printAll
    normName = args.normName
    plotDir = tInfo.plotDir
    
    if printAll:
        print("FE: variance analysis")
        gm.tabprint("X shape: %s" % str(X.shape))
    
    # hardcoded variables    
    n = 51
    plotX = np.linspace( 0.5, 1, n )
    plotY = np.zeros(n)
    
    for i, p in enumerate( plotX ):
        var = VarianceThreshold( threshold=( p * (1 - p) ) )
        varX = var.fit_transform(X)
        plotY[i] = varX.shape[1]
        if printAll: gm.tabprint( '%2d/%2d' % (i,n), end='\r' )
    
    if printAll: gm.tabprint( '%2d/%2d Complete' % (n,n), end='\r' )
    
    plt.plot( plotX, plotY )
    plt.ylabel( 'Retained Features' )
    plt.xlabel( 'Variance Threshold: p' )
    plt.title( 'Variance Threshold:\n%s' % normName )
    plt.savefig( plotDir + '%s_variance_threshold.png' % ( normName ))
    
# End variance threshold

def target_apply_filter( args=None, tInfo=None ):
    
    if args == None: args = gm.inArgClass()
    printBase = args.printBase
    printAll = args.printAll
    param_name = args.get('paramName', None)
    param_filter = args.get('scoreParams',None)
    norm_name = args.get('normName',None)
    
    if printBase: print("FE: target_apply_filter:")
        
    if tInfo == None:
        if printBase: print("WARNING: FE: target_apply_filter: Please provide target_info_class")
        return
    
    if param_filter == None:
        if printBase: print("WARNING: FE: target_apply_filter: Please provide target_info_class")
        return
    
    # Use score params as a filter for images
    imgList = None
    if param_filter != None:
        
        if printAll: gm.tabprint("Gathering images names to filter")
        
        imgList = []
        
        # go through parameters and grab image names for filtering
        for pkey in param_filter:
            imgName = param_filter[pkey]['imgArg']['name']
            if imgName not in imgList:
                imgList.append( imgName )
        
        if printAll: gm.tabprint("Found %d unique images" % len(imgList) )
        
        if len( imgList ) == 0:
            print("WARNING: FE: target_apply_filter: No images found")
            return
        
    # Read starting dataframe to filter
    startDF = None
    
    if norm_name == None:        
        if printAll: gm.tabprint("Reading Raw WNDCHRM Features")
        rawRunDF = pd.read_pickle( tInfo.wndRunRawDFLoc )
        rawTargetDF = pd.read_pickle( tInfo.wndTargetRawDFLoc )
        
        startDF = pd.concat( [ rawRunDF, rawTargetDF ] )
        
        if printAll:
            gm.tabprint("Raw Dataframes Read")
            gm.tabprint( "\tTarget shape: %s" % str( rawTargetDF.shape ) )
            gm.tabprint( "\tAll runs shape: %s" % str( rawRunDF.shape ) )
            gm.tabprint( "\tCombined shape: %s" % str( startDF.shape ) )
        
        del rawRunDF
        del rawTargetDF
    
    if norm_name != None:
        if printAll: gm.tabprint("Reading Normalized WNCHRM Features: %s" % norm_name )
        startDF = tInfo.readWndchrmDF( norm_name )
        if printAll: gm.tabprint("\tStarting Shape: %s" % str(startDF.shape) )
    
    if type(startDF) == type(None):
        if printBase: print("WARNING: FE: target_apply_filter: No Dataframe to filter")
        return
        

    if printAll: gm.tabprint("Filtering scores by image")

    filterDF = startDF[ startDF['image_name'].isin( imgList ) ]

    if printAll:
        gm.tabprint("\tFiltered shape: %s" % str( filterDF.shape ) )
    
    return filterDF     
        
# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
