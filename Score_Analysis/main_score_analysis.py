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
        
def basicHeatPlot( s1, s2, s3, titleName=None, ):

    # Specify heat colors
    cMap = plt.cm.get_cmap('RdYlBu_r')
    
    # Add points and trendline to plot
    plot = plt.scatter( s1['scores'], s2['scores'], c=s3['scores'], s=5, cmap=cMap ) 
    
    # Add colorbar
    cBar = plt.colorbar(plot)
    
    # Setup plot
    plt.xlabel(s1['name'])
    plt.ylabel(s2['name'])
    cBar.set_label( s3['name'] )
    
    ax = plt.gca()
    #ax.set_facecolor("xkcd:grey")

    if titleName != None:
        plt.set_title( '%s' % titleName)

    return plot


def getNamedPlot( scores = None, sName = 'new_score', pertName = 'base_perturbation', printAll = False ):
        
    # Grab dataframe of human, machine, and perturbation scores, drop invalid rows. 
    hmScores = scores[['zoo_merger_score',sName,pertName]].dropna()
    
    if printAll:
        print("SA: valid scores: %d"%len(hmScores))
        print(hmScores)
    
    # Seperate human and machine scores        
    hScores = hmScores['zoo_merger_score'].values
    mScores = hmScores[sName].values
    pScores = hmScores[pertName].values

    corr = np.corrcoef( hScores, mScores )[0,1]

    hs = { 'scores':hScores, 'name':'zoo_merger_score' }
    ms = { 'scores':mScores, 'name':sName }   
    ps = { 'scores':pScores, 'name':pertName }

    title = '%s:' % (sName) 
    title += '\nCorr: %.4f' % corr
    
    fig, ax = plt.subplots(1)

    basicHeatSubPlot( fig, ax, hs, ms, ps, titleName=title)    
    return ax

def basicHeatSubPlot( fig, ax, s1, s2, s3, titleName='', plotLoc = None ):
    
    from matplotlib.cm import ScalarMappable

    # Specify heat colors
    cMap = plt.cm.get_cmap('RdYlBu_r')
    
    # Add points and trendline to plot
    plot = ax.scatter( s1['scores'], s2['scores'], c=s3['scores'], s=15, cmap=cMap )
    
    # Add colorbar
    cb = fig.colorbar( plot, ax=ax)
    
    # Setup plot
    ax.set_xlabel(s1['name'])
    ax.set_ylabel(s2['name'])
    cb.set_label( s3['name'] )
    ax.set_facecolor("gray")
    ax.set_title( '%s' % titleName)

    return ax

def basicSubPlot( ax, s1, s2, titleName=None, plotLoc = None ):


    # Add points and trendline to plot
    ax.scatter( s1['scores'], s2['scores'] )

    # Setup plot
    ax.set( xlabel = s1['name'], ylabel = s2['name'] )

    if titleName != None:
        ax.set_title( '%s' % titleName)

    return ax


def target_report_1( tInfo=None, scoreLoc = None, plotLoc = None):
    
    from copy import deepcopy
    
    if tInfo != None:
        scores = tInfo.getScores()
    elif scoreLoc != None:
        scores = gm.getScores(scoreLoc)
    
    scoreHeaders = list( scores.columns )
    
    msNames = deepcopy(scoreHeaders)
    msNames.pop(0)
    msNames.pop(0)
    
    n = len(msNames)
    
    if n == 0:
        print('NNOOOOOOO')
    else:
        print("YAYYYY")
    
    fig, axs = plt.subplots( n+2, figsize=(7,5*(n+2)) )    
    
    if tInfo != None:
        # Show target image
        tImg = tInfo.getTargetImage( 'zoo' )
        axs[0].imshow(tImg, cmap='gray')
        axs[0].set_title("Target Image")

        # Show a model image
        rId = 'r00001'
        runArg = gm.inArgClass()
        runArg.setArg("printBase",False)
        rInfo = tInfo.getRunInfo(rID=rId, rArg = runArg)
        rImg = rInfo.getModelImg( )
        rN = scores.shape[0]

        if type(rImg) != type(None):
            axs[1].imshow(rImg, cmap='gray')
            axs[1].set_title('Model Image')

    # Get human scores
    hScores = scores['zoo_merger_score']
    
    # Go through and plot stuff
    for i, sName in enumerate(msNames):     
        hmScores = scores[['zoo_merger_score',sName]].dropna()
        hScores = hmScores['zoo_merger_score'].values
        mScores = hmScores[sName].values
        corr = np.corrcoef( hScores, mScores )[0,1]
        print('%s: %4d/%4d' % (sName,hmScores.shape[0],rN), corr)
        
        hs = { 'scores':hScores, 'name':'zoo_merger_score' }
        ms = { 'scores':mScores, 'name':sName }   
                
        title = '%s: %4d/%4d ' % (sName,hmScores.shape[0],rN) 
        title += '\nCorr: %.4f' % corr
        
        sa.basicSubPlot( axs[i+2], hs, ms, titleName=title)
        
    plt.tight_layout()
    
    if tInfo != None:
        plotLoc = tInfo.plotDir + 'basic_target_report.pdf'
        print('PLotLoc: ', plotLoc)
        fig.savefig(plotLoc, bbox_inches='tight')
    
         

def target_report_2( tInfo=None, printBase = True, printAll = False ):
    
    from copy import deepcopy
    
    scores = tInfo.getScores()
    scoreHeaders = list( scores.columns )
    allParams = tInfo.get('score_parameters')

    targKeys = []
    pertKeys = []
    unknKeys = []
    
    for pKey in allParams:
        
        if allParams[pKey]['scoreType'] == 'target' and pKey in scores:
            targKeys.append(pKey)
            
        elif allParams[pKey]['scoreType'] == 'perturbation' and pKey in scores:
            pertKeys.append(pKey)
        else:
            unknKeys.append(pKey)
    
    print("Target Machine Scores Found:", len(targKeys))
    for name in targKeys: gm.tabprint(name)
        
    print("Perturbation Scores Found:", len(pertKeys))
    for name in pertKeys: gm.tabprint(name)
        
    print("Unknown Scores Found:", len(unknKeys))
    for name in unknKeys: gm.tabprint(name)
        
    pertName = 'base_perturbation'
    if printBase: print("SA: target report 2: Hard coded perturbation: ",pertName)
        
    if not pertName in scores:
        print("SA: target report 2: Base Perturbation Not found:")
        gm.tabprint('target: %s'%tInfo.get('target_id'))
        gm.tabprint("perturb: %s"%pertName)
        return None
        
    n = len(targKeys)
    
    if n == 0:
        return None
    
    fig, axs = plt.subplots( n+2, figsize=(9,7*(n+2)) )  
    
        
    # Show target image
    tImg = tInfo.getTargetImage( 'zoo' )
    axs[0].imshow(tImg, cmap='gray')
    axs[0].set_title("Target Image")

    # Show a model image
    rId = 'r00001'
    runArg = gm.inArgClass()
    runArg.setArg("printBase",False)
    rInfo = tInfo.getRunInfo(rID=rId, rArg = runArg)
    rImg = rInfo.getModelImg( )
    rN = scores.shape[0]

    if type(rImg) != type(None):
        axs[1].imshow(rImg, cmap='gray')
        axs[1].set_title('Model Image')

    # Go through and plot Target Machine Scores
    for i, sName in enumerate(targKeys):     
        
        hmScores = scores[['zoo_merger_score',sName,pertName]].dropna()
        hScores = hmScores['zoo_merger_score'].values
        mScores = hmScores[sName].values
        pScores = hmScores[pertName].values
        
        corr = np.corrcoef( hScores, mScores )[0,1]
        
        print('%s: %4d/%4d' % (sName,hmScores.shape[0],rN), corr)
        
        hs = { 'scores':hScores, 'name':'zoo_merger_score' }
        ms = { 'scores':mScores, 'name':sName }   
        ps = { 'scores':pScores, 'name':pertName }
                
        title = '%s: %4d/%4d ' % (sName,hmScores.shape[0],rN) 
        title += '\nCorr: %.4f' % corr
        
        basicHeatSubPlot( fig, axs[i+2], hs, ms, ps, titleName=title)        

    plt.tight_layout()
    
    plotLoc = tInfo.plotDir + 'report_2_'+tInfo.get('target_id')+'.pdf'
    print('PLotLoc: ', plotLoc)
    fig.savefig(plotLoc, bbox_inches='tight')



# Run main after declaring functions
if __name__ == '__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )


   
