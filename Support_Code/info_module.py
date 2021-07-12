'''
    Author:	 Matthew Ogden
    Created:	21 Feb 2020
Description:	Module for accessing information and files stored on disk.
'''

from os import path, listdir
import json
from copy import deepcopy

import pandas as pd
import numpy as np
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD    
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

from sys import path as sysPath
sysPath.append('../')


from pprint import PrettyPrinter
pp = PrettyPrinter(width=41, compact=True)
pprint = pp.pprint

# For loading in Matt's general purpose python libraries
import general_module as gm

def test():
    print("IM: Hi!  You're in Matthew's information module for SPAM")
# End test print

def main(arg):

    print("IM: Hi!  You're in Matthew Information module's main function.")

    # Print stuff
    if arg.printAll:
        arg.printArg()
        gm.test()

    if arg.runDir != None:

        rInfo = run_info_class( \
                runDir = arg.runDir, \
                printAll = True, \
                newInfo = arg.get('newInfo', None ), \
                newRun = arg.get( 'newRun', False ), \
                )

        if rInfo.status == False:
            return

        rInfo.printInfo()
        if getattr( arg, 'updateInfo', False ):
            rInfo.updateInfo()

        rInfo.printScores()

    if arg.targetDir != None:

        tInfo = target_info_class( \
                targetDir = arg.targetDir, \
                printAll = arg.printAll, \
                newInfo = arg.get('newInfo', False ), \
            )

        if tInfo.status:
            print("IM: target_info_class good")


# End main

class run_info_class:

    rDict = None	# Primary dict of information contained in info.json
    initDict = None	# State of json upon initial reading.
    baseDict = None

    status = False  # State of this class.  For initiating.
    
    # Main directories
    runDir = None
    
    ptsDir = None
    imgDir = None
    miscDir = None
    wndDir = None
    tmpDir = None

    # Useful files in directory
    infoLoc = None
    baseLoc = None

    wndFitLoc = None
    wndAllLoc = None
    
    tLink = None



    def __init__( self, runDir=None, rArg = gm.inArgClass(), \
                 printBase=None, printAll=None, ):

        # get arguments
        if printBase == None:  self.printBase = rArg.printBase
        else:  self.printBase = printBase
        
        if printAll == None:  self.printAll = rArg.printAll
        else:  self.printAll = printAll
            
        if runDir != None:
            rArg.runDir = runDir
        
        if rArg.get('tInfo',None) != None: 
            self.tInfo = rArg.tInfo

        # Avoiding confusing things
        if self.printAll: self.printBase = True

        if self.printAll: 
            print("IM: run_info_class.__init__")
            print("\t - runDir: " , runDir )


        # initialize directory structure for run
        dirGood = self.initRunDir( rArg )

        if not dirGood:
            if self.printBase: 
                print("WARNING: IM: Run_info_class: Directory not set up properly")
                gm.tabprint('runDir: %s' % rArg.runDir )
            self.status = False
            return
                 
        # Read info file
        if path.exists( self.infoLoc ):

            if self.printAll: print('\t - Reading Info file.')
            with open( self.infoLoc, 'r' ) as iFile:
                self.rDict = json.load( iFile )

        if self.rDict != None:
            self.status = True

        if self.printAll: 
            print("IM: Run.__init__: Initalized: %s" % self.status)

    # end __init__
    
    
    def initRunDir( self, rArg ):
        
        self.runDir = gm.validPath( rArg.runDir )
            
        # Double check if run directory is valid
        if self.runDir == None:
            if self.printBase:
                print("IM: WARNING: run_info_class.__init__: Invalid run dir")
                tabprint('runDir: - %s' % rArg.runDir )
            return False

        # Print stuff
        if self.printAll:
            print("IM: run.initRunDir")
            print("\t - runDir: %s" % rArg.runDir )

        # Save base directory
        if self.runDir[-1] != '/': self.runDir += '/'

        # Hard code location of main objects
        self.ptsDir = self.runDir + 'particle_files/'
        self.imgDir = self.runDir + 'model_images/'
        self.miscDir = self.runDir + 'misc_images/'
        self.wndDir = self.runDir + 'wndchrm_files/'
        self.tmpDir = self.runDir + 'tmp/'
        self.infoLoc = self.runDir + 'info.json'
        self.baseLoc = self.runDir + 'base_info.json'
        
        self.wndFitLoc = self.wndDir + 'wndchrm_all.fit'
        self.wndAllLoc = self.wndDir + 'wndchrm_all.csv'
    
        # If newInfo or newBase
        if rArg.get('newInfo',False):
            self.newRunSetup(rArg)

        # Print stuff if needed
        if self.printAll:
            print("\t -  runDir: (%s) %s" % ( path.exists( self.runDir ), self.runDir ) )
            print("\t -  ptsDir: (%s) %s" % ( path.exists( self.ptsDir ), self.ptsDir ) )
            print("\t -  imgDir: (%s) %s" % ( path.exists( self.imgDir ), self.imgDir ) )
            print("\t - miscDir: (%s) %s" % ( path.exists( self.miscDir ), self.miscDir ) )
            print("\t - wndDir: (%s) %s" % ( path.exists( self.wndDir ), self.wndDir ) )
            print("\t - infoLoc: (%s) %s" % ( path.exists( self.infoLoc ), self.infoLoc ) )
            print("\t - baseLoc: (%s) %s" % ( path.exists( self.baseLoc ), self.baseLoc ) )

        # Check if things are working
        dirGood = True

        if not path.exists( self.ptsDir ) or not path.exists( self.imgDir ) or not path.exists( self.miscDir ) or not path.exists( self.wndDir ):
            dirGood = False

        # If you made it this far.  
        return dirGood

    # End initialize run directory structure
    
    # If asked to create new info file
    def newRunSetup( self, rArg ):
        
        from os import remove, mkdir
        from shutil import copyfile
        
        # Remove current info file
        if path.exists( self.infoLoc ): remove( self.infoLoc )

        # Copy base info file.
        if path.exists( self.baseLoc ): copyfile( self.baseLoc, self.infoLoc )     
        
        # Create directories if not found
        if not path.exists( self.ptsDir ): mkdir( self.ptsDir )
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.miscDir ): mkdir( self.miscDir )
        if not path.exists( self.wndDir ): mkdir( self.wndDir ) 
        if not path.exists( self.tmpDir ): mkdir( self.tmpDir ) 


            
    def __del__(self,):
        pass

    def delTmp(self,):

        from os import remove
        from shutil import rmtree
        
        for f in listdir( self.tmpDir ):
            
            fLoc = self.tmpDir + f
            
            if path.isfile(fLoc):
                remove(fLoc)
                
            elif path.isdir(fLoc):
                rmtree(fLoc)
            

    def findPtsLoc( self, ptsName ):

        ptsLoc = self.ptsDir + '%s.zip' % ptsName
        oldLoc1 = self.ptsDir + '%s_pts.zip' % ptsName
        oldLoc2 = self.ptsDir + '100000_pts.zip'
        
        # Updating how particile files are saved
        if path.exists( oldLoc1 ) or path.exists( oldLoc2 ):
            from os import rename
            
            if path.exists( oldLoc1 ):
                rename( oldLoc1, ptsLoc )
                
            if path.exists( oldLoc2 ):
                rename( oldLoc2, ptsLoc )

        if gm.validPath( ptsLoc ):
            return ptsLoc
        else:
            return None

    # End findPtsFile
    
    def getParticles( self, ptsName ):
        
        from zipfile import ZipFile    

        zipLoc = self.findPtsLoc( ptsName )        
        if zipLoc == None:
            return None
             
        # Check if zip files need rezipping
        self.delTmp()
        rezip = True
               
        with ZipFile( zipLoc ) as zip:
            
            for zip_info in zip.infolist():
                
                if zip_info.filename[-1] == '/':
                    continue
                if len( zip_info.filename.split('/') ) > 1:
                    rezip = True
                    
                zip_info.filename = path.basename( zip_info.filename)
                zip.extract(zip_info, self.tmpDir)

        if rezip:
            from os import remove
            remove(zipLoc)
            
            with ZipFile( zipLoc, 'w' ) as zip:
                for f in listdir( self.tmpDir ):
                    fLoc = self.tmpDir + f
                    zip.write( fLoc, path.basename(fLoc) )
                    
        files = listdir( self.tmpDir )
        pts1 = None
        pts2 = None
        
        for f in files:
            fLoc = self.tmpDir + f
            
            if '.000' in f:
                pts1 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values
                
            if '.101' in f:
                pts2 = pd.read_csv( fLoc, header=None, delim_whitespace=True ).values
        
        return pts1, pts2

    
    def getModelImage( self, imgName = 'zoo_0', imgType = 'model', toType=np.float32, overWrite=False ):
        
        # Create place to store images if needed.
        if self.get('img',None) == None: self.img = {}            
        if self.get('init',None) == None: self.init = {}
        
        # Return model image if already loaded.
        if imgType == 'model' and not overWrite:
            mImg = self.img.get(imgName,None)
            if type(mImg) != type(None) and mImg.dtype == toType:
                return mImg
        
        elif imgType == 'init' and not overWrite:            
            img = self.init.get(imgName,None)
            if type(img) != type(None) and iImg.dtype == toType:
                return img
        
        # Get image location
        imgLoc = self.findImgLoc( imgName = imgName, imgType = imgType )
        if self.printAll: 
            print("IM: Loading: %s:")
            gm.tabprint( 'imgType: %s' % imgType )
            gm.tabprint( 'imgLoc: %s' % imgLoc )
            
        if imgLoc == None: return None
        
        # Read image
        img = gm.readImg(imgLoc, toType=toType)
        
        # Store image if called upon later
        if imgType == 'model':  self.img[imgName] = img            
        elif imgType == 'init': self.init[imgName] = img

        # Return image
        return img
    
    # End getting model, unperturbed image


    def findImgLoc( self, imgName, imgType = 'model', newImg=False ):

        # Assume model image
        if imgType == 'model':
            imgLoc = self.imgDir + imgName + '_model.png'

        elif imgType == 'init':
            imgLoc = self.miscDir + imgName + '_init.png'
            
        elif imgType == 'wndchrm':
            imgLoc = self.wndDir + imgName + '.tiff'
            
        else:
            imgLoc = self.miscDir + imgName + '.png'

        # If new image, return the location it will become
        if newImg: 
            return imgLoc
        
        # Return if path exists
        if path.exists( imgLoc ):
            return imgLoc
        
        else:
            return None

    # End findImgFile

    def getAllImgLocs( self, miscImg=False ):

        imgDir = self.imgDir
        if miscImg:
            imgDir = self.miscDir

        imgList = listdir(imgDir)
        retList = []
        for imgName in imgList:
            retList.append(imgDir+imgName)

        return retList

    def printInfo( self,):
        from pprint import PrettyPrinter
        pp = PrettyPrinter( indent = 2 )

        print( "IM: run_info_class.printInfo():")

        pprint( self.rDict )

    def getScore( self, sName,  ):

        if self.rDict == None:
            return None
        elif self.get('machine_scores') == None:
            self.rDict['machine_scores']
            return None

        score = self.rDict['machine_scores'].get( sName, None )

        return score

    def getAllScores( self  ):
        return self.get('machine_scores')

    def printScores( self, allScores=True):

        print("IM: run_info_class.printScores()")
        tabprint( 'run_id: %s' % self.get('run_id') )
        tabprint( 'zoo_merger: %s' % str( self.get('zoo_merger_score')) )
        tabprint( 'machine_scores: %d' % len(self.rDict['machine_scores']) )

        if allScores:
            for sKey in self.rDict['machine_scores']:
                print( '\t - %s: %f' %  (sKey, self.getScore( sKey )) )

    def addScore( self, name=None, score=None ):

        if name == None or score == None:
            if self.printAll:
                print("IM: WARNING: run.addScore. name or score not given")
                print('\t - name: ', name)
                print('\t - score: ', score)
            return None

        self.rDict['machine_scores'][name] = score

        return self.rDict['machine_scores'][name]


    def get( self, inVal, defaultVal = None ):

        cVal = getattr( self, inVal, defaultVal )

        if cVal != defaultVal:
            return cVal

        dVal = self.rDict.get( inVal, defaultVal )
        if dVal != defaultVal:
            return dVal

        return defaultVal


    def txt2Json( self, ):
        
        print("\n" + "#"*120 + '\n')
        print("THIS FUNCTION SHOULD BE FOREVER OBSOLETE!")
        print("\n" + "#"*120 + '\n')

        if self.printAll: print("IM: Run.txt2Json")

        self.rDict = self.createBlank()

        oldLoc = self.runDir + 'info.txt'

        if self.printAll:
            print( '\t- oldLoc: (%s) %s' % ( path.exists( oldLoc ), oldLoc ) )

        # Check if info location have a file
        if not path.exists( oldLoc ):
            print("IM: Run.txt2Json: ERROR:") 
            print("\t - info.txt file not found.")
            print("\t - oldLoc: %s" % oldLoc)
            return False

        infoData = gm.readFile( oldLoc )

        # Check if retrieved info
        if infoData == None:
            print("Error: IM: Info.txt file not found to create new info.json file")
            print("\t - oldLoc: %s" % oldLoc)
            return None

        # Go through info file and add appropriate information to json

        if self.printAll: print("\t - Reading info.txt")

        tid = None
        rNum = None
        gNum = None
        mData = None
        hScore = None
        wins = None

        for i,l in enumerate( infoData ):
            line = l.strip().split()
            if self.printAll: print('\t-',i,line)
                
            if len(l) < 2: continue

            if line[0] == 'sdss_name':
                tid = line[1]

            if line[0] == 'generation':
                gNum = line[1]

            if line[0] == 'run_number':
                rNum = line[1]

            if line[0] == 'model_data':
                mData = line[1]

            if line[0] == 'wins/total':
                wins = line[1]

            if line[0] == 'human_score':
                hScore = line[1]


        # check if all infor information found
        if ( mData == None or wins == None or hScore == None ):
            if self.printBase: print("Error: IM: Needed information not found in info.txt")
            if self.printBase : print("\t - infoLoc: %s" % oldLoc)
            return None

        # readjust run_id

        # Assume name of directory I'm in is the run_id
        self.rDict['run_id'] = 'r' + str( rNum )
        self.rDict['model_data'] = str(mData)
        self.rDict['zoo_merger_score'] = float(hScore)

        # created initial info.json file from info.txt
        self.baseDict = deepcopy(self.rDict)

        # Save results
        self.saveInfoFile( saveBase = True )

    # End txt2Json


    # end info2Dict

    def saveInfoFile( self, saveBase=False, ):

        if self.printAll: print("IM: Run.saveInfoFile: Saving info data file...")

            
        with open( self.infoLoc, 'w' ) as infoFile:
            json.dump( self.rDict, infoFile, indent=4 )
            retVal = True

            
        if saveBase:
            with open( self.baseLoc, 'w' ) as bFile:
                json.dump( self.baseDict, bFile, indent=4 )

        self.initDict = deepcopy( self.rDict )
    # End save info file

# End Run info class


class target_info_class:

    tDict = None	    # Primary dictionary of info for all data 
    baseDict = None	    # Simple dictionary of info for basic info.
    progDict = None	    # For saving progress of current target
    status = False	    # State of this class.  For initiating.

    sFrame = None	   # Score dataframe
    rInfo = None	   # run_info_class for accessing run directories

    printBase = True
    printAll = False

    targetDir = None
    zooMergerDir = None
    plotDir = None

    baseInfoLoc = None
    allInfoLoc = None
    scoreLoc = None
    baseScoreLoc = None
    zooMergerLoc = None


    # For user to see headers.
    targetHeaders = ( 'target_id', 'target_images', 'simulation_parameters', 'image_parameters', 'zoo_merger_models', 'score_parameters', 'progress', 'model_sets' )
    
    baseHeaders = ( 'target_id', )


    def __init__( self, targetDir = None, tArg = gm.inArgClass(), \
            printBase = None, printAll = None, ):

        # Tell class what to print
        if printBase != None:
            self.printBase = printBase
        else:
            self.printBase = tArg.printBase
        
        if tArg.targetDir == None: tArg.targetDir = targetDir
            
        if printAll != None:   
            self.printAll = printAll
        else:            
            self.printAll = tArg.printAll            
            
        # To avoid future confusion
        if self.printAll:  self.printBase = True

        if self.printAll: 
            print("IM: target_info_class.__init__:")
            print('\t - targetDir: ', tArg.targetDir)
            
        # Check if directory has correct structure
        newInfo = tArg.get('newInfo',False)
        dirGood = self.initTargetDir( tArg )

        # Complain if not
        if not dirGood:
            if self.printBase:
                print("IM: Target.__init__(): ")
                print("\t - WARNING: Something went wrong initializing directory.")
            return


        if self.printAll: 
            print("IM: Target: Opening target info json")

        # Read info file
        with open( self.allInfoLoc ) as iFile:
            self.tDict = json.load( iFile )
        
        # Open score file
        if path.exists( self.scoreLoc ):
            # Read all as string
            self.sFrame = pd.read_csv( self.scoreLoc )
        
        self.status = True

    # End target init 

    def getTargetImage( self, tName = None, overwrite=False, printAll = False ):
        
        if printAll: print("IM: target_info_class: getTargetImage:")
        
        # Create place to store images if needed.
        if self.get('targetImgs',None) == None:
            if printAll: gm.tabprint("Creating dict to store loaded target images")
            self.targetImgs = {}
        
        # Return target image if already loaded.
        tImg = self.targetImgs.get(tName,None)
        if type(tImg) != type(None) and not overwrite:
            if printAll:  gm.tabprint("Returning preloaded target image.")
            return tImg
        
        # Else find and open target image
        if printAll:  gm.tabprint("Searching for target image: %s" % tName)
        tLoc = self.findTargetImage(tName)
        if printAll:  gm.tabprint("Found: %s" % tLoc)
            
        if not gm.validPath(tLoc,):
            return None
        
        else:
            self.targetImgs[tName] = gm.readImg(tLoc)
            return self.targetImgs[tName]
        
    # End getTargetImage()


    def findTargetImage( self, tName = None, newImg = False ):

        # Expected location
        tLoc = self.imgDir + 'target_%s.png' % tName
        
        # If needing path for new image, return
        if newImg:
            return tLoc
        
        # Only return if file exists
        if path.exists( tLoc ):
            return tLoc
        else:
            return None
    # End find target image by name

    def printParams( self, ):

        for pKey in self.tDict['score_parameters']:
            gm.pprint( self.tDict['score_parameters'][pKey] )
            

    def addScoreParameters( self, params, overWrite = False ):
    
        for pKey in params:
            
            if self.tDict['score_parameters'].get(pKey) == None or overWrite:
                self.tDict['score_parameters'][pKey] = params[pKey]


    def getScores( self, scrName = None, reload=False): 

        # Reading from file
        if type(self.sFrame) == type(None):
            if gm.validPath( self.scoreLoc, printWarning=False ):
                self.sFrame = pd.read_csv( self.scoreLoc )
                
            else:
                if self.printAll: print( "IM: WARNING: Target.getScores:" )
                self.createBaseScore()
                
        if reload:
            self.sFrame = pd.read_csv( self.scoreLoc )
            
        return self.sFrame

    def createBaseScore( self, ):

        zDict = self.tDict['zoo_merger_models']
        nRuns = len(zDict)

        scoreHeaders = [ 'run_id', 'zoo_merger_score' ]

        self.sFrame = pd.DataFrame( \
                index = np.arange( nRuns ), \
                columns = scoreHeaders 
            )

        for i, rKey in enumerate( zDict ):
            rDict = zDict[rKey]
            self.sFrame.at[i,'run_id'] = rDict['run_id'] 
            self.sFrame.at[i,'zoo_merger_score'] = rDict['zoo_merger_score']

        self.sFrame.to_csv( self.baseScoreLoc, index = False, quoting=2 )
        self.sFrame.to_csv( self.scoreLoc, index = False, quoting=2 )

    # End create Base Score file

    def updateScores( self, ):

        # Construct bare sFrame
        if type(self.sFrame) == type(None):
            print("Oh NOOO")
            return

        zDict = self.tDict['zoo_merger_models']
        nRuns = len(zDict)

        # Loop through run directories and increment count
        for i, row in self.sFrame.iterrows():

            rKey = row['run_id']
            rDict = zDict.get(rKey,None)
            
            # Occasional bug when I force end a program
            if rDict == None:
                rInfo = self.getRunInfo( rID = rKey )
                if rInfo == None:
                    continue
                else:
                    rDict = rInfo.rDict
            rScores = rDict['machine_scores']

            for sKey in rScores:
                self.sFrame.at[i,sKey] = rScores[sKey]


        # Update progress in tInfo
        if self.tDict['progress'].get('machine_scores') == None:
            self.tDict['progress']['machine_scores'] = {}

        scoreHeaders = list( self.sFrame.columns )
        for sName in scoreHeaders:
            if 'zoo_merger_score' in sName:
                continue

            sCount = self.sFrame[sName].count()
            self.tDict['progress']['machine_scores'][sName] = int( sCount )

        #print(self.tDict['progress'])

        self.saveInfoFile()
        #self.sFrame.to_csv( self.baseScoreLoc, index = False, quoting=2 )
        self.sFrame.to_csv( self.scoreLoc, index = False, quoting=2 )

    # End gathering scores and progress

    def get( self, inVal, defaultVal = None ):

        cVal = getattr( self, inVal, defaultVal )

        if cVal != defaultVal:
            return cVal
        
        if self.tDict != None:
            dVal = self.tDict.get( inVal, defaultVal )
            if dVal != defaultVal:
                return dVal

        return defaultVal


    # Gather run infos from directories
    def gatherRunInfos( self, tArg=gm.inArgClass(), rArg=gm.inArgClass(), ):

        if self.printAll: 
            print( "IM: Target.gatherRunInfos" )
            #rArg.printArg()

        runDirList = self.iter_runs()
        nRuns = len(runDirList)
        
        if rArg.get('newInfo',False):
            if self.printBase: print("IM: Target: gatherRunInfos: Adjusting run infos")
            
            if mpi_size == 1:
                for rDir in runDirList:
                    rInfo = run_info_class( runDir = rDir, rArg = rArg, printBase = self.printAll   )
                    if self.printAll and rInfo.status == False: gm.tabprint( '%s - %s' % (rInfo.status, rDir ) )
                        
            else:
                print("WARNING: IM: Target.gatherRunInfos:  initializing run directories not available in MPI environment.")
                gm.tabprint(self.get('target_id'))

        # Prepare model Set
        for h in self.targetHeaders:
            if self.tDict.get(h,None) == None:
                self.tDict[h] = {}

        # Go through directories and read run info files
        for i,rDir in enumerate(runDirList):
            if self.printAll: 
                gm.tabprint("IM: gather_run_info LOOP: %4d / %4d" % (i, nRuns), end='\r')
            rDict = self.getRunDict( rDir )
            if rDict != None and rDict.get('run_id',None) != None:
                self.tDict['zoo_merger_models'][rDict['run_id']] = rDict

        if self.printAll: gm.tabprint("IM: gather_run_info LOOP: %4d / %4d COMPLETE" % (nRuns, nRuns))
            
        self.saveInfoFile()

    # end gather Run Infos

    def getRunDict( self, rDir ):

        if rDir[-1] != '/': rDir += '/'
        dictLoc = rDir + 'info.json'
        rDict = gm.readJson( dictLoc )
        return rDict
    
    # End get Run Dir Info


    def getRunDir( self, rID=None,  ):
           
        runDir = self.zooMergerDir + '%s/' % rID        
        return gm.validPath(runDir)

    def getRunInfo( self, rID=None, rArg=None ):
        
        runDir = self.getRunDir(rID=rID)
        
        if runDir == None:
            return None

        if rArg == None:
            rInfo = run_info_class( runDir = runDir, )
            
        else: 
            rInfo = run_info_class( runDir = runDir, rArg=rArg )
        
        rInfo.tInfo = self

        if rInfo.status:
            return rInfo

        else:
            return None
    
    def addRunDict( self, rInfo ):
        rID = rInfo.get('run_id',None)
        if rID != None:
            self.tDict['zoo_merger_models'][rID] = rInfo.rDict
    
    def getAllRunDicts( self, ):
        return self.tDict['zoo_merger_models']

    def saveInfoFile( self ):

        if self.printAll: 
            print("IM: Target.saveInfoFile():")
            print("\t - Saving target info file...")

        if self.allInfoLoc == None:
            print("ERROR: IM: No target info location given...")
            return None

        with open( self.allInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
            self.initDict = deepcopy( self.tDict )

        if type(self.sFrame) != type(None):
            self.sFrame.to_csv( self.scoreLoc, index = False, quoting=2 )

   # End Target info class


    # Read in classes for run infos
    def iter_runs( self,  start=0, n = -1, ):

        # Get list of run directories
        self.runDirs = [ self.zooMergerDir + item + '/' for item in listdir( self.zooMergerDir ) if 'run' in item ]
        self.runDirs.sort()

        if start == 0:
            self.runDirs = self.runDirs[:n]
        elif n == -1:		  
            self.runDirs = self.runDirs[start:n]
        else:
            self.runDirs = self.runDirs[start:start+n]

        return self.runDirs



    # initialize target directories
    def initTargetDir( self, tArg ):
        
        from shutil import move
        from os import mkdir
        
        if self.printAll:
            print( 'IM: Target.initTargetDir():' )
            print( '\t - targetDir: %s' % tArg.targetDir )

        self.targetDir = gm.validPath( tArg.targetDir )

        # if Invalid, complain
        if type(self.targetDir) == type(None):
            if self.printBase:
                print("IM: WARNING: Invalid directory.")
                gm.tabprint('Input: %s'%tArg.targetDir)
                gm.tabprint('Full:  %s' % self.targetDir)
            return False

        # If not directory, complain
        elif not path.isdir( self.targetDir ):
            if self.printBase:
                print("IM: WARNING: Target:")
                print("\t - Target not a directory")
            return False

        # Define paths for all useful things in target structure
        
        # Main paths in target dir
        self.infoDir = self.targetDir + 'information/'
        self.gen0 = self.targetDir + 'gen000/'
        self.zooMergerDir = self.targetDir + 'zoo_merger_models/'
        self.plotDir = self.targetDir + 'plots/'
        
        # Directires inside info dir
        self.imgDir = self.infoDir + 'target_images/'
        self.maskDir = self.infoDir + 'target_masks/'        
        self.scoreParamDir = self.infoDir + 'score_parameters/'        
        self.wndDir = self.infoDir + 'wndchrm_files/'

        # Locations of files inside info dir
        self.allInfoLoc = self.infoDir + 'target_info.json'
        self.baseInfoLoc = self.infoDir + 'base_target_info.json'
        self.scoreLoc = self.infoDir + 'scores.csv'
        self.baseScoreLoc = self.infoDir + 'base_scores.csv'    
        self.zooMergerLoc = self.infoDir + 'galaxy_zoo_models.txt'
        self.imgParamLocOld = self.infoDir + 'param_target_images.json'
        
        # Files inside misc directories
        self.imgParamLoc = self.imgDir + 'param_target_images.json'
        self.wndRunRawLoc = self.wndDir + 'all_runs_raw.pkl'
        self.wndTargetRawLoc = self.wndDir + 'targets_raw.csv'
        self.wndTargetFitLoc = self.wndDir + 'targets_raw.fit'
                 
        # Create target subdirectories if not found
        if not path.exists( self.infoDir ): mkdir( self.infoDir )
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.maskDir ): mkdir( self.maskDir )
        if not path.exists( self.wndDir ): mkdir( self.wndDir )
        if not path.exists( self.scoreParamDir ): mkdir( self.scoreParamDir )
        
        # Move old directories to new location if needed.
        if gm.validPath( self.imgParamLocOld ) != None:
            print("Hi")
            print("OLD: %s"%gm.validPath(self.imgParamLocOld))
            move( self.imgParamLocOld, self.imgParamLoc)
            print("NEW: %s"%gm.validPath(self.imgParamLoc))
        
        # If using old folder layout, rename        
        if gm.validPath( self.gen0 ):
            from os import rename
            print("Found old path   : %s" % self.gen0)
            print("Propose new path : %s" % self.zooMergerDir)
            rename(self.gen0,self.zooMergerDir)
            print("New Path? : %s" % gm.validPath(self.zooMergerDir) )
        
        # Check if creating a new info file, run new Target Setup if so
        if tArg.get('newInfo',False): 
            status = self.newTargetSetup( tArg )
            if status == False: return False

        # Check if directories are found        
        if ( not path.exists( self.infoDir ) \
        or not path.exists( self.zooMergerDir ) \
        or not path.exists( self.plotDir ) ):
            
            if self.printBase:
                print("IM: Target_init: Some directories not found")
                tabprint('Info: %s' % gm.validPath(self.infoDir))
                tabprint('Plot: %s' % gm.validPath(self.plotDir))
                tabprint('Zoo Merger: %s' % gm.validPath(self.zooMergerDir))
                tabprint("Consider using -newInfo command")
                
            return False
            
        
        # Check if info directory has needed objects
        if not path.exists( self.allInfoLoc   ) \
        or not path.exists( self.baseInfoLoc  ) \
        or not path.exists( self.scoreLoc     ) \
        or not path.exists( self.baseScoreLoc ):
            if self.printBase:
                print("IM: Target_init: Needed information files not found.")
                tabprint('Main Info: %s' % gm.validPath(self.allInfoLoc))
                tabprint('Base Info: %s' % gm.validPath(self.baseInfoLoc))
                tabprint('Main Score: %s' % gm.validPath(self.scoreLoc))
                tabprint('Base Score: %s' % gm.validPath(self.baseScoreLoc))
                tabprint("Consider using -newInfo command")
            return False

        if self.printAll:
            print( '\t - targetDir: (%s) %s' % ( path.exists( self.targetDir ), self.targetDir ) )
            print( '\t - infoDir: (%s) %s' % ( path.exists( self.infoDir ), self.infoDir ) )
            print( '\t - baseInfoLoc: (%s) %s' % ( path.exists( self.baseInfoLoc ), self.baseInfoLoc ) )
            print( '\t - allInfoLoc: (%s) %s' % ( path.exists( self.allInfoLoc ), self.allInfoLoc ) )
            print( '\t - zooMergerDir: (%s) %s' % ( path.exists( self.zooMergerDir ), self.zooMergerDir ) )
            print( '\t - plotDir: (%s) %s' % ( path.exists( self.plotDir ), self.plotDir ) )

        return True

    
    # if calling new Info
    def newTargetSetup( self, tArg ):
        
        from os import mkdir, remove
        from shutil import copyfile

        if self.printBase: 
            print("IM: new_target_setup: Creating basic files and directories")
            
        # remove old files
        if path.exists( self.allInfoLoc ):  remove( self.allInfoLoc )
        if path.exists( self.scoreLoc ):    remove( self.scoreLoc )         
            
        # Create directories
        if not path.exists( self.infoDir ): mkdir( self.infoDir )
        if not path.exists( self.plotDir ): mkdir( self.plotDir )
        if not path.exists( self.imgDir ): mkdir( self.imgDir )
        if not path.exists( self.maskDir ): mkdir( self.maskDir )

        if not path.exists( self.zooMergerDir ):
            print("IM: WARNING: This message should not be seen.")
        
        # Check for base file
        newBase = tArg.get('newBase',False)
        if not path.exists( self.baseInfoLoc ) and not newBase:
            if self.printBase: 
                print("IM: WARNING: newTargetSetup:")
                tabprint("No base info file!")
                tabprint("Consider '-newBase' command")
            return False
        
        if newBase:
            if self.printBase:
                createGood = self.createBaseInfo( tArg )
                if not createGood: return False
        
        # Copy files if they exist
        if path.exists( self.baseInfoLoc ): copyfile( self.baseInfoLoc, self.allInfoLoc )
        if path.exists( self.baseScoreLoc ): copyfile( self.baseScoreLoc, self.scoreLoc )

        # Read info file
        with open( self.allInfoLoc ) as iFile:
            self.tDict = json.load( iFile )
        
        # Open score file
        if path.exists( self.scoreLoc ):
            # Read all as string
            self.sFrame = pd.read_csv( self.scoreLoc )

        # Should run infos be modified? 
        rArg = gm.inArgClass()
        rArg.printBase = False
        if tArg.get('newRunInfo',True):
            rArg.newInfo = True
        if tArg.get('newRunBase',True):
            rArg.newBase = True

        self.gatherRunInfos( tArg=tArg, rArg=rArg )
        
        # Collect info 
        if newBase: self.createBaseScore(  )
        self.updateScores()
        self.saveInfoFile( )

    # End new target info dictionary
    
    
    def createBaseInfo( self, tArg ):
        
        from os import getcwd, listdir, remove, mkdir
        from shutil import copyfile, move
        from copy import deepcopy
        
        # For basic scores later on
        
        # Get parent directory this code suite is located in.
        simrDir = __file__.split('Support_Code/info_module.py')[0]
         
        # Assume directory name of target is target name
        tName = self.targetDir.split('/')[-2]
        printAll = self.printAll
        tInfo = self
        
        if self.printAll: 
            print('IM: createBaseInfo')
            gm.tabprint('Target ID: %s'%tName)  
            gm.tabprint('Target Dir: %s'%self.get('target_dir'))
                
        # Create blank base dict
        self.tDict = {}
        for key in self.baseHeaders:
            self.tDict[key] = {}        
        self.tDict['target_id'] = tName  
        
        # Save target_id
        with open( self.baseInfoLoc, 'w' ) as infoFile:
            json.dump( self.tDict, infoFile )
        
        # Get starting target files from input data folder
        inputDir = gm.validPath( simrDir  + 'Input_Data/targets/' + tName + '/')        
        if self.printAll: gm.tabprint('Input Dir: %s'%inputDir)
            
        # Check if valid directory, exit if not
        if inputDir == None:
            if self.printBase: 
                print("WARNING: IM: Input directory not found: %s"%tName)
                gm.tabprint('Input Dir: %s'%inputDir)
            return False
        
        # Get folder contents
        inputFiles = listdir(inputDir)            

        if self.printAll: 
            gm.tabprint("Input Dir contents:")
            for f in inputFiles:
                gm.tabprint('    - %s'%f)
        
        # Search for target image
        fromLoc = None        
        for fName in inputFiles:
            if '.png' in fName:
                fromLoc = inputDir + fName      
        
        # If found, copy target image from input folder to targetInfo folder
        if gm.validPath( fromLoc ) != None:
            toLoc = self.findTargetImage( 'zoo_0', newImg = True )
            copyfile( fromLoc, toLoc )
            
        # Grab specific target image data
        metaLocRaw1 = inputDir + 'sdss%s.meta'%tName
        metaLocRaw2 = inputDir + '%s.meta'%tName
        metaLoc1 = gm.validPath( metaLocRaw1 )
        metaLoc2 = gm.validPath( metaLocRaw2 )
        
        if self.printAll: 
            gm.tabprint('Meta Data Loc1: %s'% metaLocRaw1 )
            gm.tabprint('Meta Data Loc2: %s'% metaLocRaw2 )
            
        # Open file for target galaxy zoo merger image information
        if metaLoc1 != None:
            mFile = open(metaLoc1,'r')
        elif metaLoc2 != None:
            mFile = open(metaLoc2,'r')
        else:
            if self.printBase: print("WARNING: IM: Meta data not found:")
            return False

        # Copy starting target zoo image param
        pLoc = simrDir + 'param/zoo_blank.json'
        blank_param = gm.readJson(pLoc)
        if blank_param == None:        
            if self.printBase: 
                print("WARNING: IM: Start zoo image param not found: %s"%pLoc)
            return False

        # Copy blank parameter
        base_zoo = {}
        zoo_name = 'zoo_0'
        base_zoo[zoo_name] = deepcopy(blank_param['zoo_blank'])

        # Make name and comments initial comments.
        base_zoo[zoo_name]['name'] = zoo_name
        base_zoo[zoo_name]['comment'] = 'Starting score parameters file for %s'%tName
        base_zoo[zoo_name]['imgArg']['comment'] = "Starting image parameters for %s"%tName

        # Grab information
        for l in mFile:
            l = l.strip()
            
            if 'height' in l:
                h = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['image_size']['width'] = int(h)
                
            if 'width' in l:
                w = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['image_size']['width'] = int(w)
            if 'px' in l:
                px = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['px'] = int(px)
            if 'py' in l:
                py = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['py'] = int(py)
            if 'sx' in l:
                sx = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['sx'] = int(sx)
            if 'sy' in l:
                sy = l.split('=')[1]
                base_zoo[zoo_name]['imgArg']['galaxy_centers']['sy'] = int(sy)

        if self.printAll: gm.pprint(base_zoo)

        # Save new target image parameter
        newParamLoc = self.imgParamLoc
        gm.saveJson( base_zoo, newParamLoc, pretty=True )
                
        # Create basic scoring parameters
        self.createDirectScoreParameters( base_zoo['zoo_0'] )
        
        # Create a starting score parameter file for WNDCHRM image creation. 


        # Create a blank group score parameter and copy starting parameters
        chime_name = 'chime_0'
        base_chime = {}
        base_chime[chime_name] = deepcopy( base_zoo[zoo_name] )

        # ALWAYS modify the names
        base_chime[chime_name]['name'] = chime_name
        base_chime[chime_name]['comment'] = 'Developing initial WNDCHRM implementation'

        # Resize WNDCHRM image to 100 pixels
        old_size = blank_param['zoo_blank']['imgArg']['image_size']
        chime_size = 100

        max_side = np.amax( [ int(old_size['width']), int(old_size['height']) ] )
        redox_ratio = float( chime_size / max_side )

        if printAll: 
            print("IM: target_info_class.createBaseInfo: Creating WNDCHRM score parameter")
            gm.tabprint("Old Size: %d" % max_side)
            gm.tabprint("New Size: %d" % chime_size)
            gm.tabprint("Reduction Ratio: %d" % redox_ratio)

        # Create new image parameters
        base_chime[chime_name]['imgArg']['name'] = chime_name
        base_chime[chime_name]['imgArg']['type'] = 'wndchrm'
        base_chime[chime_name]['imgArg']['comment'] = 'Starting image for WNDCHRM feature extraction'

        # Adjust Image Size
        base_chime[chime_name]['imgArg']['image_size']['width' ] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['image_size']['width' ]) )
        base_chime[chime_name]['imgArg']['image_size']['height'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['image_size']['height']) )

        # Adjust galaxy centers
        base_chime[chime_name]['imgArg']['galaxy_centers']['px'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['px'] ) )
        base_chime[chime_name]['imgArg']['galaxy_centers']['py'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['py'] ) )
        base_chime[chime_name]['imgArg']['galaxy_centers']['sx'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['sx'] ) )
        base_chime[chime_name]['imgArg']['galaxy_centers']['sy'] = int( np.rint( redox_ratio * base_chime[chime_name]['imgArg']['galaxy_centers']['sy'] ) )

        # Add feature arguments
        base_chime[chime_name]['featArg'] = {}
        base_chime[chime_name]['featArg']['type'] = 'wndchrm_all'
        base_chime[chime_name]['featArg']['normalization'] = None

        # WORKING
        # Change to feature comparison
        base_chime[chime_name]['cmpArg']['targetName'] = chime_name
        base_chime[chime_name]['cmpArg']['type'] = 'direct_feature_comparison'

        if printAll: 
            gm.tabprint("Saving Base WNDCHRM parameter")
            gm.pprint(base_chime)
            
        # Save score param for image creation later
        tInfo.saveScoreParam( base_chime, chime_name )

        # Find pair file 
        pairPath1 = gm.validPath( inputDir + 'sdss%s.pair' % tName )
        pairPath2 = gm.validPath( inputDir + '%s.pair' % tName )
        if self.printAll: 
            gm.tabprint( 'pairPath1: %s' % pairPath1 )
            gm.tabprint( 'pairPath2: %s' % pairPath2 )
            
        if pairPath1 != None:            
            pairFile = open(pairPath1, 'r' )
            
        elif pairPath2 != None:            
            pairFile = open(pairPath2, 'r' )
            
        else:
            if self.printBase: print( "WARNING: IM: Pair data not found: %s" % pairPath )
            return False

        # Create a basic mask that covers the target galaxy based on pair file. 
        start_roi_mask = gm.readJson(simrDir+'param/mask_roi_blank.json')

        start_roi_mask['name'] = 'mask_roi_zoo_0'
        start_roi_mask['comment'] = 'Starting mask for %s' % tName
        start_roi_mask['target_name'] =  tName
        start_roi_mask['primary_start']['thickness'] =  10
        start_roi_mask['secondary_start']['thickness'] =  10

        for l in pairFile: 
            
            if ';' in l:                
                l = l[0:-2]

            if 'primaryA=' in l:
                l = l.strip()
                start_roi_mask['primary_start']['A'] =  round( float( l.split('=')[1] ) )

            elif 'primaryB=' in l:
                start_roi_mask['primary_start']['B'] =  round( float( l.split('=')[1] ) )

            elif 'primaryAngle=' in l:
                start_roi_mask['primary_start']['angle'] =   float( l.split('=')[1] ) 

            elif 'primaryX=' in l:
                start_roi_mask['primary_start']['center'][0] =  round( float( l.split('=')[1] ) )

            elif 'primaryY=' in l:
                start_roi_mask['primary_start']['center'][1] =  round( float( l.split('=')[1] ) )

            elif 'secondaryA=' in l:
                l = l.strip()
                start_roi_mask['secondary_start']['A'] =  round( float( l.split('=')[1] ) )

            elif 'secondaryB=' in l:
                start_roi_mask['secondary_start']['B'] =  round( float( l.split('=')[1] ) )

            elif 'secondaryAngle=' in l:
                start_roi_mask['secondary_start']['angle'] =   float( l.split('=')[1] ) 

            elif 'secondaryX=' in l:
                start_roi_mask['secondary_start']['center'][0] =  round( float( l.split('=')[1] ) )

            elif 'secondaryY=' in l:
                start_roi_mask['secondary_start']['center'][1] =  round( float( l.split('=')[1] ) )


        if self.printAll: 
            gm.pprint(start_roi_mask)

        self.saveMaskRoi( start_roi_mask, 'mask_roi_zoo_0')
        
                
        # Create run base info files.

        # If creating new run base infos
        if tArg.get("newRunBase",False):
            
            # Get directory with galaxy zoo merger files. 
            modelLoc = gm.validPath( simrDir  + 'Input_Data/zoo_models/' + tName + '.txt')    
            if self.printAll: gm.tabprint("Model File: %s - %s" % ( path.exists(modelLoc), modelLoc ) )

            modelFile = gm.readFile( modelLoc )
            if modelFile == None:
                if self.printAll: print("WARNING: IM: target.createBaseInfo: Failed to open zoo model file.")
                return False
            
            # Create a blank run info dict for copying
            runHeaders = ( 'run_id', 'model_data', \
                    'zoo_merger_score', 'machine_scores', 'human_scores')
            
            blank_run_info = {}
            for rh in runHeaders:
                blank_run_info[rh] = {}
            
            for i,l in enumerate(modelFile):
                
                # Grab galaxy zoo merger model data from file
                score_data, model_data = l.strip().split()                
                scores = score_data.split(',')        
                if len( scores ) != 4:
                    if self.printAll: gm.tabprint("Found Models: %d"%i)
                    break
                
                zoo_merger_score = scores[1]
                wins = scores[2] 
                losses = scores[3]
                
                # Create / goto run directory
                run_id = 'run_%s' % str(i).zfill(4)
                runDir = self.zooMergerDir + run_id + '/'
                
                # Move old directory if found
                oldDir = self.zooMergerDir + 'run_%s' % str(i).zfill(5) + '/'
                if gm.validPath( oldDir ) != None:
                    move( oldDir, runDir )
                    
                if gm.validPath( runDir ) == None:
                    mkdir( runDir )
                
                # Copy and fill in run info
                rInfo = deepcopy( blank_run_info )
                
                rInfo['run_id'] = run_id
                rInfo['zoo_merger_score'] = zoo_merger_score
                rInfo['model_data'] = model_data
                rInfo['human_scores']['zoo_merger_wins_losses'] = '%s/%s' % ( wins, losses )
                rInfo['human_scores']['zoo_merger_score'] = zoo_merger_score
                
                # Save base run info
                rInfoLoc = runDir + 'base_info.json'
                gm.saveJson( rInfo, rInfoLoc, pretty = True )
                
                # Remove old info file if found
                oldInfoLoc = runDir + 'info.txt'
                if gm.validPath( oldInfoLoc ) != None:
                    remove( oldInfoLoc )
                               

        return True
        
    # end creating base info file
    
    def createDirectScoreParameters( self, startParam ):
        
        # Create basic scoring parameters with zoo_0
        import Machine_Score.direct_image_compare as dc
        score_functions = dc.get_score_functions()

        direct_params = {}
        imgName = startParam['imgArg']['name']
        
        modParam = deepcopy( startParam )
        modParam['cmpArg']['type'] = 'direct_image_comparison'
        modParam['comment'] = 'Direct Image Comparison functions for image %s' % imgName
        
        for score_name, ptr in score_functions:
            new_name = '%s_%s' % ( imgName, score_name )
            direct_params[ new_name ] = deepcopy( modParam )
            direct_params[ new_name ]['name'] = new_name            
            direct_params[ new_name ]['cmpArg']['direct_compare_function'] =  score_name
            
        if self.printAll: 
            print("IM: Target_Class: createDirectScoreParameters")
            for name in direct_params.keys():
                gm.tabprint(name)
        
        # Save created score files
        self.saveScoreParam( direct_params, '%s_direct_scores'%imgName)            
        
    

    
    def getImageParams( self, imgName=None):
        
        from copy import deepcopy
        
        # Read image parameters common for target
        img_params = gm.readJson( self.imgParamLoc )
        
        # Return all if none specified.
        if imgName == None:     
            return deepcopy( img_params )
        
        # Return single image parameter if specified.
        else:
            return deepcopy( img_params.get(imgName,None) )
        
    # End getting image parameters
    
    def addImageParams( self, in_params, overWrite = False ):
        
        # If file doesn't exist
        if gm.validPath( self.imgParamLoc ) == None:
            gm.saveJson( in_params, self.imgParamLoc, pretty=True )
            return
        
        # Else file does exsit
        old_params = gm.readJson( self.imgParamLoc )
        
        # Loop through new image parameters
        for sKey in in_params:
            
            # Add to old if overwrite or not found
            if overWrite or sKey not in old_params:
                old_params[sKey] = in_params[sKey]            
        
        gm.saveJson( old_params, self.imgParamLoc, pretty=True )
    
    def overWriteImageParams( self, in_params ):
        gm.saveJson( in_params, self.imgParamLoc )
    
    def saveMaskImage( self, maskImage, maskName ):
        maskLoc = self.maskDir + '%s.png'%maskName
        gm.saveImg( maskImage, maskLoc)
    
    def getMaskImage( self, maskName ):
        
        if self.get('targetMasks',None) == None:
            self.targetMasks = {}
            
        if type( self.targetMasks.get(maskName,None) ) != type( None ):
            return self.targetMasks[maskName]
        
        else:
            mask = self.readMaskImage( maskName )
            self.targetMasks[maskName] = mask
            return self.targetMasks[maskName]
        
    
    def readMaskImage( self, maskName ):
        maskLoc = self.maskDir + '%s.png'%maskName
        mask = gm.readImg(  maskLoc )
        return mask
    
    def saveScoreParam( self, score_params, param_file_name ):
        paramLoc = self.scoreParamDir + '%s.json'%param_file_name
        gm.saveJson( score_params, paramLoc, pretty=True )
    
    def readScoreParam( self, param_file_name ):
        paramLoc = self.scoreParamDir + '%s.json'%param_file_name
        score_params = gm.readJson(  paramLoc )
        return score_params
    
    def saveMaskRoi( self, mask_roi, file_name ):
        roiLoc = self.maskDir + '%s.json'%file_name
        gm.saveJson( mask_roi, roiLoc, pretty=True )
    
    def readMaskRoi( self, file_name ):
        roiLoc = self.maskDir + '%s.json'%file_name
        mask_roi = gm.readJson(  roiLoc )
        return mask_roi
    
    def saveWndchrmImage( self, img, img_name ):
        imgLoc = self.wndDir + '%s.tiff' % img_name
        gm.saveImg( imgLoc, img )
        
    def saveWndchrmScaler( self, scaler, name ):
        from pickle import dump
        scalerLoc = self.wndDir + '%s_scaler.pkl' % name
        dump( scaler, open( scalerLoc, 'wb') )
        
    def readWndchrmScaler( self, name ):
        from pickle import load
        scalerLoc = self.wndDir + '%s_scaler.pkl' % name
        scaler = load( open( scalerLoc, 'rb') )
        return scaler
        
    def saveWndchrmDF( self, df, name ):
        dfLoc = self.wndDir + '%s.pkl' % name
        df.to_pickle( dfLoc )
        
    def readWndchrmDF( self, name ):
        dfLoc = self.wndDir + '%s.pkl' % name
        df = pd.read_pickle( dfLoc )
        return df
        
    
    
        
# End target info class
    
def tabprint( inprint, begin = '\t - ', end = '\n' ):
    print('%s%s' % (begin,inprint), end=end )


    
if __name__=='__main__':

    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

# End main

