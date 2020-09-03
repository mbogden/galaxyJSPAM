'''
    Author:     Matthew Ogden
    Created:    01 Sep 2020
Description:    Hopefully my primary code for calling all things Galaxy Simulation
'''

# Python module imports
from os import path, listdir

# For loading in Matt's general purpose python libraries
import Support_Code.general_module as gm
import Support_Code.info_module as im
import Simulator.main_simulator as ss


def main(arg):

    if arg.printBase:
        print("SIMR: Hi!  You're in Matthew's main program for all things galaxy collisions")

    if arg.printAll:
        arg.printArg()
        gm.test()
        im.test()
        ss.test()

    # end main print

    # Read param file
    param = pipeline_param_class( paramLoc = arg.getArg( 'paramLoc' ), printBase=arg.printBase, printAll = arg.printAll,  )

    if param.status == False:
        print("SIMR.main: Bad param file. Exiting....")
        return

    elif arg.printBase:
        print("SIMR: param.status: Good")

    if arg.simple:
        if arg.printBase: 
            print("SIMR: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None:
        pipelineRun( arg.runDir, param, printAll=arg.printAll, )

    elif arg.targetDir != None:
        procTarget( arg.targetDir, param, printAll=arg.printAll )

    elif arg.dataDir != None:
        procAllData( arg.dataDir, param, printAll=arg.printAll )

    else:
        print("SIMR: Nothing selected!")
        print("SIMR: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

# End main
    

def pipelineRun( rDir, param, printBase=True, printAll=False ):

    if printBase:
        print("SIMR.pipelineRun: Inputs")
        print("\t - rDir:", rDir)

    rInfo = im.run_info_class( runDir=rDir, printBase = True, printAll=printAll )

    if printBase:
        print('SIMR.pipelineRun: ')
        print('\t - rInfo.status: ', rInfo.status )

    if rInfo.status == False:
        print("SIMR.pipelineRun: WARNING: runInfo bad")
        return

    ptsLoc = procRunSim( rInfo, param, printBase = printBase, printAll = printAll )
    if ptsLoc == None:
        return 

    imgLoc = procRunImg( rInfo, param, printBase = printBase, printAll = printAll )
    if imgLoc == None:
        return

    score = procRunMach( rInfo, param, printBase = printBase, printAll = printAll )
    if score == None:
        return

# end processing run


# end procRunMach
def procRunMach( rInfo, param, printBase = True, printAll = False, createPlots=True ):

    # Get desired number of particles for simulation
    sName = getattr( param.machArg, 'score', None )

    if sName == None:
        print("SIMR.procRunMach: WARNING:")
        print("\t - Please specifcy score name")
        print("\t -score: score_absolute_difference")
        return None


def procRunImg( rInfo, param, printBase = True, printAll = False ):

    # Get desired number of particles for simulation
    imgParam = getattr( param.imgArg, 'imgParam', None )

    if imgParam == None:
        print("SIMR.procRunImg: WARNING:")
        print("\t - Please specificy imgParam")
        return None

    imgLoc = rInfo.findImgFile( imgParam )
    imgLoc, initLoc = rInfo.findImgFile( imgParam, initImg = True )

    if printBase:
        print('\t - imgParam: ' , imgParam)
        print('\t - imgLoc: %s' % imgLoc)
        print('\t - initLoc: %s' % initLoc)

    if imgLoc == None:
        print("SIMR.procRunImg: WARNING:")
        print("\t - image not found")
        print("\t - creating new image not yet implemented")
        return None

    return imgLoc

def procRunSim( rInfo, param, printBase = True, printAll = False ):


    # Get desired number of particles for simulation
    dPts = getattr( param.simArg, 'nPts', None )

    if dPts == None:
        print("SIMR.procRunSim: WARNING:")
        print("\t - Please specificy nPts")
        return None

    ptsLoc = rInfo.findPtsFile( dPts )

    if printBase:
        print('\t - dPts: ' , dPts)
        print('\t - ptsLoc: %s' % ptsLoc)

    if ptsLoc == None:
        print("SIMR.procRunSim: WARNING:")
        print("\t - nPts file not found")
        print("\t - creating new file not yet implemented")
        return None

    return ptsLoc


# end processing run dir


# Process target directory
def procTarget( tDir, printBase = True, printAll=False ):

    if printBase:
        print("SIMR.procTarget:")
        print("\t - tDir: " , tDir)

    tInfo = im.target_info_class( targetDir=tDir, printAll=True )

    if printBase:
        print("SIMR.procTarget:")
        print("\t - tInfo.status: %s" % tInfo.status )

    # Check if target is valid
    if tInfo.status == False:
        print("SIMR: WARNING: procTarget:")
        print("\t - Target not good.")
        return

    if printAll:
        tInfo.printInfo()

    tInfo.gatherRunInfos()
    
    for r in tInfo.runDirs:
        print("t")

    

# End processing sdss dir

def procAllData( dataDir, printBase=True, printAll=False ):

    from os import listdir

    if printBase: 
        print("SIMR.procAllData")
        print("\t - dataDir: %s" % arg.dataDir )

    # Check if directory exists
    if not path.exists( dataDir ):  
        print("SIMR.procAllData: WARNING: Directory not found")
        print('\t - ' , dataDir )
        return

    # Append trailing '/' if needed
    if dataDir[-1] != '/': dataDir += '/'  

    dataList = listdir( dataDir )   # List of items found in folder
    tDirList = []  # List of folders that are target directories

    # Find target directories
    for folder in dataList:
        tDir = dataDir + folder
        tempInfo = im.target_info_class( targetDir=tDir, printAll=False )

        # if a valid target directory
        if tempInfo.status:  tDirList.append( tempInfo )

    if printBase:
        print( '\t - Target Directories: %d' % len( tDirList ) )

class pipeline_param_class:

    status = False

    name = None
    simArg = gm.inArgClass()
    imgArg = gm.inArgClass()
    machArg = gm.inArgClass()

    def __init__( self, paramLoc = None, printBase = True, printAll = False ):

        self.printBase = printBase
        self.printAll = printAll
        if self.printAll: self.printBase == True

        if self.printBase:
            print("SIMR: pipeline_param_class.__init__")
            print("\t - paramLoc: ", paramLoc)

        self.readParam( paramLoc )
        
        if self.printAll:
            print("SIMR: pipeline_param_class.__init__")
            print("\t - name: %s" % self.name)
            print("SIMR: papam.simArg")
            self.simArg.printArg()

            print("SIMR: papam.imgArg")
            self.imgArg.printArg()

            print("SIMR: papam.machArg")
            self.machArg.printArg()

    def readParam( self, paramLoc ):
        
        if self.printAll:
            print("SIMR: pipeline_param_class.readParam")
            print("\t - paramLoc: ", paramLoc)
        
        # Check if param File is valid
        if paramLoc == None:
            print('SIMR: WARNING: Please give a param File Location')
            print('\t -paramLoc /path/to/file.txt')
            return

        elif type( paramLoc) != type('String'):
            print('SIMR: WARNING: paramLoc variable not string')
            print('\t -paramLoc: %s ' % type(paramLoc), paramLoc)
            return

        elif not path.exists( paramLoc ):
            print('SIMR: WARNING: Param File location not found')
            print('\t -paramLoc: %s' % paramLoc)
            return

        # Read file Contents
        pContents = gm.readFile( paramLoc, stripLine=True )

        if pContents == None:
            print('SIMR: WARNING: Failed to read param file')
            print('\t -paramLoc: %s' % paramLoc)
            return
        

        # Parse file contents
        argPtr = None
        args = []

        for l in pContents:

            # No content in line
            if len(l) == 0:
                continue
            
            sl = l.split()

            # Save name of this parameter file
            if sl[0] == 'paramName':
                self.name = sl[1]
                args = []

            # append contents and switch to new inputs
            elif sl[0] == 'Simulator_Input':
                argPtr = self.simArg

            elif sl[0] == 'Image_Creator_Input':
                argPtr.updateArg( args )
                args = []
                argPtr = self.imgArg

            elif sl[0] == 'Machine_Score_Input':
                argPtr.updateArg( args )
                args = []
                argPtr = self.machArg

            # gather contents if not special header
            else:
                args.extend( sl )

        argPtr.updateArg( args )
        # Finished looping throuh file contents

        if self.name != None:
            self.status = True

    # End reading param file





# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
