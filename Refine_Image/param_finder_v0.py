'''
    Author:     Matthew Ogden
    Created:    31 July 2019
    Altered:    8 Aug 2019
Description:    This code tries to find the optimal image parameters for creating an image
                    out of SPAM data and a comparison method.
'''

from sys import \
        exit, \
        argv

from os import \
        listdir, \
        path, \
        system

from importlib import import_module

import numpy as np
import multiprocessing as mp



# Global constants
printAll = True

parProc = True
nProc = 1

runDir = ''
paramDir = ''
makeDir = False

startParamLoc = ''
endParamLoc = ''
saveParamDir = 'Input_Data/image_parameters/'
paramName = 'tparam_v0_0000.txt'

imgCreatorLoc = 'Image_Creator/python/image_creator_v2.py'
imgCompareLoc = 'Comparison_Methods/compare_v1.py'

methodName = ''

useImgMod = False
imgModule = ''

useCompMod = False
compModule = ''


def main():

    endEarly = readArg()

    if printAll:
        print('Beginning Image Parameter Finder\n')
        print('Param directory: %s' % paramDir)
        print('Requesting cores: %d' % nProc)
        print('Number of cores available: %s' % mp.cpu_count())

    if endEarly:
        print('Exiting...')
        sys.exit(-1)

    # Make parameter directory if needed
    if makeDir:
        makeParamDir(paramDir,runDir)

    # Create initial parameter
    pWorking = imageParameterClass()

    if startParamLoc != '':
        pWorking.readParamFile(startParamLoc)


    initScore = scoreParam(pWorking)

    if printAll:
        print("Starting image parameter values")
        pWorking.printVal()
        print("InitScore: %f" % initScore)

    pName = [ 'gWeight','rConst', 'nVal' ]
    stepSize = [5, 0.25, 0.25 ]


    # Loop x times
    for i in range( 3 ):

        # Find best for this parameter
        for j,name in enumerate(pName):

            step = stepSize[j]
            pWorking = improveParam_v1( pWorking, name, step )

    # Exit loop through range (1)

    finalScore = scoreParam(pWorking)

    if printAll:
        print('Final Param values')
        pWorking.printVal()

        print("Init Score: %f" % initScore)
        print("Final Score: %f" % finalScore)

    if endParamLoc != '':
        pWorking.writeParam(endParamLoc)
    else:
        pWorking.writeParam(paramDir + 'tparam_1.txt')

    #p.writeParam(saveParamDir + paramName)

# End main

def scoreParam(p):
    imgLoc = paramDir + 't_img.png'

    createImg_v2(p, imgLoc)
    score = scoreImg_v1(p, imgLoc)

    return score
# End score


def scoreParam_v2(p):

    imgLoc = paramDir + 't_img.png'
    mod_createImg_v2(p, imgLoc)
    score = scoreImg_v1(p, imgLoc)

    return score
# End score

# This uses image_creator_v2.py
def mod_createImg_v2(p, imgLoc):

    if printAll:
        #print('In create Image function')
        pass

    pLoc = paramDir + "t1.txt"
    p.writeParam(pLoc)

    imgArgList = []

    imgArgList.append('-runDir')
    imgArgList.append(paramDir)
    imgArgList.append('-paramLoc')
    imgArgList.append(pLoc)
    imgArgList.append('-overwrite')
    imgArgList.append('-noprint')
    imgArgList.append('-imageLoc')
    imgArgList.append(imgLoc)

    if printAll:
        print('About to call img module: \'%s\'' % imgCreatorLoc)
        pass

    #system(imgCmd)
    imgModule.runImageCreator_v2( imgArgList )

# End creat Img_v2


# My first simple implementation for improving a parameter
def improveParam_v1( p, name, step ):

    # Get score for current val
    pVal1 = getattr( p, name ) 
    s1 = scoreParam(p)


    # Get score for increase in value
    pVal2 = pVal1 + step
    setattr( p, name, pVal2 )
    s2 = scoreParam( p )

    # Do I need to increase value?
    if s2 > s1:
        print('Increasing %s' % name)
        inc = True
        nVal = pVal2
        nScore = s2
        cScore = s1

    # Do I need to decrease value? 
    elif s1 > s2:
        print('Decreasing %s' % name)
        inc = False
        nVal = pVal1
        nScore = s1
        cScore = s2

    else:
        print("No change in score for: %s" % name)
        setattr( p, name, pVal1 )
        return p

    c = 0
    # While new value is greater than current Score
    while( nScore > cScore ):

        cVal = nVal
        cScore = nScore

        if inc:
            nVal = cVal + step
        else:
            nVal = cVal - step

        if nVal < 0:
            print("Param %s at minimum before 0: %f" % (name,cVal)) 
            break
        
        # Get new score
        setattr( p, name, nVal )
        nScore = scoreParam(p)

        c += 1
        print(c,nScore)

    # End while loop

    # reset p value back to previous 
    setattr( p, name, cVal )

    if printAll:
        print('Found %s max after %d steps: %f' % ( name, int(c), cVal) )

    return p

# End improveParam


def scoreImg_v1(p, imgLoc):
    if printAll:
        #print('In score image')
        pass


    infoLoc = paramDir + 'info.txt'
    imgParam = p.name
    scoreLoc = paramDir + 'tscore.txt'

    cmd = "python3 %s" %  imgCompareLoc
    cmd += " -runDir %s" % paramDir
    cmd += " -printScore"
    cmd += " -argFile Input_Data/comparison_methods/arg_compare_paramFinder.txt"
    cmd += " -image %s" % imgLoc
    cmd += " -imgParam %s" % p.name
    cmd += " -noprint"
    cmd += " -overWriteScore"
    cmd += " -noscore"
    
    cmd += " -%s" % methodName

    cmd += " > %s" % scoreLoc
    
    # remove previous score file if present
    if path.isfile(scoreLoc):
        system("rm %s"%scoreLoc)

    # Execute command
    system(cmd)

    fList, good = readFile(scoreLoc)

    try:
        score = float(fList[0])
    except:
        print("Failed to grab float from: %s" % fList[0] )
        exit(-1)
    else:
        return score

# End score image

# This uses image_creator_v2.py
def createImg_v2(p, imgLoc):

    if printAll:
        #print('In create Image function')
        pass

    pLoc = paramDir + "t1.txt"
    p.writeParam(pLoc)
    
    imgCmd = "python3 %s" % imgCreatorLoc
    imgCmd += " -runDir %s" % paramDir
    imgCmd += " -paramLoc %s" % pLoc
    imgCmd += " -overwrite"
    imgCmd += " -noprint"
    imgCmd += " -dotImg"
    imgCmd += " -imageLoc %s" % imgLoc

    if printAll:
        #print('About to execute: \'%s\'' % imgCmd)
        pass

    system(imgCmd)

# End creat Img_v2

'''python3 Image_Creator/python/image_creator_v2.py 
    -runDir /nfshome/mbo2d/tSpamDir/hst_Arp_273/run_0_0/ 
    -paramLoc Input_Data/image_parameters/test_param.txt 
    -argFile Input_Data/image_creator/arg_v2_test.txt
'''


def makeParamDir( paramDir, runDir ):

    if printAll:
        print('Making paramDir: %s' % paramDir)
        print('Using run dir: %s' % runDir)

    if not path.exists( runDir ):
        print('Could not find run directory to copy')
        exit(-1)

    system('mkdir -p %s' % paramDir )
    system('cp %s* %s.' % ( runDir, paramDir) ) 

    if printAll:
        print('Made directory: %s' % paramDir)


def testPrint(arg1, arg2):

    myRank = mp.current_process()._identity[0]
    print('Began - %s' % ( myRank))

# End testPrint



def readArg():

    global printAll, nProc, parProc
    global runDir, makeDir, paramDir, methodName
    global startParamLoc, endParamLoc, imgModule
    global useImgMod, imgModule
    global useCompMod, compModule

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

        elif arg == '-runDir':
            runDir = argList[i+1]
            if runDir[-1] != '/':
                runDir += '/'

        elif arg == '-newParamDir':
            makeDir = True

        elif arg == '-paramDir':
            paramDir = argList[i+1]
            if paramDir[-1] != '/':
                paramDir += '/'
 
        elif arg == '-pp':
            nProc = argList[i+1]

        elif arg == '-methodName':
            methodName = argList[i+1]

        elif arg == '-startParam':
            startParamLoc = argList[i+1]

        elif arg == '-saveParamLoc':
            endParamLoc = argList[i+1]

        elif arg == '-imgMod':
            useImgMod = True

        elif arg == '-compMod':
            useCompMod = True


    # End looping through command line arguments

    # Check if input arguments were valid

    # Check if number of processors is practical
    try:
        nProc = int( nProc )

    except:
        print('WARNING: numbers of cores requested not a number. \'%s\'' % nProc )
        endEarly = True

    else:
        if nProc > 1:
            max_CPU_cores = int(mp.cpu_count())

            if nProc > max_CPU_cores:
                print('WARNING:  Number of cores requested is greater than the number of cores available.')
                print('Requested: %d' % nProc)
                print('Available: %d' % max_CPU_cores)
                endEarly = True

    # Check if paramDirectory exists
    if not makeDir and paramDir == '':
        print('No paramDirectory given.')
        endEarly = True

    # Check if making a new directory
    if makeDir and runDir == '':
        print('Please specify inital runDir if making a new param finding directory')
        endEarly = True

    # import imageCreatorLoc as module
    if imgCreatorLoc == '':
        print('Please specify location of image creator')
        endEarly = True

    elif not path.isfile( imgCreatorLoc ):
        print('Image Creator not found at: %s' % imgCreatorLoc)
        endEarly = True

    elif useImgMod:
        try:
            imgModule = import_module( imgCreatorLoc )
        except:
            print("Failed to import image creator as python module: %s" % imageCreatorLoc)
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
        # End going through arg file 

    return argList

# end read argument file


# Define image parameter class
class imageParameterClass:

    def __init__(self):
        self.name    = 'tparam_v0'
        self.gSize   = int(25)     # gaussian size
        self.gWeight = 5         # gaussian size
        self.rConst  = 5         # radial constant
        self.bConst  = 5         # birghtness constant
        self.nVal    = 5         # normalization constant
        self.nRow    = int(800)   # number of rows
        self.nCol    = int(1200)   # number of col
        self.gCenter = np.array( [[ 400, 800 ],      # [[ x1, x2 ] 
                             [ 400, 400 ]])     #  [ y1, y2 ]]
        self.comment = 'blank comment'

        # Step size for numerical derivative
        self.h_gSize = 1
        self.h_gWeight = 0.05
        self.h_rConst = 0.05
        self.h_bConst = 0.05
        self.h_nVal = 0.05
    # end init

    def readParamFile(self, pInLoc):

        pFile, fileGood = readFile(pInLoc)

        for line in pFile:
            l = line.strip()
            if len(l) == 0:
                continue
            
            pL = l.split()

            if pL[0] == 'gaussian_size':
                self.gSize = int(pL[1])   

            if pL[0] == 'gaussian_weight':
                self.gWeight = float(pL[1])   

            if pL[0] == 'radial_constant':
                self.rConst = float(pL[1])   

            if pL[0] == 'brightness_constant':
                self.bConst = float(pL[1]) 

            if pL[0] == 'norm_value':
                self.nVal = float(pL[1]) 

            if pL[0] == 'image_rows':
                self.nRow = float(pL[1]) 

            if pL[0] == 'image_cols':
                self.nCol = int(pL[1]) 

            if pL[0] == 'galaxy1_center':
                self.gCenter[0,0] = int(pL[1])
                self.gCenter[1,0] = int(pL[2])

            if pL[0] == 'galaxy2_center':
                self.gCenter[0,1] = int(pL[1])
                self.gCenter[1,1] = int(pL[2])

    # end read param file


    def printVal(self):
        print(' Name: %s' % self.name)
        print(' Comment: %s' % self.comment)
        print(' Gaussian size: %d' % self.gSize)
        print(' Gaussian weight: %f' % self.gWeight)
        print(' Radial constant: %f' % self.rConst)
        print(' Brightness constant: %f' % self.bConst)
        print(' Normalization constant: %f' % self.nVal)
        print(' Number of rows: %d' % self.nRow)
        print(' Number of columns: %d' % self.nCol)
        print('galaxy 1 center %d %d' % ( int(self.gCenter[0,0]), int(self.gCenter[0,1]) ))
        print('galaxy 2 center %d %d' % ( int(self.gCenter[1,0]), int(self.gCenter[1,1]) ))
    # end print

    def writeParam(self, saveLoc):
        try:
            pFile = open(saveLoc,'w')
        except:
            print('Failed to create: %s' % saveLoc)
        else:
            pFile.write('parameter_name %s\n' % self.name)
            pFile.write('comment %s\n\n' % self.comment)
            pFile.write('gaussian_size %d\n' % self.gSize)
            pFile.write('gaussian_weight %f\n' % self.gWeight)
            pFile.write('radial_constant %f\n' % self.rConst)
            pFile.write('brightness_constant %f\n' % self.bConst)
            pFile.write('norm_value %f\n' % self.nVal)
            pFile.write('image_rows %d\n' % self.nRow)
            pFile.write('image_cols %d\n' % self.nCol)
            pFile.write('galaxy_1_center %d %d\n' % ( int(self.gCenter[0,0]), int(self.gCenter[0,1]) ))
            pFile.write('galaxy_2_center %d %d\n' % ( int(self.gCenter[1,0]), int(self.gCenter[1,1]) ))
            pFile.close()


# End parameter class

def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print('File does not exist: %s' % fileLoc)
        return [], False

    try:
        inFile = open(fileLoc,'r')

    except:
        print('Failed to open/read: %s' % fileLoc)
        return [], False

    else:
        inList = list(inFile)
        inFile.close()
        return inList, True

# End simple file read

main()

# Example on how to begin a parrel environment
'''
processes = []
arg1 = 'temp_arg'
arg2 = 15

# Start all processes
for i in range( nProc ):
    p = mp.Process( target=testPrint, args=( arg1, arg2, ) )
    processes.append( p )
    p.start()

# Wait until all processes are complete
for p in processes:
    p.join()
'''
