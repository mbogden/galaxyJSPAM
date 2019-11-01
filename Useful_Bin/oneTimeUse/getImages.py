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
        listdir, \
        system

printAll = True

fromDir = ''
toDir = ''

def main():

    endEarly = readArg()

    if printAll:
        print("fromDir: %s" % fromDir)
        print("toDir: %s" % toDir)

    if endEarly:
        print('Exiting...')
        exit(-1)

    print("Going through dirs:")

    pl_sdss_dir(fromDir)

# End main

def pl_sdss_dir( fSdssDir ):

    fDir = fSdssDir + 'gen000/'
    runDirList = listdir( fDir )
    runDirList.sort()

    for rDir in runDirList:
        runDir = fDir + rDir + '/'

        if not path.exists( runDir ):
            print("runDir doesn't exist: %s" % runDir)
            continue

        if 'run' not in runDir:
            print("Not a run dir: %s" % runDir)
            continue

        rNum = int(rDir.split('_')[1])

        pl_run_dir( rNum, runDir )
        
# End pipeline sdss_dir

def pl_run_dir(rNum, fDir):

    fInfoLoc = fDir + 'info.txt'
    imgDir = fDir + 'model_images/'


    modelLoc = ''
    initLoc = ''

    fFiles = listdir( imgDir )
    for f in fFiles:
        if 'model.png' in f:
            modelLoc = imgDir  + f
        if 'init.png' in f:
            initLoc = imgDir + f

    if modelLoc == '' or initLoc == '':
        print("Failed to find images in: %s" % fDir)
        print("modelLoc: %s" % modelLoc)
        print("initLoc: %s" % initLoc)
        return

    cpCmd1 = 'cp %s %s' % ( fInfoLoc, toDir + '%04d_info.txt' % rNum)
    cpCmd2 = 'cp %s %s' % ( modelLoc, toDir + '%04d_model.png' % rNum)
    cpCmd3 = 'cp %s %s' % ( initLoc, toDir + '%04d_init.png' % rNum)

    system(cpCmd1)
    system(cpCmd2)
    system(cpCmd3)

# End run dir



def sdss_dir( fSdssDir ):

    runDirList = listdir( fSdssDir )
    runDirList.sort()

    for rDir in runDirList:
        runDir = fSdssDir + rDir + '/'

        if not path.exists( runDir ):
            print("runDir doesn't exist: %s" % runDir)
            continue

        if 'run' not in runDir:
            print("run not in dir: %s" % runDir)
            continue

        rNum = int(rDir.split('_')[2])
        print(rNum,rDir)

        run_dir( rNum, runDir )
        
# End sdss_dir

def run_dir(rNum, fDir):

    fInfoLoc = fDir + 'info.txt'
    fFiles = listdir( fDir )

    modelLoc = ''
    initLoc = ''

    for f in fFiles:
        if 'model.png' in f:
            modelLoc = fDir  + f
        if 'init.png' in f:
            initLoc = fDir + f

    if modelLoc == '' or initLoc == '':
        print("Failed to find images in: %s" % fDir)
        print("modelLoc: %s" % modelLoc)
        print("initLoc: %s" % initLoc)
        return

    cpCmd1 = 'cp %s %s' % ( fInfoLoc, toDir + '%04d_info.txt' % rNum)
    cpCmd2 = 'cp %s %s' % ( modelLoc, toDir + '%04d_model.png' % rNum)
    cpCmd3 = 'cp %s %s' % ( initLoc, toDir + '%04d_init.png' % rNum)

    system(cpCmd1)
    system(cpCmd2)
    system(cpCmd3)

# End run dir

def checkInfoFiles( fInfo, tInfo ):

    fModel = ''
    tModel = ''

    for l in fInfo:
        if 'model_data' in l:
            fModel = l

    for l in tInfo:
        if 'model data' in l:
            tModel = l

    if fModel == '' or tModel == '':
        print("Warning: Info files don't match")
        print("fModel: %s" % fModel)
        print("tModel: %s" % tModel )
        return False
    
    fModel = fModel.split(' ')[1].strip()
    tModel = tModel.split(' ')[2].strip()

    if fModel == tModel:
        return True
    else:
        print("Warning: Info files don't match")
        print("fModel: %s" % fModel)
        print("tModel: %s" % tModel )
        return False


 
def readArg():
    global printAll, fromDir, toDir 
    endEarly = False

    argList = argv

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-noprint':
            printAll = False

        elif arg == '-fromDir':
            fromDir = argList[i+1]
            if fromDir[-1] != '/':
                fromDir += '/'

        elif arg == '-toDir':
            toDir = argList[i+1]
            if toDir[-1] != '/':
                toDir += '/'

    # Check if input arguments were valid

    if not path.exists(fromDir):
        print('\tfromDir not found: \'%s\'' % fromDir)
        endEarly = True

    if not path.exists(toDir):
        print('\ttoDir not found: \'%s\'' % toDir)
        endEarly = True

    return endEarly

# End reading command line arguments

def readFile( fileLoc ):

    if not path.exists( fileLoc ):
        print("File not found: %s" % fileLoc)
        return []

    try:
        fileIn = open(fileLoc,'r')

    except:
        print("File failed to open: %s" % fileLoc)
        return []

    else:
        fileContents = list(fileIn)
        fileIn.close()
        return fileContents

# Run main after declaring functions
main()
