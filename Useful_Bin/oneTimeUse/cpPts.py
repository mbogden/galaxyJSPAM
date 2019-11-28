'''
    Author:     Matthew Ogden
    Created:    10 Oct 2019
    Altered:    11 Oct 2019
Description:    This program will move existing points to new format for pipeline
'''

from sys import \
        exit, \
        argv, \
        stdout

from os import \
        path, \
        listdir, \
        system

import queue
import multiprocessing as mp


# Global input arguments

nProc = 1

printAll = True

fromDir = ''
toDir = ''
sDirIn = ''
rm = False

def main():

    endEarly = readArg()

    if printAll:
        print("fromDir: %s" % fromDir)
        print("toDir: %s" % toDir)
        print("nProc: %d" % nProc)
        if rm:
            print("Removing old zip...")
        else:
            print("Copying...")

    if endEarly:
        print('Exiting...')
        exit(-1)

    if nProc == 1:
        print("Going through dirs:")
        return
        toSdssDirs = listdir(toDir)
        fromSdssDirs = listdir(fromDir)

        for sDir in toSdssDirs:

            toSdssDir = toDir + sDir + '/gen000/'
            sName = sDir
            
            if sName in fromSdssDirs:
                fromSdssDir = fromDir + sName + '/'
            else:
                print("Failed to find %s" % sName)
                break

            print('\t',sDir)
            sdss_dir(toSdssDir, fromSdssDir)



    else:
        print("Using parallel processing")

        # paralle stuff
        jobQueue = mp.Queue()
        lock = mp.Lock()
        processes = []

        # job stuff
        if sDirIn != '':
            toSdssDirs =  [ sDirIn ]
        else:
            toSdssDirs = listdir(toDir)

        fromSdssDirs = listdir(fromDir)

        jobList = []

        # create list of jobs
        for sDir in toSdssDirs:

            if sDirIn != '':
                toSdssDir = sDirIn + '/gen000/'
                sName = sDirIn.split('/')[-2]
            else:
                toSdssDir = toDir + sDir + '/gen000/'
                sName = sDir

            
            if sName in fromSdssDirs:
                fromSdssDir = fromDir + sName + '/'
            else:
                print("Failed to find %s" % sName)
                break

            print('\t',sDir)


            #sdss_dir(toSdssDir, fromSdssDir)
            jobQueue.put( ( toSdssDir, fromSdssDir ) )


        # Start all processes
        for i in range( nProc ):
            p = mp.Process( target=execute, args=( jobQueue, lock, ) )
            processes.append( p )
            p.start()


        # Wait until all processes are complete
        for p in processes:
            p.join()

        print('')


    
# End main

def sdss_dir( tSdssDir, fSdssDir ):

    runDirList = listdir( tSdssDir )
    runDirList.sort()

    for rDir in runDirList:
        runDir = tSdssDir + rDir + '/'

        if not path.exists( runDir ):
            print("runDir doesn't exist: %s" % runDir)
            continue

        if 'run' not in runDir:
            print("run not in dir: %s" % runDir)
            continue

        rNum = int(rDir.split('_')[1])

        fromRunDir = fSdssDir + 'run_0_%d/' % rNum
        if not path.exists(fromRunDir):
            print("WARNING: dir not found: %s" %fromRunDir)
            continue
        
        run_dir( runDir, fromRunDir )
        
    return False

# End sdss_dir

def execute( jobQueue, lock ):

    # Keep going until shared queue is empty
    while True:

        try:
            tSdssDir, fSdssDir = jobQueue.get_nowait()
        
        # Will raise empty queue if empty
        except queue.Empty:
            print('%s - queue empty' % mp.current_process().name)
            break

        else:
            sdss_dir( tSdssDir, fSdssDir )
            
# End exectute function


def run_dir(tDir, fDir):

    toPtsDir = tDir + 'particle_files/'
    zipLoc = toPtsDir+'100000_pts.zip'

    if path.exists(zipLoc):
        return


    fInfo = readFile( fDir + 'info.txt')
    tInfo = readFile( tDir + 'info.txt')

    if len(fInfo) == 0:
        print("Info files empty in run: %s" % tDir)
        return 
 
    if len(tInfo) == 0:
        print("Info files empty in run: %s" % tDir)
        return 
 
    infoMatch = checkInfoFiles( fInfo, tInfo )
    
    if not infoMatch:
        print("Warning: Info files don't match")
        print("From: %s" % fDir)
        print("  To: %s" % tDir)
        return
   
    fFiles = listdir( fDir )
    tFiles = listdir( tDir )

    zip1 = ''
    zip2 = ''

    for f in fFiles:
        if 'pts_000.zip' in f:
            zip1 = fDir  + f
        if 'pts_101.zip' in f:
            zip2 = fDir + f

    if zip1 == '' or zip2 == '':
        print("Failed to find particle files in: %s" % fDir)
        return

    toPtsDir = tDir + 'particle_files/'
    if not path.exists(toPtsDir):
        print("Failed to find particles dir in %s" % toPtsDir)
        return

    unZcmd1 = 'unzip -o %s -d %s' % ( zip1, toPtsDir)
    unZcmd2 = 'unzip -o %s -d %s' % ( zip2, toPtsDir)
    system(unZcmd1)
    system(unZcmd2)

    ptsFiles = listdir( toPtsDir )
    print(ptsFiles)

    pts1 = ''
    pts2 = ''

    for f in ptsFiles:
        if '.000' in f:
            pts1 = toPtsDir  + f
        if '.101' in f:
            pts2 = toPtsDir + f

    if pts1 == '' or pts2 == '':
        print("Failed to find pt files after unzip")
        print("pts1: " , pts1)
        print("pts2: " , pts2)
        return

    mvCmd1 = 'mv %s %s100000_pts.000' % ( pts1, toPtsDir)     
    mvCmd2 = 'mv %s %s100000_pts.101' % ( pts2, toPtsDir)     

    print(mvCmd1)
    print(mvCmd2)


    system(mvCmd1)
    system(mvCmd2)

    ptsFiles = listdir( toPtsDir )
    print(ptsFiles)

    zipCmd = 'zip -j -q %s100000_pts.zip %s100000_pts.000 %s100000_pts.101' % ( toPtsDir, toPtsDir, toPtsDir)
    print(zipCmd)
    system(zipCmd)

    if rm:
        rmCmd = 'rm %s100000_pts.000 %s100000_pts.101 %s %s' % (toPtsDir, toPtsDir, zip1, zip2)
    else:
        rmCmd = 'rm %s100000_pts.000 %s100000_pts.101' % (toPtsDir, toPtsDir)
    print(rmCmd)
    system(rmCmd)

    ptsFiles = listdir( toPtsDir )
    print(ptsFiles)



# End run dir

def checkInfoFiles( fInfo, tInfo ):

    fModel = ''
    tModel = ''

    for l in fInfo:
        if 'model_data' in l:
            fModel = l

    for l in tInfo:
        if 'model_data' in l:
            tModel = l

    if fModel == '' or tModel == '':
        print("Warning: Info files don't match")
        print("fModel: %s" % fModel)
        print("tModel: %s" % tModel )
        return False
    
    fModel = fModel.split(' ')[1].strip()
    tModel = tModel.split(' ')[1].strip()

    if fModel == tModel:
        return True
    else:
        print("Warning: Info files don't match")
        print("fModel: %s" % fModel)
        print("tModel: %s" % tModel )
        return False


 
def readArg():
    global printAll, fromDir, toDir, sDirIn, nProc, rm
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

        elif arg == '-sdssDir':
            sDirIn = argList[i+1]
            if sDirIn[-1] != '/':
                sDirIn += '/'
        
        elif arg == '-pp':
            nProc = int( argList[i+1] )

        elif arg == '-rm':
            rm = True


    # Check if input arguments were valid

    if not path.exists(fromDir):
        print('\tfromDir not found: \'%s\'' % fromDir)
        endEarly = True

    if not path.exists(toDir) and not path.exists(sDirIn):
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


def pp_main():

    try:
        inFile = open( fileLoc, 'r' )
    except:
        print('Failed to open file \'%s\'' % fileLoc )
        print('Exiting...')
        sys.exit(-1)
    
    inList = list( inFile )

    if reverse:
        inList.reverse()

    if shuffleList:
        shuffle( inList)

    lenList = len( inList )

    jobQueue = mp.Queue()
    lock = mp.Lock()

    processes = []


    # Populate queue with cmds from file
    for i,cmd in enumerate(inList):
        cmd = cmd.strip()
        jobQueue.put(( i, lenList, cmd))


    if printAll:
        printArg(lenList)

    # Start all processes
    for i in range( nProc ):
        p = mp.Process( target=execute, args=( jobQueue, lock, ) )
        processes.append( p )
        p.start()


    # Wait until all processes are complete
    for p in processes:
        p.join()

    print('')

# End main


# Run main after declaring functions
main()

