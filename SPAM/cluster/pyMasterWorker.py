#   Author:     Matthew Ogden
#   Created:    25 JUN 2019
#   Altered:    28 JUN 2019
#
# This code is meant to run SPAM on our local cluster

import os
import time
from sys import argv
from sys import exit
from subprocess import call

myRank = -1
nProc = -1

nPart = 100000
pName = '100k'

offset = 200

coreCount = []

reverse = False
reverse = True

locSpamDir = 'spam_cluster/'
mySpamDir = ''
modName = 'myModelFile.txt'
myModLoc = ''

basicExec = ''
zooPyExec = ''

SIMR_dir = 'localstorage/mbo2d/galaxyJSPAM/'

model_src = SIMR_dir + 'Input_Data/zoo_models/bigCompList.txt'

spam_data_loc  = 'localstorage/public/SPAM_data/%s/' % pName

if not os.path.exists(spam_data_loc):
    print('\'%s\' does not exist.  Please create first')
    #exit('-1')
    call([ 'mkdir','-p',spam_data_loc])


def main():

    # Read arguments
    readArg()

    #temp measure while other job is working on cluster
    global myRank
    myRank = myRank + offset


    print('pyMasterWorker.py %2d : Entering program.'%(myRank))

    prepDirectory()

    if myRank == offset:
        print('pyMasterWorker.py %d - I am the master task giver.' % myRank)

        # Create list of counts cores have done
        for i in range( nProc):
            coreCount.append(0)

        bigFile = list( open( model_src, 'r') )
        nList = len(bigFile)

        if reverse:
            bigFile = reversed( bigFile )

        for i, l in enumerate( bigFile ):

            l = l.strip()

            print(i,l)

            lineDone = False
            while not lineDone:

                # go through workers directories to give someone new model line
                for workerNum in range(1 + offset, nProc + offset):

                    workerCompLoc = locSpamDir + '%s/' % str(workerNum).zfill(3) + modName

                    if not os.path.isfile(workerCompLoc):

                        # Create new file with model
                        wFile = open(workerCompLoc, 'w')
                        wFile.write(l)
                        wFile.close()
                        lineDone = True
                        
                        # Write progress 
                        coreCount[ workerNum - offset] += 1
                        writeProg(i, nList, l, coreCount)
                        break # break to next line item

                    # End write file if not present

                # If no one got line, wait and repeat
                if not lineDone:
                    time.sleep(2)

            # End not lineDone

            # Troubleshoot break
            if i > 6:
                break

        # Primary loop through file

        print('pyMasterWorker.py %d - Master - Sending completion flag to workers' % myRank)
        for i in range(1+offset, nProc+offset):

            workerCompLoc = locSpamDir + '%s/' % str(i).zfill(3) + 'completed.txt'
            compFile = open( workerCompLoc, 'w' )
            compFile.write('completed')
            compFile.close()

    # End master

    else:
        print('pyMasterWorker.py %d - I am a productive worker.  Unlike master...' % myRank)
        fileFound = findFile()  # Will wait for 30 sec to find File

        if fileFound:
            keepWorking, sdssName, genNum, runNum, data = readFile()  

        else:
            keepWorking = False
            print('pyMasterWorker.py %d - model file \'%s\' not found after 30 seconds.' % (myRank, myModLoc))

        while keepWorking:

            runZooPy(sdssName, genNum, runNum, nPart, data)

            fileFound = findFile() 

            if fileFound:
                keepWorking, sdssName, genNum, runNum, data = readFile()  
            else:
                keepWorking = False
                print('pyMasterWorker.py %d - model file \'%s\' not found after 30 seconds' % (myRank, myModLoc))

        # end while keepWorking

        print('pyMasterWorker.py %d - Worker done working' % myRank)

    print('pyMasterWorker.py %d - exiting program'% myRank)

    # end Worker 
# End main

def writeProg( i, nList, l, coreCount):
    pFile = open('myProg.txt','a')
    pFile.write(' %d - %d : %s \n' % (i, nList, l))
    pFile.close()


# Will run for 30 second checking for file
def findFile():

    initTime = time.time()

    if os.path.isfile(myModLoc):
        return True

    # Check if completion path is there
    if os.path.isfile(mySPAMdir + 'completed.txt'):
        print('pyMasterWorker.py %d - Worker found completion file.  Ending' % myRank)
        return False

    while not os.path.isfile(myModLoc):
   
        elapsedTime = time.time() - initTime

        # Check if completion path is there
        if os.path.isfile(mySPAMdir + 'completed.txt'):
            print('pyMasterWorker.py %d - Worker - Found completion flag.  Ending' % myRank)
            return False

        if elapsedTime > 30:
            break 

        time.sleep(2)

    # End while loop

    # one last check just in case
    if os.path.isfile(myModLoc):
        return True
    else:
        return False

#end find file


def readFile():
    mFile = list( open(myModLoc,'r') )
    l = mFile[0].strip()

    try:
        sdssName, genNum, runNum, score, mData = l.split(' ')
    else:
        print('pyMasterWorker.py %d - Worker - Error reading \'%s\'' % ( myRank, l) )
        return False, 0, 0, 0, 0

    mFile.close()

    rmCmd = 'rm %s' % myModLoc
    os.system(rmCmd)

    return True, sdssName, genNum, runNum, mData

# end read file



def runZooPy(sdssName, genNum, runNum, nPart, data):
        
    oDir = spam_data_loc + sdssName + '/'

    pyCmd = 'python %s -compress -name %s -spam %s -uID %d -n %d -o %s -sdss %s -gen %s -run %s -createDir -m %s' % ( zooPyExec, pName, basicExec, myRank, nPart, oDir, sdssName, genNum, runNum, data  )

    os.system(pyCmd)

# end reZooPy


# Prepare local diskspace for performing everything
# This is likely redundant, but just in case nodes in cluster don't share memory space
def prepDirectory():
    global mySPAMdir, myModLoc, zooPyExec, basicExec

    # Create director for SPAM info if needed
    if not os.path.exists(locSpamDir):
        try:
            os.makedirs(locSpamDir)
        except:
            pass

    # Copy SPAM code to a unique directory
    mySPAMdir = locSpamDir + '%s/' % str(myRank).zfill(3)
    myModLoc = mySPAMdir + modName

    if os.path.exists(mySPAMdir):
        call( [ 'rm', '-rf', mySPAMdir ] )

    call( [ 'cp', '-r', SIMR_dir + 'SPAM/', mySPAMdir ] )

    call( [ 'make', '-C', mySPAMdir ] )

    # check if needed files are ready
    basicExec = mySPAMdir+'bin/basic_run'
    zooPyExec = mySPAMdir+'bin/zooRun.py'
    if os.path.isfile(basicExec) and os.path.isfile(zooPyExec):
        pass
    else:
        print('###  WARNING  ###')
        print('Did not find spam executable files in \'%s\' or \'%s\'' % (basicExec, zooPyExec))
        print('Exiting masterWorker.py')
        exit(-1)

# end prep dir


def readArg():
    global myRank, nProc

    for i,arg in enumerate(argv):
        if (arg == '-cr'):
            myRank = int(argv[i+1])
        elif (arg == '-cs'):
            nProc = int(argv[i+1])
        elif (arg == '-r'):
            reverse = True


main()
