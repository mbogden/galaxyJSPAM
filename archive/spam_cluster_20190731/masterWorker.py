#   Author:     Matthew Ogden
#   Created:    25 JUN 2019
#   Altered:    28 JUN 2019
#
# This code is meant to run SPAM on our local cluster

import os
from sys import argv
from sys import exit
from subprocess import call

myRank = -1
nProc = -1

nPart = 100000
pName = '100k'

locSpamDir = 'spam_cluster'
SIMR_dir = 'localstorage/mbo2d/galaxyJSPAM/'

SPAM_src_dir = SIMR_dir + 'SPAM/'
model_src = SIMR_dir + 'Input_Data/zoo_models/bigCompList.txt'
    

spam_data_loc  = 'localstorage/public/SPAM_data/%s/' % pName
print(spam_data_loc)

if not os.path.exists(spam_data_loc):
    print('\'%s\' does not exist.  Please create first')
    #exit('-1')
    call([ 'mkdir','-p',spam_data_loc])

def main():

    # Read arguments
    readArg()
    print('py %2d %2d: In gal python script'%(myRank,nProc))


    # Create director for SPAM info if needed
    if not os.path.exists(locSpamDir):
        try:
            os.makedirs(locSpamDir)
        except:
            pass

    # Copy SPAM code to a unique directory
    mySPAMdir = locSpamDir + '/%s' % str(myRank).zfill(3)

    if os.path.exists(mySPAMdir):
        call( [ 'rm', '-rf', mySPAMdir ] )

    call( [ 'cp', '-r', SPAM_src_dir, mySPAMdir ] )

    call( [ 'make', '-C', mySPAMdir ] )

    # Copy file of models
    myModLoc = mySPAMdir + '/modelFile.txt' 
    call( [ 'cp', '-r', model_src, myModLoc] )

    # check if needed files are ready
    basicExec = mySPAMdir+'/bin/basic_run'
    zooPyExec = mySPAMdir+'/bin/zooRun.py'
    if os.path.isfile(basicExec) and os.path.isfile(zooPyExec):
        pass
    else:
        print('###  WARNING  ###')
        print('Did not find spam executable files')
        return -1

    # Open Model File
    mFile = open(myModLoc,'r')

    #nLine = len(mFile)

    for i,l in enumerate(mFile):

        l = l.strip()
        sdssName, genNum, runNum, score, spamData = l.split(' ')

        # This is my quick way of charing work across cores.
        if i % nProc == myRank:
            

            oDir = spam_data_loc + sdssName + '/'

            pyCmd = 'python %s -compress -name %s -spam %s -uID %d -n %d -o %s -sdss %s -gen %s -run %s -createDir -m %s' % ( zooPyExec, pName, basicExec, myRank, nPart, oDir, sdssName, genNum, runNum, spamData  )
            #pyCmd = 'time python pyTest.py'

            myProgFile = open(mySPAMdir + '/prog.txt', 'a')
            myProgFile.write('%6d : %s\n' %(i,pyCmd) )
            myProgFile.close()

            os.system(pyCmd)


    # End mFile loop

    mFile.close()
    


def readArg():
    global myRank, nProc

    for i,arg in enumerate(argv):
        if (arg == '-cr'):
            myRank = int(argv[i+1])
        elif (arg == '-cs'):
            nProc = int(argv[i+1])


main()
