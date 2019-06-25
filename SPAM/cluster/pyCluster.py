#   Author:     Matthew Ogden
#   Created:    25 JUN 2019
#
# This code is meant to do most of the prep for running SPAM on our local cluster

from sys import argv
from subprocess import call

myRank = -1
nProc = -1

SPAM_src_dir = 'localstorage/mbo2d/galaxyJSPAM/SPAM/'

print("Starting python")


def main():

    # Read arguments
    readArg()
    print('py %2d %2d: In gal python script'%(myRank,nProc))

    # Copy SPAM to a unique directory
    mySPAMdir = 'SPAM_%s' % str(myRank).zfill(2)
    call( [ 'cp', '-r', SPAM_src_dir, mySPAMdir ] )

    # Call make file
    call( [ 'make', '-C', mySPAMdir ] )


def readArg():
    global myRank, nProc

    for i,arg in enumerate(argv):
        if (arg == '-cr'):
            myRank = int(argv[i+1])
        elif (arg == '-cs'):
            nProc = int(argv[i+1])



print('')
main()
print('Check point',myRank)
