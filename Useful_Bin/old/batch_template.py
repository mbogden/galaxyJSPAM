'''
    Author:     Matthew Ogden
    Created:    15 July 2019
    Altered:    15 July 2019
Description:    This program is meant to do one thing since I find myself doing 
                something very similar in many seperate programs over and over again.
                This program will read in a file, loop through the lines,
                and execute each line as a system command.
                This program will takes care of parallel processing or 
                any other out of program needs.
'''


from os import \
        path, \
        system
import sys
import platform
from random import shuffle
import multiprocessing as mp

# Global input arguments

parProc = True
nProc = 1

printAll = False
printAll = True     # Keep true for inital development


def main():

    exitEarly = readArg()

    if printAll:
        print('Beginning main')
        print('Requesting cores: %d' % nProc)
        print('Number of cores available: %s' % mp.cpu_count())
        print("Main running on PC: %s" % mp.current_process().name)
        print("PC Name: %s" % platform.node())

    if exitEarly:
        print('Exiting...')
        sys.exit(-1)

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

# End main

def testPrint(arg1, arg2):

    cP = mp.current_process()
    myName = cP.name
    myId = cP._identity
    myRank = cP._identity[0]
    pcName = platform.node()

    print('%s - %s : %s' % ( myRank, myName, pcName))

# End testPrint

def readArg():
    global nProc, parProc
    global printAll

    argList = sys.argv
    print(argList)

    for i,arg in enumerate(argList):

        # ignore arguments without specifier
        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            arg = readArgFile(argList, argFileLoc)
        
        # Would you like parallel processing on this machine? 
        elif arg == '-pp':
            parProc = True
            nProc = argList[i+1]

        # Would you like to print / not print progress? 
        elif arg == '-print':
            printAll = True

    # End looping through arguments


    # Check if input arguments are valid
    exitEarly = False

    try:
        nProc = int( nProc )

        if nProc > 1:
            max_CPU_cores = int(mp.cpu_count())

            if nProc > max_CPU_cores:
                print('WARNING:  Number of cores requested is greater than the number of cores available.')
                print('Requested: %d' % nProc)
                print('Available: %d' % max_CPU_cores)
                exitEarly = True

    except:
        print('WARNING: numbers of cores requested not a number. \'%s\'' % nProc )
        exitEarly = True

    return exitEarly

# End read Arguments

def readArgFile( argList, argFileLoc):
    
    if not path.isfile( argFileLoc):
        print('Warning:  Could not find \'%s\'' % argFileLoc)
        return argList

    try:
        argFile = open( argFileLoc, 'r')
    except:
        print('Warning:  Could not open \'%s\'' % argFileLoc)
        return argList
    else:
        for l in argFile:
            l = l.strip()
            if l[0] == '#':
                continue
            lineItems = l.split()

            for item in lineItems:
                argList.append(item)
    return argList
#End read arg file

# Execute main after functions defined
main()
