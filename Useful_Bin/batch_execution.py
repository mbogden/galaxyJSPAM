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


import os
import sys
import queue
import multiprocessing as mp

# Global input arguments

fileLoc = ''

parProc = False
nProc = 1

printAll = False
printAll = True     # Keep true for inital development

recordProg = False
recordLoc = ''

insert_uID = False
offset_uID = 0

reverse = False


def main():

    exitEarly = readArg()

    if exitEarly:
        print('Exiting...')
        sys.exit(-1)

    try:
        inFile = open( fileLoc, 'r' )
    except:
        print('Failed to open file \'%s\'' % fileLoc )
        print('Exiting...')
        sys.exit(-1)
    
    inList = list( inFile )
    if reverse:
        inList.reverse()
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


# End main


def execute( jobQueue, lock ):

    # Keep going until shared queue is empty
    while True:

        try:
            i, n, cmd = jobQueue.get_nowait()
        
        # Will raise empty queue if empty
        except queue.Empty:
            print('%s - queue empty' % mp.current_process().name)
            break

        else:
            

            if insert_uID:
                cmd = cmd.replace('-uID', '-uID %d' % (mp.current_process()._identity[0] + offset))

            print('%d about to execute \'%s\'' % ( mp.current_process()._identity[0], cmd))

            os.system(cmd)
            printStr = '%s completed %d / %d - \'%s\'' % \
                        ( mp.current_process().name, i, n, cmd ) 

            if printAll:
                print(printStr)

            if recordProg:
                lock.acquire()
                oFile = open( recordLoc, 'a')
                oFile.write(printStr + '\n')
                oFile.close()
                lock.release()

# End exectute function

def printArg( lenList ):
    print('Using input file \'%s\'' % fileLoc)
    print('Found %d lines in file' % lenList)
    
    if nProc > 1:
        print('Using %d cores to run'% nProc)

    if recordProg:
        print('Recording progress to file \'%s\'' % recordLoc )

# End print Arguments



def readArg():
    global fileLoc, parProc, nProc, insert_uID, offset
    global printAll, recordProg, recordLoc, reverse

    for i,arg in enumerate(sys.argv):

        # ignore arguments without specifier
        if arg[0] != '-':
            continue

        elif arg == '-i':
            fileLoc = sys.argv[i+1]

        # Would you like parallel processing on this machine? 
        elif arg == '-pp':
            parProc = True
            nProc = sys.argv[i+1]

        # Would you like to print / not print progress? 
        elif arg == '-print':
            printAll = True

        # Would you like to record progress in file? 
        elif arg == '-write':
            recordProg = True
            recordLoc = sys.argv[i+1]

        # Tells batch_execution to insert uID where found
        elif arg == '-uID':
            insert_uID = True
        
        #Offset uID
        elif arg == '-offset':
            offset = int(sys.argv[i+1])

        # reverse through list
        elif arg == '-r':
            reverse = True

    # End looping through arguments



    # Check if input arguments are valid
    exitEarly = False

    if not os.path.isfile(fileLoc):
        print('No file found at \'%s\'.' % fileLoc)
        exitEarly = True

    try:
        nProc = int( nProc )

        if nProc > 1:
            max_CPU_cores = mp.cpu_count()

            if nProc > int(max_CPU_cores):
                print('WARNING:  Number of cores requested is greater than the number of cores available.')
                print('Requested: %d' % nProc)
                print('Available: %d' % int(max_CPU_cores))
                exitEarly = True

    except:
        print('WARNING: numbers of cores requested not a number. \'%s\'' % nProc )
        exitEarly = True

    return exitEarly

# End read Arguments

# Execute main after functions defined
main()
