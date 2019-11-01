'''
    Author:     Matthew Ogden
    Created:    30 Oct 2019
    Altered:    
Description:    This is my attempt at making a general purpose parallel processing module
'''

import multiprocessing as mp

# Global input arguments

class ppClass:

    nCores = 1
    printProg = False

    jobQueue = mp.Queue()
    nQueue = 0

    lock = mp.Lock()

    funcPtr = None

    def __init__(self, nCore):

        self.nCores = nCore
        print(self.nCores)

    def printProgBar(self):
        self.printProg = True


    # Assuming you're only running a single function
    def loadQueue( self, funcIn, inList ):

        self.funcPtr = funcIn
        self.nQueue = len( inList )

        for args in inList:

            self.jobQueue.put(args)


    def runCores( self ):

        if self.nCores == 1:
            print("Why use parallel processing with 1 core?")
            return1

        coreList = []

        # Start all processes
        for i in range( self.nCores ):
            p = mp.Process( target=self.coreFunc1 )
            coreList.append( p )
            p.start()

        # Wait until all processes are complete
        for p in coreList:
            p.join()

    
    def coreFunc1( self ):

        # Keep core running until shared queue is empty
        while True:

            try:
                funcArgs = self.jobQueue.get_nowait()
            
            # Will exist loop if queue is empty
            except:

                # WTF?!  
                if not self.jobQueue.empty:
                    print('%s - queue empty' % mp.current_process().name)
                    break
                else:
                    continue

            if self.printProg:
                p = int(self.nQueue) - int(self.jobQueue.qsize())
                perc = ( p / int(self.nQueue) ) * 100
                print("%.1f - %d / %d" % ( perc, p, self.nQueue ), end='\r' )

            # Run desired function on core
            self.funcPtr(funcArgs)

    # End exectute function

 
    def coreFunc2( self ):

        # Keep core running until shared queue is empty
        while True:

            try:
                funcArgs = self.jobQueue.get_nowait()
            
            # Will exist loop if queue is empty
            except:
                print('%s - queue empty' % mp.current_process().name)
                break

            if self.printProg:
                p = int(self.nQueue) - int(self.jobQueue.qsize())
                perc = ( p / int(self.nQueue) ) * 100
                print("%.1f - %d / %d" % ( perc, p, self.nQueue ), end='\r' )

            # Run desired function on core
            self.funcPtr(*funcArgs)

    # End exectute function

# End parallel processing class

def testPrint():
    print("Inside parallel processing python Module.  Written by Matthew Ogden")

def printVal( n1, n2 ):
    import time
    #print("Val: %d %d" % ( n1, n2))
    time.sleep(1)


if __name__ == '__main__':
    print("This is an example method for using the pp Module class")

    nCores = 2
    pHolder = ppClass( nCores )

    pHolder.printProgBar()

    argList = []

    for i in range( 4 ): 
        argList.append(( i, i ))

    pHolder.loadQueue( printVal, argList )

    pHolder.runCores()


