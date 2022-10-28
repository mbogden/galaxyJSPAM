'''
    Author:     Matthew Ogden
    Created:    10 May 2019
Description:    This program is the main function that prepares galaxy models,
                runs them through the JSPAM software and stores particle files
'''

from os import system, remove, listdir, getcwd, chdir, path, rename
from sys import path as sysPath
from zipfile import (
    ZipFile,
    ZIP_DEFLATED,
    ZIP_STORED
)

# For loading in Matt's general purpose python libraries
supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
sysPath.append( supportPath )
import general_module as gm
import info_module as im


def test():
    print("SM: Hi!  You're in Matthew's main code for all things simulation.")

# For testing and developement from outside. 
new_func = None
def set_new_func( new_func_ptr ):
    global new_func
    new_func = new_func_ptr
# End set new function
    
# Expected SPAM files names and executable location

spam_exe_loc = gm.validPath( __file__[0: __file__.rfind('/')] + "/bin/basic_run" )
siName = "a_0.000"
sfName = "a_0.101"

spam_param_names = [ \
     'sec_vec_1', \
     'sec_vec_2', \
     'sec_vec_3', \
     'sec_vec_4', \
     'sec_vec_5', \
     'sec_vec_6', \
     'mass_1', \
     'mass_2', \
     'rout_1', \
     'rout_2', \
     'phi_1', \
     'phi_2', \
     'theta_1', \
     'theta_2', \
     'epsilon_1', \
     'epsilon_2', \
     'rscale_1_1', \
     'rscale_1_2', \
     'rscale_1_3', \
     'rscale_2_1', \
     'rscale_2_2', \
     'rscale_2_3', \
    ]

#print("SIM: SPAM LOC: ", spam_exe_loc)

# Will return None is not found
spam_exe = gm.validPath( spam_exe_loc )

nPart = 0
maxN = 1e5  # Arbitrary limit in case of typo, can be changed if needed

def main(arg):

    if arg.printBase:
        test()
        arg.printArg()
        gm.test()
        im.test()

    # end main print
    
    if arg.simple:
        if arg.printBase: 
            print("SM: Simple!~")
            print("\t- Nothing else to see here")

    elif arg.runDir != None:
        sim_run( arg.runDir, printAll=arg.printAll )

    else:
        print("SS: Nothing selected!")
        print("SS: Recommended options")
        print("\t - simple")
        print("\t - runDir /path/to/dir/")
        print("\t - targetDir /path/to/dir/")
        print("\t - dataDir /path/to/dir/")

# End main


def main_sm_run( rInfo, cmdArg = gm.inArgClass() ):
    
    
    printBase = rInfo.printBase
    printAll = rInfo.printAll

    if printBase:
        print( "SM.main_sm_run:" )
    
    # Check if run is valid
    if rInfo.status == False:
        print("SM: WARNING:")
        print("\t - Run directory not good.")
        return

    elif printBase:
        im.tabprint("Run ID: %s" % rInfo.get("run_id"))
        im.tabprint( "rInfo status: %s" % rInfo.status )

    # Get score parameter data for creating new files
    scoreParams = cmdArg.get('scoreParams',None)
    
    if scoreParams == None:
        if printBase: print("SM: WARNING: Please provide score parameters")
        return
    
    elif printAll:
        im.tabprint("Score parameters len: %d"%len(scoreParams))
    
    # Extract unique simulation scenerios
    simParams = {}
    for pKey in scoreParams:
        simKey = scoreParams[pKey]['simArg']['name']
        if simKey not in simParams:
            simParams[simKey] = scoreParams[pKey]['simArg']
    
    # Check if simulation already exists
    todoList = []
    overWrite =  cmdArg.get('overWrite', False)
    if overWrite and printAll:
        im.tabprint("Overwriting old simulations")
    
    # Loop through score parameters and grab unique simulation scenarios. 
    for simKey in simParams:
        
        # If overwriting command given, add.
        if overWrite:            
            todoList.append(simKey)
            continue
        
        # Find if points file exists, add if it doesn't
        iLoc, fLoc = rInfo.findPtsLoc( simParams[simKey]['name'] )        
        if iLoc == None or fLoc == None:
            todoList.append(simKey)
        else:
            im.tabprint("Particles Found: %s" % simParams[simKey]['name'])
    # End simKey in simParams
    
    if printBase:
        im.tabprint("Simulations to run: %d" % len(todoList))
    
    # if no simluations, leave early
    if len(todoList) == 0:
        return
        
    # Double check spam's executable is already built.
    if spam_exe == None:
        print("WARNING: SM.main_sm_run")
        im.tabprint("SPAM executable command not found")
        im.tabprint("location expected: %s" % spam_exe_loc )
        return
        
    for simKey in todoList:
        new_simulation( rInfo, simParams[simKey], cmdArg )
    
# end processing run dir

def new_simulation( rInfo, simArg, cmdArg ):
    
    printBase = rInfo.printBase
    printAll = rInfo.printAll
        
    if printBase:
        print("\nSM.new_simulation:")
        im.tabprint("New Simulation Name: %s" % simArg['name'])
    
    # Grab needed variables and directories.
    model_data = rInfo.get('model_data', None)
    ptsDir = rInfo.get('ptsDir', None)
    tmpDir = rInfo.get('tmpDir', None)
    nPts = str(simArg.get('nPts',None))
                
    if printAll:
        im.tabprint("n particles: %s" % nPts)
        im.tabprint("model_data: %s" % model_data)
        im.tabprint("ptsDir: %s" % ptsDir)
        im.tabprint("tmpDir: %s" % tmpDir)
    
    # Check for valid number of particles
    if nPts != None:
        # Check if using "k" abbreviation in num_particles
        if 'k' in nPts:
            kLoc = nPts.index('k')
            n = int( nPts[0:kLoc] ) * 1000
            nPts = n
        
        nPts = int( nPts )
        
        # Check if particle count is over max.
        if nPts > maxN:
            if printBase: 
                print("\nWARNING: SM.new_simulation:")
                im.tabprint("Number of points is greater than max")
                im.tabprint("n particles: %s" % nPts)
                im.tabprint("n max: %s" % maxN)
            return False
    # end if nPts
    
    # Check if needed info has been obtained
    if (model_data == None or ptsDir == None or tmpDir == None or nPts == None):
        
        if printBase: 
            print("\nWARNING: SM.new_simulation:")
            im.tabprint("A required argument was invalid")
            im.tabprint("n particles: %s" % nPts)
            im.tabprint("model_data: %s" % model_data)
            im.tabprint("ptsDir: %s" % ptsDir)
            im.tabprint("tmpDir: %s" % tmpDir)
        return False
    
    # Save current working directory and move to temp folder
    prevDir = getcwd()
    chdir( tmpDir )
    
    if printAll:
        im.tabprint(' Current Working Dir: %s' % getcwd() )
    
    # Call SPAM wrapper
    goodRun, retVal = spam_wrapper( nPts, model_data, printCmd = printAll )
    
    # Print results
    if not goodRun and printBase:
        print("WARNING: SM.new_simulation")
        im.tabprint("New simulation failed.  Error given")
        print(retVal)
        return
    
    if goodRun and printAll:
        print("SM.new_simulation: Good simulation, value returned")
        print(retVal)
    
    # Generate file names and locations
    
    # Unique names
    iName = '%s_pts.000' % simArg['name']
    fName = '%s_pts.101' % simArg['name']
    
    # Particle location created by SPAM.
    siLoc = tmpDir + siName
    sfLoc = tmpDir + sfName
    
    # Unique Loc
    iLoc = tmpDir + iName
    fLoc = tmpDir + fName
    
    # Rename temp particle files to prevent overwriting and later use
    rename( siLoc, iLoc )
    rename( sfLoc, fLoc )
    
    if printAll:
        im.tabprint("Particles Generated")
        im.tabprint("I: (%s) - %s" % ( path.isfile( iLoc ), iLoc ) )
        im.tabprint("F: (%s) - %s" % ( path.isfile( fLoc ), fLoc ) )
    
    # Remove other files and outputs
    remove(tmpDir + "fort.21")
    remove(tmpDir + "fort.24")
    remove(tmpDir + "fort.50")
    remove(tmpDir + "gmon.out")
    remove(tmpDir + "gscript")
    
    # Check if saving particles files is required.
    if cmdArg.get("zipSim",False):
        
        zipName = '%s.zip' % simArg['name']
        
        if printAll: 
            im.tabprint("Zipping Files to: %s" % (ptsDir + zipName ) )
            
        # Auto closed on with exit
        with ZipFile( ptsDir + zipName, 'w') as myzip:
            myzip.write(iLoc, iName, compress_type=ZIP_DEFLATED)
            myzip.write(fLoc, fName, compress_type=ZIP_DEFLATED)
    
    # Return to previous working directory to prevent potential errors elsewhere in code.
    chdir( prevDir )

# End Def new_simulation

def spam_wrapper( nPts, model_data, printCmd ):
    
    # Build and Call SPAM executable with arguments
    sysCmd = '%s -m %d -n1 %d -n2 %d %s' % (spam_exe, 0, nPts, nPts, model_data)
    
    if printCmd:
        print("\nSM.spam_wrapper: ")
        im.tabprint('Calling Shell cmd: %s' % sysCmd)
        
    try:
        retVal = system(sysCmd)        
        return True, retVal
            
    except Exception as e:
        return False, e
    
# End spam_wrapper


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )
