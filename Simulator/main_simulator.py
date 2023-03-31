'''
    Author:     Matthew Ogden
    Created:    10 May 2019
Description:    This program is the main function that prepares galaxy models,
                runs them through the JSPAM software and stores particle files
'''

import os
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
    print("SIM: Hi!  You're in Matthew's main code for all things simulation.")

# For testing and developement from outside. 
new_func = None
def set_new_func( new_func_ptr ):
    global new_func
    new_func = new_func_ptr
# End set new function
    
# Expected SPAM files names and executable location

spam_exe_loc = gm.validPath( __file__[0: __file__.rfind('/')] + "/bin/basic_run" )
spam_orb_loc = gm.validPath( __file__[0: __file__.rfind('/')] + "/bin/orb_run" )
spam_many_endings_loc = gm.validPath( __file__[0: __file__.rfind('/')] + "/bin/many_endings_run" )

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
spam_orb = gm.validPath( spam_orb_loc )
spam_many_endings_exe = gm.validPath( spam_many_endings_loc )

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
            print("SIM: Simple!~")
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


def main_sim_run( rInfo, cmdArg = gm.inArgClass() ):
    
    printBase = rInfo.printBase
    printAll = rInfo.printAll

    if printBase:
        print( "SIM: main_sm_run:" )
    
    # Check if run is valid
    if rInfo.status == False:
        gm.eprint("WARNING: SIM.main_sm_run:")
        gm.etabprint("Run not good: %s" % rInfo.get('runDir'))
        return

    if printBase:
        gm.tabprint("Run ID: %s" % rInfo.get("run_id"))
    
    # Check if orbit and min file are there, if not run.
    orbLoc = gm.validPath( rInfo.get('orbLoc',None) )
    minLoc = gm.validPath( rInfo.get('minLoc',None) )
    
    if orbLoc == None or minLoc == None:
        orbit_simulation( rInfo, cmdArg )

    # Get score parameter data for creating new files
    scoreParams = cmdArg.get('scoreParams',None)
    
    if scoreParams == None:
        gm.eprint("WARNING: SIM.main_rm_run: ")
        gm.etabprint("Please provide score parameters")
        return
    
    if printAll:
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
        zLoc = rInfo.findZippedPtsLoc( simParams[simKey]['name'] )        
        if zLoc == None:
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
        gm.eprint("WARNING: SIM: main_sm_run")
        gm.etabprint("SPAM executable command not found")
        gm.etabprint("location expected: %s" % spam_exe_loc )
        return
        
    for simKey in todoList:
        simType = simParams[simKey].get( 'type', 'basic_run' )

        if printBase:
            gm.tabprint("Sim Key/Type: %s/%s" % (simKey,simType) )

        if simType == 'basic_run':
            basic_run( rInfo, simParams[simKey], cmdArg )

        if simType == 'many_endings':
            many_endings( rInfo, simParams[simKey], cmdArg )
    
# end processing run dir
    

def many_endings( rInfo, simArg, cmdArg ):
    
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
        
    if printBase:
        print("\nSIM: many_endings:")
        im.tabprint("New Simulation Name: %s" % simArg['name'])
    
    # Grab needed variables and directories.
    model_data = rInfo.get('model_data', None)
    ptsDir = rInfo.get('ptsDir', None)
    tmpDir = rInfo.get('tmpDir', None)
    nPts = str( simArg.get('nPts',None) )
                
    if printAll:
        gm.tabprint("n particles: %s" % nPts)
        gm.tabprint("model_data: %s" % model_data)
        gm.tabprint("ptsDir: %s" % ptsDir)
        gm.tabprint("tmpDir: %s" % tmpDir)
    
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
            gm.eprint("\nWARNING: SIM: many_endings:")
            gm.etabprint("Number of points is greater than max")
            gm.etabprint("n particles: %s" % nPts)
            gm.etabprint("n max: %s" % maxN)
            return False
    # end if nPts
    
    # Check if needed info has been obtained
    if (model_data == None or ptsDir == None or tmpDir == None or nPts == None):
        
        gm.eprint("\nWARNING: SIM: many_endings:")
        gm.etabprint("A required argument was invalid")
        gm.etabprint("n particles: %s" % nPts)
        gm.etabprint("model_data: %s" % model_data)
        gm.etabprint("ptsDir: %s" % ptsDir)
        gm.etabprint("tmpDir: %s" % tmpDir)
        return False
    
    # Save current working directory and move to temp folder
    prevDir = getcwd()
    chdir( tmpDir )
    
    if printAll:
        im.tabprint(' Current Working Dir: %s' % getcwd() )
    
    # Call SPAM wrapper
    goodRun, retVal = many_endings_wrapper( nPts, model_data, printCmd = printAll )
    
    # Print results
    if not goodRun:
        gm.eprint("WARNING: SIM: many_endings")
        gm.etabprint("New simulation failed.  Error given: ")
        gm.eprint(retVal)
        return
    
    if goodRun and printAll:
        print("SIM: many_endings: Good simulation, value returned")
        print(retVal)
    
    # Generate file names and locations
    
    # Unique names
    fName = '%s_pts.101' % simArg['name']
    
    # Particle location created by SPAM.
    sfLoc = tmpDir + sfName
    
    # Unique Loc
    fLoc = tmpDir + fName
    
    # Rename SPAM temp particle files to prevent overwriting
    rename( sfLoc, fLoc )
    
    if printAll:
        im.tabprint("Particles Generated")
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
            myzip.write(fLoc, fName, compress_type=ZIP_DEFLATED)
    
    # Return to previous working directory to prevent potential errors elsewhere in code.
    chdir( prevDir )

# End Def many_endings

def many_endings_wrapper( nPts, model_data, printCmd ):
    sysCmd = '%s -m %d -n1 %d -n2 %d %s' % ( spam_many_endings_exe, 0, nPts, nPts, model_data) 

    if printCmd:
        print(sysCmd)
    
    # return False, "Not Implemented"
        
    try:
        retVal = system(sysCmd)        
        return True, retVal
            
    except Exception as e:
        return False, e

# End Def many_endings_wrapper


def orbit_simulation( rInfo, cmdArg ):
        
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
        
    if printBase:
        print("SIM: orbit_simulation:")
    
    # Check if orbit exectued is created
    if spam_orb == None:
        gm.eprint("WARNING: SIM.orbit_simulation:")
        gm.etabprint("SPAM Orbit Executable not found")
        gm.etabprint("Considering running 'Make'")
        gm.etabprint("Expected Location: %s\n" % spam_orb_loc )
        return 
    
    # Change to tmpDir, save previous working dir
    pDir = os.getcwd()
    os.chdir( rInfo.get('tmpDir') )
        
    model_data = rInfo.get('model_data') 
    
    # Grab needed variables and directories.
    model_data = rInfo.get('model_data', None)
    runDir = rInfo.get('runDir',None)
    ptsDir = rInfo.get('ptsDir', None)
    tmpDir = rInfo.get('tmpDir', None)
                
    if printAll:
        gm.tabprint("model_data: %s" % model_data)
        gm.tabprint("runDir: %s" % runDir)
        gm.tabprint("ptsDir: %s" % ptsDir)
        gm.tabprint("tmpDir: %s" % tmpDir)
        
    # Call fortran orbital wrapper
    retVal = orbit_wrapper( model_data, printAll )
    
    # From location
    orbName = 'orbit.txt'
    minName = 'rmin.txt'
    
    orbLocFrom = gm.validPath( tmpDir + orbName )
    minLocFrom = gm.validPath( tmpDir + minName )
    
    # To locaiton
    orbLocTo = rInfo.get("orbLoc", None)
    minLocTo = rInfo.get('minLoc', None)
    
    # Print Warning if file note found
    if orbLocFrom == None or minLocFrom == None:
        gm.eprint( 'WARNING: SIM.orbit_simulation: %s' % rInfo.get("run_id") )
        gm.etabprint("Orbit file or Min file not found")
        gm.etabprint("(%s) - %s" % (os.path.exists( tmpDir + orbName ), tmpDir + orbName ) )
        gm.etabprint("(%s) - %s" % (os.path.exists( tmpDir + minName ), tmpDir + minName ) )
    
    # If files found, move
    if orbLocFrom != None: os.rename( orbLocFrom, orbLocTo )    
    if minLocFrom != None: os.rename( minLocFrom, minLocTo )
    
    # Change back to previous directory
    os.chdir( pDir )
    
# End orbit_simulation

def basic_run( rInfo, simArg, cmdArg ):
    
    printBase = cmdArg.printBase
    printAll = cmdArg.printAll
        
    if printBase:
        print("\nSIM: basic_run:")
        im.tabprint("New Simulation Name: %s" % simArg['name'])
    
    # Grab needed variables and directories.
    model_data = rInfo.get('model_data', None)
    ptsDir = rInfo.get('ptsDir', None)
    tmpDir = rInfo.get('tmpDir', None)
    nPts = str(simArg.get('nPts',None))
                
    if printAll:
        gm.tabprint("n particles: %s" % nPts)
        gm.tabprint("model_data: %s" % model_data)
        gm.tabprint("ptsDir: %s" % ptsDir)
        gm.tabprint("tmpDir: %s" % tmpDir)
    
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
            gm.eprint("\nWARNING: SIM: basic_run:")
            gm.etabprint("Number of points is greater than max")
            gm.etabprint("n particles: %s" % nPts)
            gm.etabprint("n max: %s" % maxN)
            return False
    # end if nPts
    
    # Check if needed info has been obtained
    if (model_data == None or ptsDir == None or tmpDir == None or nPts == None):
        
        gm.eprint("\nWARNING: SIM: basic_run:")
        gm.etabprint("A required argument was invalid")
        gm.etabprint("n particles: %s" % nPts)
        gm.etabprint("model_data: %s" % model_data)
        gm.etabprint("ptsDir: %s" % ptsDir)
        gm.etabprint("tmpDir: %s" % tmpDir)
        return False
    
    # Save current working directory and move to temp folder
    prevDir = getcwd()
    chdir( tmpDir )
    
    if printAll:
        im.tabprint(' Current Working Dir: %s' % getcwd() )
    
    # Call SPAM wrapper
    goodRun, retVal = spam_wrapper( nPts, model_data, printCmd = printAll )
    
    # Print results
    if not goodRun:
        gm.eprint("WARNING: SIM: basic_run")
        gm.etabprint("New simulation failed.  Error given")
        gm.eprint(retVal)
        return
    
    if goodRun and printAll:
        print("SIM: basic_run: Good simulation, value returned")
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

# End Def basic_run


def orbit_wrapper( model_data, printCmd ):
    
    sysCmd = '%s %s' % (spam_orb, model_data)
    
    if printCmd:
        print( 'SIM: orbit_wrapper:')
        gm.tabprint( 'cwd: %s' % os.getcwd() )
        gm.tabprint( 'cmd: %s' % sysCmd )
        gm.tabprint( 'Executing Orbit...' )
        
    retVal = os.system(sysCmd) 
    
    if printCmd:
        print('SIM: orbit_wrapper: return: %s'%str(retVal))
        print('')
        
    return retVal

# End orbit_wrapper

def spam_wrapper( nPts, model_data, printCmd ):
    
    # Build and Call SPAM executable with arguments
    sysCmd = '%s -m %d -n1 %d -n2 %d %s' % (spam_exe, 0, nPts, nPts, model_data)
    
    if printCmd:
        print("\nSIM: spam_wrapper: ")
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
