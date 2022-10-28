'''
    Author:     Matthew Ogden
    Created:    20 Jan 2020
    Altered:    07 Feb 2020
Description:    This is my python template for how I've been making my python programs
'''

from sys import ( exit, argv, path as sysPath )
from os import ( path, listdir, system )

import numpy as np

printAll = True

tDir = 'Public/target_images/'
dataDir = 'localstorage/spam_data_pl3/'

def main(argList):

    endEarly = readArg(argList)

    if endEarly:
        exit(-1)

    #gatherTargetImgs()
    #createParamFile()
    #renameImgs()
    #gatherTargetInfo()
    #newTargetImg()
    #getPartFiles()
    #createModelImages()
    #listGoodPairs()
    #moveParamFiles()
    #createProgFiles()
    #updateInfoFiles()
    #renameParameters()
    reZipFiles()


# End main


def reZipFiles():

    from zipfile import ZipFile

    sDirs = listdir( dataDir )
    sDirs.sort()

    for i,t in enumerate(sDirs):
        print("Sdss %4d / %4d" % ( i, len(sDirs)))
        sName = t
        sDir = dataDir + sName + '/'
        gDir = sDir + 'gen000/'
        pDir = sDir + 'information/'

        rDirs = listdir( gDir )
        rDirs.sort()

        for j,rDir in enumerate( rDirs ):

            rDir = gDir + rDir + '/'

            partDir = rDir + 'particle_files/'

            zLoc = partDir + '100000_pts.zip'

            if not path.exists( zLoc ):
                continue

            else:
                with ZipFile(zLoc, 'r') as zObj:
                    zFiles = zObj.namelist()

                    if len( zFiles[0].split('/') ) > 1:
                        #print(zFiles[0])
                        unZipCmd = 'unzip -qq -j -o %s -d %s' % ( zLoc, partDir )
                        print(unZipCmd)
                        try:
                            system( unZipCmd )
                        except:
                            print("huh")
                        rmCmd = 'rm %s' % zLoc
                        system( rmCmd )

                        ptsFiles = listdir( partDir )

                        zipCmd = 'zip -r -j %s %s' % ( zLoc, partDir+'*' )
                        system( zipCmd )

                        for ptsLoc in ptsFiles:
                            system( 'rm %s' % partDir + ptsLoc )
            print("Run %4d / %4d" % ( j, len(rDirs)) , end='\r')

        print("Sdss Good: %s" % sName)




def renameParameters():

    sDirs = listdir( dataDir )
    sDirs.sort()

    for i,t in enumerate(sDirs):
        sName = t
        sDir = dataDir + sName + '/'
        pDir = sDir + 'sdssParameters/'
        pLoc = pDir + 'parameters.txt'

        toDir = sDir + 'information/'
        mvCmd = 'mv %s %s' % ( pDir, toDir ) 
        print(mvCmd)

        system( mvCmd )

        updateProgFile( toDir, 'Completed... Renamed parameters to information folder.\n' )



def updateInfoFiles():

    sDirs = listdir( dataDir )
    sDirs.sort()

    for i,t in enumerate(sDirs):
        sName = t
        sDir = dataDir + sName + '/'
        pDir = sDir + 'information/'
        pLoc = pDir + 'parameters.txt'

        updateProgFile( pDir, 'Completed... New parameters format. v2\n' )

        pList = readFile( pLoc )

        tInfo = ''
        pl = ''
        p2 = ''

        for line in pList:
            
            if 'target_zoo.png' in line:
                tInfo = line

            if 'primary_lum' in line:
                p1 = line

            if 'secondary_lum' in line:
                p2 = line

        if tInfo == '' or p1 == '' or p2 == '':
            print("NOOOOOOO")
            break
        
        newFile = []
        newFile.append('# Target Image Data\n')
        newFile.append( tInfo )
        newFile.append( p1 )
        newFile.append( p2 )
        newFile.append( '\n' )
        newFile.append( '# Model Image Parameters\n' )
        newFile.append( 'default zoo_default_param.txt\n' )

        pFile = open(pLoc, 'w' )
        for l in newFile:
            pFile.write(l)
        pFile.close()

        #updateProgFile( tpDir, 'Working... Creating/updating sdss info file.\n' )

        


def updateProgFile( pDir, line ):

    pLoc = pDir + 'progress.txt'
    pList = readFile( pLoc )

    notFound = True

    for l in pList:
        if line == l:
            print("Found")
            return 

    pFile = open(pLoc, 'a')
    pFile.write(line)
    pFile.close()
    



def createProgFiles():

    sDirs = listdir( dataDir )
    sDirs.sort()

    for i,t in enumerate(sDirs):
        sName = t
        sDir = dataDir + sName + '/'
        pDir = sDir + 'sdssParameters/'
        pLoc = pDir + 'progress.txt'

        pFile = open(pLoc, 'w' )
        pFile.write("# Progress File for %s\n" % sName)
        pFile.write("Completed... 100k particle files\n")
        pFile.close()

        

def moveParamFiles():

    tFiles = listdir( tDir )
    pFiles = [ t for t in tFiles if 'param.txt' in t ]
    pFiles.sort()

    print(len(pFiles))

    for i,t in enumerate(pFiles):
        sName = t.split('_param.')[0]
        fpLoc = tDir + t
        tsDir = dataDir + sName + '/'
        tpDir = tsDir + 'sdssParameters/'
        tpLoc = tpDir + 'zoo_default_param.txt'
        if not ( path.isfile( fpLoc ) and path.exists( tpDir ) ):
            Print("NO")
            break
        
        cpCmd = 'cp %s %s' % ( fpLoc, tpLoc )
        #print( cpCmd )
        #system( cpCmd )
        
        if not path.isfile( tpLoc ):
            print("ERROR")
            break
        
        updateProgFile( tpDir, 'Completed... Galaxy Zoo Target Photo with info\n' )
        updateProgFile( tpDir, 'Completed... Default Image Parameter\n' )

        

        





def listGoodPairs():

    tFiles = listdir( tDir )
    mImgs = [ t for t in tFiles if 'model.png' in t ]
    mImgs.sort()

    print(len(mImgs))

    goodList = open("goodList.txt",'w')

    for i,t in enumerate(mImgs):
        sName = t.split('_model.')[0]
        goodList.write("%s\n" % sName )





def createModelImages():
    print("Creating model images")

    sysPath.append( 'galaxyJSPAM/Image_Creator/python/' )
    import image_creator_v4 as ic

    tfiles = listdir( tdir )
    timgs = [ t for t in tfiles if 'zoo.png' in t ]
    timgs.sort()


    for i,t in enumerate(tImgs):
        sName = t.split('_zoo.')[0]
        
        p1Loc = tDir + '%s_pts.000' % sName
        p2Loc = tDir + '%s_pts.101' % sName
        pLoc = tDir + '%s_param.txt' % sName
        toLoc = tDir + '%s_model.png' % sName

        if not path.exists( p1Loc ) or not path.exists( p2Loc ) or not path.exists( pLoc ):
            print("%s - files not found" % sName)
            continue
        
        if path.exists( toLoc ):
            #continue
            pass

        ic.Image_From_Files( pLoc, p1Loc, p2Loc, imgLoc=toLoc, circles=True )
        print('%d / %d' % (i,len(tImgs)) ,end='/')

        #break



def getPartFiles():
    print("Getting particle files")

    sDirs = listdir( dataDir )
    sDirs.sort()

    for sName in sDirs:
        print(sName)
        sDir = dataDir + sName + '/'
        rDir = sDir + 'gen000/run_00000/'
        pDir = rDir + 'particle_files/'
        zLoc = pDir + '100000_pts.zip'
        print(zLoc)
        print(path.exists(zLoc))

        unZipCmd = 'unzip -j -o %s -d %s' % ( zLoc, pDir )
        print( unZipCmd)
        system( unZipCmd )

        p1Loc = pDir + '100000_pts.000'
        p2Loc = pDir + '100000_pts.101'

        print( path.exists( p1Loc), path.exists( p2Loc ))

        p1To = tDir + '%s_pts.000' % sName
        p2To = tDir + '%s_pts.101' % sName

        mv1Cmd = 'mv %s %s' % ( p1Loc, p1To ) 
        mv2Cmd = 'mv %s %s' % ( p2Loc, p2To ) 

        print( mv1Cmd )
        system( mv1Cmd )
        system( mv2Cmd )

        #break



def newTargetImg():

    # append misc function path
    sysPath.append( 'galaxyJSPAM/Image_Creator/python/' )
    import misc_func as mf
    mf.test()

    import cv2

    tFiles = listdir( tDir )
    tImgs = [ t for t in tFiles if 'zoo.png' in t ]
    tImgs.sort()
    print( len( tImgs )) 

    for otName in tImgs:
        tLoc = tDir + otName

        tName = otName.split('_zoo.png')[0]
        
        if not path.isfile( tLoc ):
            print("warning")

        if '1614' not in tLoc:
            #continue
            pass
        
        print(tLoc)

        g1c = np.zeros(2)
        g2c = np.zeros(2)

        infoLoc = tDir + '%s_info.txt' % tName

        if path.isfile( infoLoc ):

            infoFile = readFile( infoLoc )

            for l in infoFile:

                if 'target_zoo' in l:
                    l = l.strip().split()
                    g1c[0] = int(l[1])
                    g1c[1] = int(l[2])
                    g2c[0] = int(l[3])
                    g2c[1] = int(l[4])
            
            print( g1c, g2c )
        else:
            print('WARNING')

        tImg = cv2.imread( tLoc, 0 )

        iImg = mf.addCircle( tImg, ( int(g1c[0]), int(g1c[1]) ) )
        iImg = mf.addCircle( tImg, ( int(g2c[0]), int(g2c[1]) ) )

        ntLoc = tDir + tName + '_new.png'
        cv2.imwrite( ntLoc, tImg )

    

def gatherTargetInfo():
    print("Gathering info")
    
    tContents = listdir( dataDir )
    print( len(tContents) )

    for sName in tContents:
        sdDir = dataDir + sName + '/'       # sdss data directory
        pDir = sdDir + 'sdssParameters/'
        pLoc = pDir + 'parameters.txt'
        #pFile = readFile(pLoc)
        #print( pFile )
        cpCmd = 'cp %s %s%s_info.txt' % ( pLoc, tDir, sName )
        system( cpCmd )
        print( cpCmd )
        #break



def renameImgs():

    tFiles = listdir( tDir )
    tImgs = [ t for t in tFiles if '.png' in t ]
    tImgs.sort()

    for t in tImgs:
        old = tDir + t

        if not path.isfile( old ):
            print("WARNING!")

        try:
            tName = t.split('.')[0].split('sdss')[1]
        except:
            tName = t.split('.')[0]

        new = tDir + tName + '_zoo.png'

        mvCmd = 'mv %s %s' % ( old, new )

        print( mvCmd )
        system( mvCmd )




def createParamFile():

    tFiles = listdir( tDir )
    tImgs = [ t for t in tFiles if 'zoo.png' in t ]
    tImgs.sort()

    if len(tImgs) != 63:
        print("WAIT!", len(tImgs) )
        exit(-1)

    pFile = readFile( 'galaxyJSPAM/Input_Data/image_parameters/param_v3_default.txt' )

    for t in tImgs:
        nFile = pFile.copy()
        
        try:
            tName = t.split('.')[0].split('_zoo')[0].split('sdss')[1]
        except:
            tName = t.split('.')[0].split('_zoo')[0]

        infoLoc = tDir + '%s_info.txt' % tName

        g1b = 1
        g2b = 1

        g1c = np.zeros(2)
        g2c = np.zeros(2)

        if path.isfile( infoLoc ):

            infoFile = readFile( infoLoc )

            for l in infoFile:

                if 'primary_luminosity' in l:
                    l = l.strip().split()
                    g1b = float( l[1] )
            
                if 'secondary_luminosity' in l:
                    l = l.strip().split()
                    g2b = float( l[1] )

                if 'target_zoo' in l:
                    l = l.strip().split()
                    g1c[0] = l[1]
                    g1c[1] = l[2]
                    g2c[0] = l[3]
                    g2c[1] = l[4]
            
        else:
            print('WARNING')

        # get resolution of target_zoo image
        import cv2
        tLoc = tDir + t

        timg = cv2.imread( tLoc )

        # modify default param file
        for i, l in enumerate(nFile):
            
            l = l.strip().split()

            if len(l) == 0:
                continue

            if l[0] == 'parameter_name':
                nFile[i] = 'parameter_name default\n'

            if l[0] == '#':
                nFile[i] = '# Starting image parameter file for %s\n' % tName

            if l[0] == 'image_rows':
                nFile[i] = 'image_rows %d\n' % ( timg.shape[0] ) 

            if l[0] == 'image_cols':
                nFile[i] = 'image_cols %d\n' % ( timg.shape[1] ) 

            if l[0] == 'sdss_name':
                nFile[i] = 'sdss_name %s\n' % tName

            if l[0] == 'galaxy_1_center':
                nFile[i] = 'galaxy_1_center %d %d\n' % ( int( g1c[0] ) , int(g1c[1]) )

            if l[0] == 'galaxy_2_center':
                nFile[i] = 'galaxy_2_center %d %d\n' % ( int( g2c[0] ) , int(g2c[1]) )

        nFile.append('galaxy_1_luminosity %f\n' % g1b)
        nFile.append('galaxy_2_luminosity %f\n' % g2b)
        
        pLoc = tDir + '%s_param.txt' % tName
        oFile = open( pLoc, 'w' )
        
        for l in nFile:
            oFile.write( l )
        oFile.close()



def gatherTargetImgs():
    allDir = 'galaxyJSPAM/Input_Data/targets/'
    targetFolders = listdir( allDir )

    fDir = 'Public/target_images/'

    for t in targetFolders:

        tDir = allDir + t + '/'
        tFiles = listdir( tDir )

        otImg = [ f for f in tFiles if '.png' in f ]

        if len( otImg ) != 1:
            print("Warning!", tDir )

        mvCmd = 'mv %s %s' % ( tDir + otImg[0], fDir + '.' )
        system( mvCmd )
# end gather target images


def readArg(argList):

    global printAll

    endEarly = False

    for i,arg in enumerate(argList):

        if arg[0] != '-':
            continue

        elif arg == '-argFile':
            argFileLoc = argList[i+1]
            argList = readArgFile( argList, argFileLoc ) 

        elif arg == '-noprint':
            printAll = False

    # Check if input arguments are valid

    return endEarly

# End reading command line arguments


def readArgFile(argList, argFileLoc):

    try:
        argFile = open( argFileLoc, 'r')
    except:
        print("Failed to open/read argument file '%s'" % argFileLoc)
    else:

        for l in argFile:
            l = l.strip()

            # Skip line if empty
            if len(l) == 0:
                continue

            # Skip line if comment
            if l[0] == '#':
                continue

            lineItems = l.split()
            for item in lineItems:
                argList.append(item)

        # End going through file

    return argList
# end read argument file


def readFile( fileLoc ):

    if not path.isfile( fileLoc ):
        print("File does not exist: %s" % fileLoc)
        return []
    
    try:
        inFile = open( fileLoc, 'r' )

    except:
        print('Failed to open/read file at \'%s\'' % fileLoc)
        return []

    else:
        inList = list(inFile)
        inFile.close()
        return inList

# End simple read file

# Run main after declaring functions
if __name__ == '__main__':
    argList = argv
    main(argList)
