#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
    Author:     Matthew Ogden
    Created:    04 Nov 2020
Description:    For the newest version of images creation
'''

print("HI")
from os import path, listdir
from sys import path as sysPath
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# For loading in Matt's general purpose python libraries
sysPath.append( path.abspath( "Support_Code/" ) )
sysPath.append( path.abspath( "Image_Creator/" ) )
import general_module as gm
import info_module as im
gm.test()
im.test()


tInfo = im.target_info_class( targetDir = '../spam_data_pl3/587722984435351614/', printAll = False, newInfo=False)
#tInfo = im.target_info_class( targetDir = '../spam_data_pl3/587722984435351614/', printAll = False, newInfo=True)

runDirList = tInfo.iter_runs( )
rInfo = im.run_info_class( runDir = runDirList[0], printAll = False )
if rInfo.status == False:
    print("NOOOOOO")
print("yay")


# In[2]:


top500 = runDirList[0:500]
print(len(top500))
print(top500[0])


# In[3]:


import subprocess

def prepAndCallWNDCHARM( rDir, overwrite = False ):

    rInfo = im.run_info_class(runDir=rDir,printBase=False)    
    imgListLoc = rInfo.runDir + 'wndcharmImgList.txt'

    '''
    imgList = rInfo.getAllImgLocs(  )
    
    wndcharmList = [ imgLoc for imgLoc in imgList if 'wndcharm' in imgLoc ]
    
    with open(imgListLoc,'w') as oFile:
        for imgLoc in wndcharmList:            
            oFile.write(imgLoc+'\n')
    '''
    
    wndcharmPath = path.abspath('Machine_Compare/feature_extraction.py')    
              
    wndcharm_command = "python2 %s %s %s" % (wndcharmPath,imgListLoc,rInfo.runDir) # launch your python2 script using bash

    process = subprocess.Popen(wndcharm_command.split(), stdout=subprocess.PIPE, cwd=rInfo.tmpDir)
    output, error = process.communicate()  # receive output from the python2 script

    rInfo.delTmp()
#prepAndCallWNDCHARM( top500[0] )


# In[4]:


argList = [ dict( rDir=rDir ) for rDir in top500 ]
print(len(argList))
print(argList[0])


# In[ ]:


# Prepare parallel class
ppClass = gm.ppClass( 7, printProg=True )
sharedModelSet = ppClass.manager.dict()

ppClass.loadQueue( prepAndCallWNDCHARM, argList )

# Do parallel
print("start")
ppClass.runCores()
print('boop')


# In[ ]:




