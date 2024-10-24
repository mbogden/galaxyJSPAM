# This is a quick python script to create Model lists that are easier to read for my own purposes

print('In accumulate model script')

import os
import re

just1Line = True


genNum = 0 # Because we're reading from zoo model files


rawModelLoc = 'from_site/'
new_loc = 'qual_example.txt'

sdssModelList = os.listdir(rawModelLoc)

bigFile = open(new_loc,'w')

total = 0
    
for docLoc in sdssModelList:

    sdssName = docLoc.split('.')[0]
    n = 0

    docFullLoc = rawModelLoc + docLoc
    inFile = open(docFullLoc, 'r')

    for i,line in enumerate(inFile):


        l = line.strip()
        t1 = l.split('\t')
        t2 = t1[0].split(',')

        # Break if reached point in file with no Competitive score
        if len(t2) != 4:
            break

        modelName = t2[0]
        score = t2[1]
        runData = t1[1]
        
        oLine = '%s %d %d %s %s %s\n' % (sdssName, genNum, n, score, modelName, runData)
        bigFile.write(oLine)

        n += 1
    # End for i,line in enumerate(inFile):

    print( '%s \t- %5.5d' %(sdssName, n))

    total += n

    inFile.close()

#end for docLoc in sdssModelList:

bigFile.close()

print(' sdss: %d - models: %d ' % ( len( sdssModelList), total ) )





