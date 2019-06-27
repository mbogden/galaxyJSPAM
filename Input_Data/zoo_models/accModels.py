# This is a quick python script to create Model lists that are easier to read for my own purposes

print('In accumulate model script')

import os
import re

just1Line = True


genNum = 0 # Because we're reading from zoo model files


rawModelLoc = 'from_site/'
new_loc = 'for_cluster/'

sdssModelList = os.listdir(rawModelLoc)

bigFile = open('sdss2List.txt','w')
    
for docLoc in sdssModelList:

    sdssName = docLoc.split('.')[0]

    n = 1
    docFullLoc = rawModelLoc + docLoc
    inFile = open(docFullLoc, 'r')

    for i,line in enumerate(inFile):
        l = line.strip()
        t1 = l.split('\t')
        t2 = t1[0].split(',')

        # Break if reached point in file with no Competitive score
        if len(t2) != 4:
            break

        score = t2[1]
        runData = t1[1]
        
        oLine = '%s %d %d %s %s\n' % (sdssName, genNum, n, score, runData)
        bigFile.write(oLine)

        if just1Line:
            break

        n += 1
    # End for i,line in enumerate(inFile):

    print( '%d  -  %s' %(n,sdssName))

    inFile.close()

#end for docLoc in sdssModelList:

bigFile.close()

print(len(sdssModelList))





