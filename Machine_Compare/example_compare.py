# Example program for using machine score modules

import main_compare as mc
import machine_score_methods as ms

mc.test() 
ms.test()

# For images
from os import path
from sys import exit

img1Loc = path.abspath( '/nfshome/mbo2d/Public/miscImgs/0000_model.png' )
img2Loc = path.abspath( '/nfshome/mbo2d/Public/miscImgs/target.png' )

print( path.exists( img1Loc ), img1Loc )
print( path.exists( img2Loc ), img2Loc )

print("")
# Read in images seperately

img1 = mc.getImg( img1Loc )
print( type( img1 ), img1.shape )

img2 = mc.getImg( img2Loc )
print( type( img2 ), img2.shape )

print("")
# Or read images together

img1, img2 = mc.getImages( img1Loc, img2Loc )
print( type( img1 ), img1.shape )
print( type( img2 ), img2.shape )




print("")
# You can give the two images to main_compare and they'll do everything
scoreList, infoList = mc.allScores( img1, img2 )

for i in range( len( scoreList ) ):
    print("Score: %.5f - Name: %s" % ( scoreList[i], infoList[i] ) )







# Or you can ignore main_compare and reach straight into machine_score_methods
# Get list of available score names and function pointers from machine score methods
print("")
scoreFunctions = ms.getScoreFunctions()

for name, ptr in scoreFunctions:

    print( name, ptr( img1, img2 ) )

# If you want to grab a specific scoring function
print("")
myFunc = ms.getScoreFunc( 'typo', printAll = True ) 
myFunc = ms.getScoreFunc( 'score_overlap_fraction', printAll = True ) 

print("")
# How to use score function ptrs

# simple use.  give two images and it returns a float
score = myFunc( img1, img2 ) 
print( score )

print("")
# Or get more info if variables are involved
score, info = myFunc( img1, img2, simple=False )
print( score, info )

print("")
# Note overlap_fraction has a variable you can change. You can modify 'h'
print( myFunc( img1, img2, simple=False, h = 10 )[1] )
print( myFunc( img1, img2, simple=False, h = 100 )[1] )
print( myFunc( img1, img2, simple=False, h = 200 )[1] )
