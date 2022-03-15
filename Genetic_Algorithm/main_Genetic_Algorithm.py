'''
    Author:     Matthew Ogden and Graham West
    Created:    28 Mar 2022
Description:    This code facilitates the Genetic Algorithm for evolving Galactic Models.
                Large portions of are derived from Graham West's Code found at the link.
                https://github.com/gtw2i/GA-Galaxy
'''

# For loading in Matt's general purpose python libraries
from os import path
from sys import path as sysPath
supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
sysPath.append( supportPath )
import general_module as gm
import info_module as im

# For loading other dependencies
from copy import deepcopy
import numpy as np
import numpy.linalg as LA
import math
import random
import pandas as pd


def test():
    print("GA: Hi!  You're in Matthew's main code for all things genetric algorithm.")

# For testing and developement from outside. 
new_func = None
def set_new_func( new_func_ptr ):
    global new_func
    new_func = new_func_ptr
# End set new function

# NOTE: The following converts the SPAM parameters into one of 8 true symmetries Graham identies. 
def convert_spam_to_ga( in_param ):
    
    if len( in_param.shape ) == 1:
        in_param = np.expand_dims(in_param, axis=0)
    
    out_param = deepcopy(in_param)
    psi = []
    
    for i in range( out_param.shape[0]):

        psi_p = 1
        psi_s = 1

        if( out_param[i,2] < 0 ):
            out_param[i,2]  =  -1 * out_param[i,2]
            out_param[i,5]  =  -1 * out_param[i,5]
            out_param[i,10] = 180 + out_param[i,10]
            out_param[i,11] = 180 + out_param[i,11]
        # end
        out_param[i,10] %= 360
        out_param[i,11] %= 360
        if( out_param[i,10] > 180 ):
            out_param[i,10] = out_param[i,10] - 180
            out_param[i,12] = -1 * out_param[i,12]
        # end
        if( out_param[i,11] > 180 ):
            out_param[i,11] = out_param[i,11] - 180
            out_param[i,13] = -1 * out_param[i,13]
        # end
        out_param[i,12] %= 360
        out_param[i,13] %= 360
        if( out_param[i,12] > 180 ):
            out_param[i,12] = out_param[i,12] - 360
        # end
        if( out_param[i,13] > 180 ):
            out_param[i,13] = out_param[i,13] - 360
        # end

        if( out_param[i,12] > 90 ):
            out_param[i,12] = out_param[i,12] - 180
            psi_p = -1
        elif( out_param[i,12] < -90 ):
            out_param[i,12] = out_param[i,12] + 180
            psi_p = -1
        # end
        if( out_param[i,13] > 90 ):
            out_param[i,13] = out_param[i,13] - 180
            psi_s = -1
        elif( out_param[i,13] < -90 ):
            out_param[i,13] = out_param[i,13] + 180
            psi_s = -1
            
        # end
        psi.append( [psi_p,psi_s] )
    # end for loop
    
    psi = np.array(psi)

    # energy
    G = 1
    r = ( out_param[:,0]**2 + out_param[:,1]**2 + out_param[:,2]**2 )**0.5
    U = -G*out_param[:,6]*out_param[:,7]/r
    v = ( out_param[:,3]**2 + out_param[:,4]**2 + out_param[:,5]**2 )**0.5
    K = 0.5*out_param[:,7]*v**2
    c = (K+U)/(K-U)

    # convert p,s mass to ratio,total mass
    t = out_param[:,6] + out_param[:,7]
    f = out_param[:,6] / t
    out_param[:,6] = f
    out_param[:,7] = t

    # spherical velocity
    phi   = ( np.arctan2( out_param[:,4], out_param[:,3] ) * 180.0 / np.pi ) % 360
    theta = ( np.arcsin( out_param[:,5] / v ) * 180.0 / np.pi )

    out_param[:,3] = c

    out_param[:,4] = phi
    out_param[:,5] = theta

    Ap = np.abs(out_param[:,8]**2*np.cos(out_param[:,12]*np.pi/180.0))
    As = np.abs(out_param[:,9]**2*np.cos(out_param[:,13]*np.pi/180.0))

    out_param[:,8] = Ap
    out_param[:,9] = As

    return out_param, psi

# End convert_spam_to_ga


def convert_ga_to_spam( in_param, psi ):
    
    # Expand in parameters if a single vector
    if len( in_param.shape ) == 1:
        in_param = np.expand_dims(in_param, axis=0)
    
    # Create seperate matrix for output parameters
    out_param = deepcopy(in_param)
    
    for i in range( out_param.shape[0]):
        
        # add for psi
        if( psi[i,0] == -1 ):
            out_param[i,12] += 180.0
        # end
        if( psi[i,1] == -1 ):
            out_param[i,13] += 180.0
        # end

        # convert mass units
        f    = out_param[i,6]
        t    = out_param[i,7]
        out_param[i,6] = f*out_param[i,7]
        out_param[i,7] = (1.0-f)*out_param[i,7]

        G = 1
        c = out_param[i,3]
        v = ( (1+c)/(1-c)*2*G*out_param[i,6]/(out_param[i,0]**2+out_param[i,1]**2+out_param[i,2]**2)**0.5 )**0.5    

        vx = v*math.cos(out_param[i,4]*np.pi/180.0)*math.cos(out_param[i,5]*np.pi/180.0)
        vy = v*math.sin(out_param[i,4]*np.pi/180.0)*math.cos(out_param[i,5]*np.pi/180.0)
        vz = v*math.sin(out_param[i,5]*np.pi/180.0)
        out_param[i,3] = vx
        out_param[i,4] = vy
        out_param[i,5] = vz
        
        Ap = out_param[i,8]
        As = out_param[i,9]
        out_param[i,8] = np.abs(Ap/np.cos(out_param[i,12]*np.pi/180.0))**0.5
        out_param[i,9] = np.abs(As/np.cos(out_param[i,13]*np.pi/180.0))**0.5
        
    return out_param

# End convert_ga_to_spam


def ReadAndCleanupData( filePath, thresh ):

    print("Cleaning target file...")

    # read data into np array
    df = pd.read_csv( filePath, sep=',|\t', engine='python', header=None )
    data1 = df.values

    # remove unranked models
    ind = 0
    while( not math.isnan(data1[ind,-1]) ):
        ind += 1
    # end
    data2 = data1[0:ind,:]
    nModel = ind + 1

    # include human score and SPAM params
    cols = list(range(4,18)) + [ 1 ]
    data2 = data2[:,cols]

    # ignore bad zoo models
    data2 = data2[data2[:,-1]>=thresh,:]
    nModel = data2.shape[0]

    data2 = np.array( data2, dtype=np.float32 )

#	data2[0,2]  = 10
#	data2[0,5]  = -10
#	data2[0,10] = 20
#	data2[0,11] = 20
#	data2[0,12] = 100
#	data2[0,13] = 100

    data3 = deepcopy(data2)

    psi = []
    for i in range(nModel):
        psi_p = 1
        psi_s = 1

        if( data2[i,2] < 0 ):
            data2[i,2]  =  -1 * data2[i,2]
            data2[i,5]  =  -1 * data2[i,5]
            data2[i,10] = 180 + data2[i,10]
            data2[i,11] = 180 + data2[i,11]
        # end
        data2[i,10] %= 360
        data2[i,11] %= 360
        if( data2[i,10] > 180 ):
            data2[i,10] = data2[i,10] - 180
            data2[i,12] = -1 * data2[i,12]
        # end
        if( data2[i,11] > 180 ):
            data2[i,11] = data2[i,11] - 180
            data2[i,13] = -1 * data2[i,13]
        # end
        data2[i,12] %= 360
        data2[i,13] %= 360
        if( data2[i,12] > 180 ):
            data2[i,12] = data2[i,12] - 360
        # end
        if( data2[i,13] > 180 ):
            data2[i,13] = data2[i,13] - 360
        # end

        if( data2[i,12] > 90 ):
            data2[i,12] = data2[i,12] - 180
            psi_p = -1
        elif( data2[i,12] < -90 ):
            data2[i,12] = data2[i,12] + 180
            psi_p = -1
        # end
        if( data2[i,13] > 90 ):
            data2[i,13] = data2[i,13] - 180
            psi_s = -1
        elif( data2[i,13] < -90 ):
            data2[i,13] = data2[i,13] + 180
            psi_s = -1
        # end
        psi.append( [psi_p,psi_s] )
    # end
    psi = np.array(psi)

    # energy
    G = 1
    r = ( data2[:,0]**2 + data2[:,1]**2 + data2[:,2]**2 )**0.5
    U = -G*data2[:,6]*data2[:,7]/r
    v = ( data2[:,3]**2 + data2[:,4]**2 + data2[:,5]**2 )**0.5
    K = 0.5*data2[:,7]*v**2
#	c = np.log(1-K/U)
    c = (K+U)/(K-U)

    # convert p,s mass to ratio,total mass
    t = data2[:,6] + data2[:,7]
    f = data2[:,6] / t
    data2[:,6] = f
    data2[:,7] = t

    # spherical velocity
    phi   = ( np.arctan2( data2[:,4], data2[:,3] ) * 180.0 / np.pi ) % 360
    theta = ( np.arcsin( data2[:,5] / v ) * 180.0 / np.pi )

#	data2[:,2] = c
#	data2[:,3] = v
    data2[:,3] = c
#	data2[:,2] = K
#	data2[:,3] = U

    data2[:,4] = phi
    data2[:,5] = theta

#	"""
    Ap = np.abs(data2[:,8]**2*np.cos(data2[:,12]*np.pi/180.0))
    As = np.abs(data2[:,9]**2*np.cos(data2[:,13]*np.pi/180.0))

    data2[:,8] = Ap
    data2[:,9] = As
#	"""

    return data2, psi, nModel, len(cols)

# end ReadandCleanUpData


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )