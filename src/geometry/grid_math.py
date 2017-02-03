# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 17:18:09 2015

@author: Nate

SIDDON uses Siddon's method (Med. Phys. 12(2), Mar/Apr 1985) to find the
cells through which a vector passes and the length of the vector found in
each cell. Note that in the writing of this function, as much of the
author's notation was preserved as possible to make it easier to
understand. There have been improvedments to Siddon's method proposed, as
in Jacobs et. al, however they are not accounted for here.
11/7/12 by Nathan T. Depenbusch in Matlab
01/6/15 by Nathan T. Depenbusch moved to Python
"""
import numpy as np

def siddon_algorithm(X1, Y1, X2, Y2, Xplane, Yplane):
    """Uses Siddon's (1985) method to compute cell crossings. This algorithm is
    pretty fast for even large cell arrays.

    Args:
        X1: the x coordinate of the first point
        Y1: the y coordinate of the first point
        X2: the x coordinate of the end point
        Y2: the y coordinate of the end point
        Xplane: the x coordinates of the gridlines dividing cells
        Yplane: the y coordinate of the gridlines dividing cells

    Returns:
        A list of two numpy arrays
        cells: the first array is the i coordinate and the second column the j
            coordinates of the cell corresponding to l.
        l: the length of the full line that lies in each cell

        (cells, l)

        Note that if a length of the path falls outside of the grid, cells will
        return [-1,-1] in the appropriate row
    """
    # make Xplane and Yplane numpy arrays (harmless if it's already a numpy array)
    Xplane = np.array(Xplane)
    Yplane = np.array(Yplane)
    # specify the distance between planes (should be regular here)
    dx = np.abs(Xplane[1] - Xplane[0])
    dy = np.abs(Yplane[1] - Yplane[0])
    # find the number of grid lines dividing cells, note that there are (Nx-1,Ny-1) voxels in this 2d array
    Nx = Xplane.size
    Ny = Yplane.size

    # calculate the range of parametric values
    if (X2-X1) != 0.0:
        ax = np.zeros(Nx)
        ax[0] = (Xplane[0]-X1)/(X2-X1)
        ax[Nx-1] = (Xplane[Nx-1]-X1)/(X2-X1)
    else:
        ax = np.zeros(0)
    if (Y2-Y1) != 0.0:
        ay = np.zeros(Ny)
        ay[0] = (Yplane[0]-Y1)/(Y2-Y1)
        ay[Ny-1] = (Yplane[Ny-1]-Y1)/(Y2-Y1)
    else:
        ay = np.zeros(0)

    if (ax.size > 0) and (ay.size > 0):
        amin = max([0.0, min(ax[0], ax[Nx-1]), min(ay[0], ay[Ny-1])])
        amax = min([1.0, max(ax[0], ax[Nx-1]), max(ay[0], ay[Ny-1])])
    elif (ax.size == 0) and (ay.size > 0):
        amin = max([0, min(ay[0], ay[Ny-1])])
        amax = min([1, max(ay[0], ay[Ny-1])])
    elif (ay.size == 0) and (ax.size > 0):
        amin = max([0, min(ax[0], ax[Nx-1])])
        amax = min([1, max(ax[0], ax[Nx-1])])
    else:
        amin = 0.0
        amax = 1.0

    # Calculate the range of indices covered
    if (X2-X1)>=0:
        imin = Nx - np.floor((Xplane[Nx-1] - amin*(X2-X1) - X1)/dx)
        imax = 1 + np.floor((X1 + amax*(X2-X1) - Xplane[0])/dx)
    else:
        imin = Nx - np.floor((Xplane[Nx-1] - amax*(X2-X1) - X1)/dx)
        imax = 1 + np.floor((X1 + amin*(X2-X1) - Xplane[0])/dx)
    if (Y2-Y1)>=0:
        jmin = Ny - np.floor((Yplane[Ny-1] - amin*(Y2-Y1) - Y1)/dy)
        jmax = 1 + np.floor((Y1 + amax*(Y2-Y1) - Yplane[0])/dy)
    else:
        jmin = Ny - np.floor((Yplane[Ny-1] - amax*(Y2-Y1) - Y1)/dy)
        jmax = 1 + np.floor((Y1 + amin*(Y2-Y1) - Yplane[0])/dy)

    # Calculate parametric sets
    if ax.size > 0:
        i = imin
        for p in range(0, (int(imax-imin)+1)):
            ax[p] = (Xplane[i-1]-X1)/(X2-X1)
            i = i + 1
        ax = ax[0:(int(imax-imin)+1)]
    if ay.size > 0:
        j = jmin
        for p in range(0, (int(jmax-jmin)+1)):
            ay[p] = (Yplane[j-1]-Y1)/(Y2-Y1)
            j = j + 1
        ay = ay[0:(int(jmax-jmin)+1)]

    # merge sets to form a
    alpha = np.unique(np.hstack([amin, ax, ay, amax]))
    # n is the index of the last term in a (n-1)
    #n = (imax-imin+1)+(jmax-jmin+1)
    # distance from point 1 to point 2
    d12 = np.sqrt((X2-X1)**2.0+(Y2-Y1)**2.0)

    # calculate voxel lengths
    # The pixel that contains the midpoint of the intersections that bound a
    # length contains the entirety of that length. We use this obvious fact
    # to return the indices of cells crossed by the vector
    l = np.zeros(alpha.size).astype(float)
    i = np.zeros(alpha.size).astype(int)
    j = np.zeros(alpha.size).astype(int)
    for m in range(1, alpha.size):
        l[m] = d12*(alpha[m]-alpha[m-1]);
        # find the midpoint of each length
        amid = (alpha[m]+alpha[m-1])/2.0;
        # Find the x index
        i[m] = np.floor((X1 + amid*(X2-X1)-Xplane[0])/dx) # 0 indexed, otherwise +1
        # find the y index
        j[m] = np.floor((Y1 + amid*(Y2-Y1)-Yplane[0])/dy) # 0 indexed, otherwise +1
    # remove the first index
    l = np.delete(l, 0)
    i = np.delete(i, 0)
    j = np.delete(j, 0)

    # now lets deal with the case when the end point is outside of the grid
    if amax < 1.0:
        arem = 1-amax
        #l = np.hstack([l, (arem*d12)])
        #cells = np.vstack([cells,[-1,-1]])
        l = np.append(l, (arem*d12))     
        i = np.append(i, -1) 
        j = np.append(j, -1) 
        
    # and of course the case where the start point is outside of the grid
    if amin > 0.0:
        arem = amin
        #l = np.hstack([(arem*d12), l])
        #cells = np.vstack([[-1, -1], cells])
        l = np.insert(l, 0, (arem*d12))     
        i = np.insert(i, 0, -1) 
        j = np.insert(j, 0, -1) 

    # put cells together as a tuple to make indexing obvious
    cells = (i, j)
    # return the cells and the lengths
    return cells, l

#Xplane = 25.0*np.arange(-40, 40)
#Yplane = 25.0*np.arange(-50, 50)
#l,cells = siddon_algorithm(-983.1,-1156.4,965.3,1148.7,Xplane,Yplane)

