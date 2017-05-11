# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:46:35 2017
Initial
@author: pmacias
"""


import numpy as np
from scipy.ndimage.filters import generic_filter
import SimpleITK

class Image_To_GLCM:
    
    def __init__(self, image, offset):
        """
        offset as tuple with the coordinates. numpy ordered. From the center point
        """
        self.image = SimpleITK.GetArrayFromImage(image) if isinstance(image, SimpleITK.Image) else image;
        self.offset = offset
        self.footprint = self.__create_footprint__()
        
    def __create_footprint__(self):
        dim = len(self.offset)
        offset_arr = np.array(self.offset)
        distance = np.max(np.abs(offset_arr))
        size = np.array([2*distance+1]*dim)
        center = size/2
        self.footprint = np.zeros( size , dtype = np.bool)
        self.footprint[tuple(center + offset_arr)] = True
        print self.footprint
        

    def spherical_to_cartesian(theta, phi,r=1):
        """
        In radians
        r = distance
        theta = azimuth
        phi = polar
        """
        x =  int(np.rint(np.sin(phi)*np.cos(theta)))
        y =  int(np.rint(np.sin(theta))*np.rint(np.sin(phi)) )
        z =  int(np.cos(theta))
        return np.array((x,y,z)) * r
    
    
    # TODO This way just works in 3D and for anologous direction of the 2D (0,45,90,135) . It should be proper to make this work on any dimension. needs to extend the angles dimension.
    # TODO the footprint is already extended    
    def create_footprint(distance, dim, direction):
        footprint = np.zeros([2*distance+1]*dim)
        angles = np.array([ (1,0,0),  #izq arriba (-1,1,0)
                 (0,1,0), #izq
                 (0,0,1), #arriba hacia delante
                 (1,1,0),
                 (-1,1,0),
                 (1,0,1),
                 (-1,0,1),
                 (0,1,1),
                 (0,-1,1),
                 (1,1,1),
                 (-1,1,1),
                 (1,-1,1),
                 (1,1,-1)]) 
    
        center=(2*distance+1)/2
        angle=angles[direction]
        col=center+distance*angle[0]
        row = center+distance*angle[1]
        height = center+distance*angle[2]
        footprint[height,row,col]=1
       #print footprint
        return footprint
    
    
    
    
    def create_GLCM(image,distance = 1,dim = 3,direction = 0, bins = 256, axis_range = None, mask = None ,normalization = True):
        """
        
        Paramenters
        ------------
        
        bins : int or sequence for different slots per axes    
        axis_range : iterable-like
                     The range of each axes at the histogram.
        """
        axis_range = [[np.min(image), np.max(image) + 1]] * 2 if axis_range == None else axis_range
        print axis_range,"axis"
        footprint = create_footprint(distance,dim,direction)
        good_neigs = np.ones(image.shape,dtype=np.bool)
        bad_neig = np.max(image) + 1
                         
    # CREO QUE ESTO YA NO HACE FALTA XQ EL FOOTPRINT YA ES 3D
    #    if len(image.shape) == 3:
    #        footprint = np.stack((np.zeros((3,3)),footprint, np.zeros((3,3))))
    
    # TODO Same thing about footprint
                         
        if mask is not None:
            mask_neigs = generic_filter(mask , lambda x:x, footprint=footprint, mode='constant', cval= 0)
            good_neigs[np.logical_or(mask_neigs == 0 , mask == 0)] = False
            
        neigs = generic_filter(image , lambda x:x, footprint=footprint, mode='constant', cval= bad_neig)
        good_neigs = np.logical_and(good_neigs, (neigs < bad_neig))

        return np.histogramdd(np.column_stack((image[good_neigs].ravel(),
                                               neigs[good_neigs].ravel())),
                                                bins=bins,range = (axis_range[0],axis_range[1]),normed=normalization)[0]
        
    