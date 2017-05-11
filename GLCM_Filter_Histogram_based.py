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
    
    def __init__(self, image, offset, bins = 256, min_max = None, mask = None, normalization = True):
        """
        offset as tuple with the coordinates. numpy ordered. From the center point
        """
        self.image = SimpleITK.GetArrayFromImage(image) if isinstance(image, SimpleITK.Image) else image;
        self.offset = None
        self.set_offset(offset)
        self.bins = bins
        self.mask = mask
        self.normalization = normalization
        #bins from to to each axes. GLCM is 2 by 2 histogram so that's why we use two axes
        self.histogram_axis = 2
        self.min_max = [[np.min(image), np.max(image) + 1]] * self.histogram_axis if min_max == None else min_max
        try:
            if not isinstance(self.min_max,list) or sum([  ((b[1]- b[0]) > 0) * len(b) == 2 for b in self.min_max ]) != self.histogram_axis:
                raise Exception("Incorrect number of axes for GLCM matrix. Must be 2. i.e. [[low1,upper1], [low2,upper2]]")
        except:
                raise Exception("Incorrect histogram axis form. i.e. [[low1,upper1], [low2,upper2]]")

        
        
    def __create_footprint__(self):#TODO Better as sparse matrix??
        dim = len(self.offset)
        offset_arr = np.array(self.offset)
        distance = np.max(np.abs(offset_arr))
        size = np.array([2*distance+1]*dim)
        center = size/2
        footprint = np.zeros( size , dtype = np.bool)
        footprint[tuple(center + offset_arr)] = True
        #print footprint
        return footprint
        
    def set_offset(self,offset):
        if not isinstance(offset, tuple) or self.image.ndim != len(offset) or np.max(np.abs(offset)) > self.image.shape[np.argmax(np.abs(offset))]/2:
            raise Exception('Incorrect offset offset as tuple with the coordinates')
        self.offset = offset
        
    def glcm(self):
        """
        Compute the GLCM matrix at the offset previously established
        
        Returns
        ---------
        The glcm matrix
        """
        good_neigs = np.ones(self.image.shape,dtype=np.bool)
        bad_neig = np.max(self.image) + 1
        footprint = self.__create_footprint__()
                         
        if self.mask is not None:
            mask_neigs = generic_filter(self.mask , lambda x:x, footprint=footprint, mode='constant', cval= 0)
            good_neigs[np.logical_or(mask_neigs == 0 , self.mask == 0)] = False
            
        neigs = generic_filter(self.image , lambda x:x, footprint=footprint, mode='constant', cval= bad_neig)
        good_neigs = np.logical_and(good_neigs, (neigs < bad_neig))

        return np.histogramdd(np.column_stack((self.image[good_neigs].ravel(),
                                               neigs[good_neigs].ravel())),
                                                bins=self.bins,range = tuple([min_max for min_max in self.min_max]),normed=self.normalization)[0]
        

    def spherical_to_cartesian(theta, phi,r=1):#TODO Esto está mal
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
        
    
    def generate_default_offsets(dim = 3, distance = 1):
    
    