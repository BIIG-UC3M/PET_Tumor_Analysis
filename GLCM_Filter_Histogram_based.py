# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:46:35 2017
Initial
@author: pmacias
"""

import numpy as np
from scipy.ndimage.filters import generic_filter
import SimpleITK
import time

def generate_default_offsets(dim = 3, distance = 1): #TODO multidimensional
    if not (1 < dim <= 3):
        raise Exception('Incorrect number of dimesion for default offsets generation')
    else:
        distance = int(distance)
            
        if dim == 2:
            return [(0,distance),(-distance,distance),(-distance,0),(-distance,-distance)]
        if dim == 3:
            return [(0,0,distance),(0,-distance,distance),(0,-distance,0),(0,-distance,-distance),
                    (-distance,distance,0),(-distance,0,0),(-distance,-distance,0),(-distance,0,-distance),
                    (-distance,0,distance),(-distance,distance,-distance),(-distance,-distance,distance),
                    (-distance,-distance,-distance),(-distance,distance,distance)]

class Image_To_GLCM:
    
    def __init__(self, image, offset, bins = 256, min_max = None, mask = None, mask_val = 1, normalization = True):
        """
        offset as tuple with the coordinates. numpy ordered. From the center point
        """
        self.image = SimpleITK.GetArrayFromImage(image) if isinstance(image, SimpleITK.Image) else image;
        self.__offset = None
        self.set_offset(offset)
        self.bins = bins
        self.mask = mask
        self.mask_val = mask_val
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
        dim = len(self.__offset)
        offset_arr = np.array(self.__offset)
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
        self.__offset = offset
                                               
    
    def glcm(self):
        return np.histogramdd(np.column_stack((crop_per_offset(self.image, np.multiply(self.__offset,-1), mask=self.mask, mask_val=self.mask_val, cval=np.max(self.min_max) + 100).ravel(),
                                               crop_per_offset(self.image,self.__offset, mask=self.mask,mask_val=self.mask_val,cval=np.max(self.min_max) + 100 ).ravel())),bins=self.bins,
                                               range = tuple([min_max for min_max in self.min_max]),
                                               normed=self.normalization)[0]

    


def crop_per_offset(mat_, offset, mask = None, mask_val = 1, cval = -1):
    mat = mat_.copy()
    if mask is not None:
        mat[mask != mask_val] = cval

    crop = [range(off,mat.shape[dimension] ) if off > 0 else range(0,mat.shape[dimension] - np.abs(off) )
    for dimension,off in enumerate(offset) ]
    
    return mat[np.ix_(*crop)]
    
""" This is inline already
def crop_image(image, mask, mask_val = 1): #TODO for each dimension. As in crop_per_offset
    mask = mask == mask_val
    coords = np.argwhere(mask)
    if image.ndim == 3:
        z0,x0, y0 = coords.min(axis=0)
        z1,x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        return image[z0:z1, x0:x1, y0:y1], mask[z0:z1, x0:x1, y0:y1]
    elif image.ndim == 2:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        return image[x0:x1, y0:y1], mask[x0:x1, y0:y1]
    else:
        print "Cannot crop a",image.ndim,"dimensional image"
"""    
if __name__ == "__main__":
    image_test2 = np.random.randint(0,4, size = (3,5,5))
    mask = np.random.randint(0,2, size = (3,5,5))
    a = Image_To_GLCM(image_test2,(0,0,-1), bins=4, normalization=False, mask=mask)
    a2 = Image_To_GLCM(image_test2,(0,0,-1), bins=4, normalization=False)
    print a2.glcm(),'\n'
    print a.glcm(),'\n'

"""   
    def glcm(self):
        good_neigs = np.ones(self.image.shape,dtype=np.bool)
        #bad_neigh must be bigger than the axis limmit
        bad_neig = np.max(self.min_max) + 1
        footprint = self.__create_footprint__()
        
                    
        if self.mask is not None:
            mask_neigs = generic_filter(self.mask , lambda x:x, footprint=footprint, mode='constant', cval= 0)
            good_neigs[np.logical_or(mask_neigs == 0 , self.mask == 0)] = False
        
        #start = time.time()    
        neigs = generic_filter(self.image , lambda x:x, footprint=footprint, mode='constant', cval= bad_neig)
        
        #print 'pregistogram',self.__offset, time.time() - start
        good_neigs = np.logical_and(good_neigs, (neigs < bad_neig))
        

        return np.histogramdd(np.column_stack((self.image[good_neigs].ravel(),
                                               neigs[good_neigs].ravel())),
                                                bins=self.bins,range = tuple([min_max for min_max in self.min_max]),normed=self.normalization)[0]
 
"""        
