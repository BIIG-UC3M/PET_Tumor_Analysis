# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:46:35 2017

@author: pmacias
"""


import numpy as np
from scipy.ndimage.filters import generic_filter
import time
import SimpleITK
#
#a = SimpleITK.ReadImage('/home/biig/Desktop/pruebas texturas/01_PET1_VOI.nii')
#a = SimpleITK.GetArrayFromImage(a)
#mk=SimpleITK.ReadImage('/home/biig/Desktop/pruebas texturas/01_PET1_VOI_label.nrrd')
#mk = SimpleITK.GetArrayFromImage(mk)
#
#A = np.array([i for i in range(27)]).reshape((3,3,3)) + 1
#
##64*64*12 with ints
#intsImage = np.random.randint(1025, size=(64,64,12))
#
#footprint2 = np.array([[0,0,0],[0,0,1],[0,0,0]])
#footprint3 = np.stack((np.zeros((3,3)),footprint2, np.zeros((3,3))))
#
#start = time.time()
#results = generic_filter(intsImage , lambda x:x, footprint=footprint3, mode='constant', cval=-1)
#H, _ = np.histogramdd(np.column_stack((intsImage.ravel() ,results.ravel())), bins=1025)
#print time.time() - start
##               
#
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
#    
#vectors=np.zeros([81,3])
#
#count=0
#count2=0
#count3=0
#for i in [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4, 2*np.pi]:
#    for j in [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4, 2*np.pi]:
#        k=spherical_to_cartesian(i,j,r=1)
#        if count == 0:
#            print('hello')
#        else:
#            count2=0
#            for i in range(count): 
#              if np.array_equal(np.absolute(k),vectors[i,:])==False:
#                  count2=count2+1
#                 
#        
#            if count2==count:
#                print(k)
#        vectors[count,:]=k
#        count=count+1
#       # print(k)
#       

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
    print  height - center, row - center, col -center
    return footprint



#
#def create_footprint(distance, dim, direction):
#    footprint = np.zeros([2*distance+1]*dim)
#    angles = np.array([ (-1,-1,-1),  #izq arriba (-1,1,0)
#             (0, -1, -1), #izq
#             (1, -1, -1), #arriba hacia delante
#             (-1, 0, -1),
#             (0, 0, -1),
#             (1, 0, -1),
#             (-1, 1, -1),
#             (0, 1, -1),
#             (1, 1, -1),
#             (-1, -1, 0),
#             (0, -1, 0),
#             (1, -1, 0),
#             (-1, 0, 0)]) 
#
#    center=(2*distance+1)/2
#    angle=angles[direction]
#    col=center+distance*angle[0]
#    row = center+distance*angle[1]
#    height = center+distance*angle[2]
#    footprint[height,row,col]=1
#   #print footprint
#    return footprint

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
    

#b=create_GLCM(a,1,3,1,256,mask=mk,normalization=False)
#is normalized?

#np.sum(b)
#
##test GLCM
#d=np.ones([87,144,144])
#e=create_GLCM(d,1,3,1,256,normalization=True)
#d=np.zeros([87,144,144])
#e=create_GLCM(d,1,3,1,256,normalization=True)

##Test
#X = np.random.randint(256,size=(3,3))
#start = time.time()glcm_features_extraction(image, mk=None, dim = 3, distances = [1], angles = 1, levels = 6, symmetry = 2, normed = False, include_angles_average = True)
#for i in range(64*64*12*13):
#    GLCM_at_0(X,bins=16)
#print time.time() - start

#footprint = np.zeros([2*distance+1]*dim)
#angles = np.array([ (1,0,0),
#             (0,1,0),
#             (0,0,1),
#             (1,1,0),
#             (-1,1,0),
#             (1,0,1),
#             (-1,0,1),
#             (0,1,1),
#             (0,-1,1),
#             (1,1,1),
#             (-1,1,1),
#             (1,-1,1),
#             (1,1,-1)]) 
#  
#     
#for i in range(13):
#    center=1
#    row=0
#    col=0
#    height=0
#    angle=angles[i]
#    row=center+distance*angle[2]
#    col = center+distance*angle[1]
#    height = center+distance*angle[0]
#    footprint[height,col,row]= footprint[height,col,row]+1
#             
#             
#             
#footprint
    



## TRIAL WITH AN IMAGE FROM MATLABCódigo de identificación personal autonómico
#By default, graycomatrix calculates the GLCM based on horizontal proximity of the pixels: [0 1]

#
#
#a = SimpleITK.ReadImage('/home/biig/img.tif')
#a = SimpleITK.GetArrayFromImage(a)
#
#
#    
#def create_footprint2(distance,dim, direction):
#    footprint = np.zeros([2*distance+1]*dim)
#    angle=direction
#    center=1
#    col = center+distance*angle[1]
#    height = center+distance*angle[0]
#    footprint[height,col]=1
#    return footprint
#
#
#def create_GLCM2(image,distance,dim,direction, bins = 256, mask = None,normalization=False):
#    footprint = create_footprint2(distance,dim,direction)
#    good_neigs = np.ones(image.shape,dtype=np.bool)
#    bad_neig = np.max(image) + 1
#                     
## CREO QUE ESTO YA NO HACE FALTA XQ EL FOOTPRINT YA ES 3D
##    if len(image.shape) == 3:
##        footprint = np.stack((np.zeros((3,3)),footprint, np.zeros((3,3))))
#        
#    if mask is not None:
#        mask_neigs = generic_filter(mask , lambda x:x, footprint=footprint, mode='constant', cval= 0)
#        good_neigs[np.logical_or(mask_neigs == 0 , mask == 0)] = False
#        
#    neigs = generic_filter(image , lambda x:x, footprint=footprint, mode='constant', cval= bad_neig)
#    good_neigs = np.logical_and(good_neigs, (neigs < bad_neig))
#    return np.histogramdd(np.column_stack((image[good_neigs].ravel() , neigs[good_neigs].ravel())), bins=bins, range = ((0,bins),(0,bins)),normed=normalization)[0]
#
#d=np.array([[ 0, 0, 1, 1],[ 0, 0, 1, 1],[0, 2, 2, 2],[2,2,3,3]])
#b=create_GLCM2(d,1,2,[0,1], bins = 8, mask = None,normalization=False)