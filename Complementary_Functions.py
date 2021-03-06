# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:22:23 2017

@author: pmacias
"""

import os
import glob
import time
import numpy as np
import SimpleITK
from GLCM_Features_Calc import Haralick_Features
from GLCM_Filter_Histogram_based import generate_default_offsets

def argus2SimpleITK(path):
  """
  Transforms Argus/IDL image types to SimpleITK usual
  """
  typemap = {"unsigned integer":'u2', "signed integer":'i2', "unsigned long":'u4',
 "signed long":'i4', "short float":'f4', "long float":'f8', "byte":np.uint8}
   
  hdrMap = {};
  #Get fields to open raw as image
  with open(path) as f:
    for line in f:
      splitted = line.split(':=') #All to lower case in order to avoid problems
      if(len(splitted) > 1):
        hdrMap[splitted[0].strip()] = splitted[1].strip()


#  print hdrMap['number format']
  rawFile = os.path.join(os.path.dirname(path), hdrMap['name of data file'])
  if os.path.exists(rawFile)==False:
      rawFile=os.path.join(os.path.dirname(path),  os.path.splitext(hdrMap['name of data file'])[0].upper()+os.path.splitext(hdrMap['name of data file'])[1])
  #TODO sometimes matrixsize[3] doesnt exists. Check for NumberTotalImages within hdrMap
  if 'matrix size [3]' in hdrMap:
      dims = (int(hdrMap['matrix size [2]']), int(hdrMap['matrix size [1]']),int( hdrMap['matrix size [3]']))
  elif 'NumberTotalImages' in hdrMap:
      dims = (int(hdrMap['matrix size [2]']), int(hdrMap['matrix size [1]']),int( hdrMap['NumberTotalImages']))
   
  spacing = ( float(hdrMap['scaling factor (mm/pixel) [2]']), float( hdrMap['scaling factor (mm/pixel) [1]']), float( hdrMap['scaling factor (mm/pixel) [3]']) )
  offset = (float(hdrMap['offset [2]']), float(hdrMap['offset [1]']), float(hdrMap['offset [3]']))
  endian = hdrMap['imagedata byte order']
  endiannes = '>' if  endian == 'bigendian' else '<'
  
#  print "Dims: ",dims, "Spacing: " ,spacing, "Offset: ",offset
  dataType = np.int16;
  if(int(hdrMap['number of bytes per pixel']) == 1):
    dataType = np.dtype(typemap["byte"])
  else:
    dataType = np.dtype(endiannes + typemap[hdrMap['number format']])
  
  arrayImage = np.fromfile(rawFile, dtype = dataType)
  dims = list(reversed(dims))
  arrayImage = arrayImage.reshape(dims, order = 'C')
  arrayImage = arrayImage.astype('<i2')

  itkImage = SimpleITK.GetImageFromArray(arrayImage)
  itkImage.SetOrigin(offset)
  itkImage.SetSpacing(spacing)
  return itkImage

def get_arrayImage(image):
    return SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(image)) if isinstance(image,str) else SimpleITK.GetArrayFromImage(image) if isinstance(image, SimpleITK.Image) else image

def read_dicom(path):
    reader = SimpleITK.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(path))
    return reader.Execute()
    
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_multilabel_textures(image,multi_label_mask, bins = 256,background_label = 0, distance = 1, extra_info_as_dicc = {}):
    image = get_arrayImage(image)
    multi_label_mask = get_arrayImage(multi_label_mask)
    labels = np.unique(multi_label_mask)
    all_labels_info = []
    print len(labels)
    for label in np.delete(labels, np.argwhere(labels == background_label)):
        print 'label',label
        har_feats = Haralick_Features(image, mask=multi_label_mask, mask_val=label, bins=bins, distance=distance)
        #har_feats.compute_features()
        sup_feats = har_feats.get_features_all_offsets()
        for i,sup in enumerate(sup_feats):
            sup_feats[i] = merge_dicts(sup, {'Label':label},extra_info_as_dicc)
        all_labels_info += sup_feats
    return all_labels_info
    

if __name__ == "__main__":
    labels_path = '/media/pmacias/DATA1/Lessions_Test/mask_lesions_STUDY_4863_R62_8SEPTEMBER2013_16W.mhd'
    raw_image_path = '/media/pmacias/DATA1/Lessions_Test/STUDY_4863_R62_8SEPTEMBER2013_16W.mhd'
    
#    raw_image = SimpleITK.ReadImage(raw_image_path)
#    labels = SimpleITK.ReadImage(labels_path)
#    
#    raw_image = SimpleITK.GetArrayFromImage(raw_image)
#    labels = SimpleITK.GetArrayFromImage(labels)
    start = time.time()
    a =  get_multilabel_textures(raw_image_path, labels_path, extra_info_as_dicc= {'todo':'Es','un':'cagarro'})
    print 'Time',time.time() - start
#    
    
#    main_path = "/media/pmacias/DATA1/amunoz/LOTE_1/Classified_lesions/"
#    studies = ['STUDY_4863', 'STUDY_4864', 'STUDY_4974', 'STUDY_2087','STUDY_2088','STUDY_2089','STUDY_2090']
#    #studies = ['STUDY_4863']
#    
#    for study in studies:
#        images = glob.glob('/media/pmacias/DATA1/amunoz/LOTE_1/'+study+'/*/*/*_interfile/*.hdr')
#        #lesions = glob.glob('/media/pmacias/DATA1/amunoz/LOTE_1/'+study+'/*/*/*_interfile/LESIONS/mask_lesions_*_modified.hdr')
#        for im in images:
#            dirpath = os.path.dirname(im)
#            lesions = glob.glob(dirpath+'/*/mask_lesions_*_modified.hdr')
#            if len(lesions) == 0:
#                lesions = glob.glob(dirpath+'/*/mask_lesions_*.mhd')
#            fields = lesions[0].split('/')[-1].split('.')[0].split('_')
#            print im
#            print fields[2], fields[3], fields[4], fields[5], fields[6]
#            print ''
#            SimpleITK.WriteImage(argus2SimpleITK(im), '/media/pmacias/DATA1/Lessions_Test/'+fields[2]+'_'+fields[3]+'_'+fields[4]+'_'+fields[5]+'_'+fields[6]+'.mhd')
#            SimpleITK.WriteImage(SimpleITK.ReadImage(lesions[0]), '/media/pmacias/DATA1/Lessions_Test/'+fields[0]+'_'+fields[1]+'_'+fields[2]+'_'+fields[3]+'_'+fields[4]+'_'+fields[5]+'_'+fields[6]+'.mhd')
#            
#            
            
            