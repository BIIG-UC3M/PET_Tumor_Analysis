# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:54:41 2017
Initial
@author: pmacias
"""
from GLCM_Filter_Histogram_based import Image_To_GLCM, generate_default_offsets
import numpy as np
import SimpleITK
from multiprocessing import  Process, Pool
from matplotlib import pyplot as plt
import time

_HARALICK = "HARALICK"
_OFFSETWA = "offset_workaround_parrallel_computing"

def compute_fake(haralick_dic):
    "Print in fake"
    Haralick = haralick_dic['HARALICK']
    return  Haralick._compute_features_(haralick_dic[_OFFSETWA])
    

def uniform_quantization(image,l = 256, g_max = 255.0, g_min = 0, out_type = np.uint8):
    """
    l --> quantization levels
    return floor(image*k1+k2)
    k1 = (l -1) / (g_max - g_min)
    k2 = 1 -k1*g-min
    """
    if l -1 > g_max:
        l = g_max +1
    k1 = (l -1) / (g_max - g_min)
    k2 = 1 - k1*g_min
    return np.floor(image*k1+k2).astype(out_type)


class Texture_Features:
    glcm_feats = ['Energy','Contrast','Correlation1','SumOfSquares','Homogenity2',
    'SumAverage','SumVariance','SumEntropy','Entropy','DifferenceVariance'
    ,'DifferenceEntropy','Autocorrelation','Correlation2','ClusterProminance','ClusterShade'
    ,'Dissimilarity','Homogenity1','InverseDifferentMomentNormalized','MaximumProbability','InformationMeasureCorrelation1'
    ,'InformationMeasureCorrelation2','InverseDifferenceNormalized']
    def __init__(self,Haralick_Features, n_offset = 0):
        self.n_offset = n_offset
        if -1 < n_offset < Haralick_Features.n_offsets:
            features_vector = Haralick_Features.features_matrix[n_offset]
        else:
            self.n_offset = -1
            features_vector = Haralick_Features.get_average_features()
        self.Energy = features_vector[0]
        self.Contrast = features_vector[1]
        self.Correlation1 = features_vector[2]
        self.Sum_Of_Squares = features_vector[3]
        self.Homogenity_2 = features_vector[4]
        self.Sum_Average = features_vector[5]
        self.Sum_Variance = features_vector[6]
        self.Sum_Entropy = features_vector[7]
        self.Entropy = features_vector[8]
        self.Difference_Variance = features_vector[9]
        self.Difference_Entropy = features_vector[10]
        self.Autocorrelation = features_vector[11]
        self.Correlation2 = features_vector[12]
        self.Cluster_Prominance = features_vector[13]
        self.Cluster_Shade = features_vector[14]
        self.Dissimilarity = features_vector[15]
        self.Homogenity_1 = features_vector[16]
        self.Inverse_Different_Moment_Normalized = features_vector[17]
        self.Maximum_Probability = features_vector[18]
        self.Information_Measure_Correlation_1 = features_vector[19]
        self.Information_Measure_Correlation_2 = features_vector[20]
        self.Inverse_Difference_Normalized = features_vector[21]
        
    def get_texture_feats_as_dicc(self):
        return {'n_offset':self.n_offset,'Energy':self.Energy, 'Contrast':self.Contrast,'Correlation1':self.Correlation1, 'Sum_of_Squares':self.Sum_Of_Squares,
                'Homogenity_2':self.Homogenity_2,'Sum_Average':self.Sum_Average,'Sum_Variance':self.Sum_Variance,'Sum_Entropy':self.Sum_Entropy,
                'Entropy':self.Entropy, 'Difference_Variance':self.Difference_Variance,'Difference_Entropy':self.Difference_Entropy,'Autocorrelation':self.Autocorrelation,
                'Correlation2':self.Correlation2, 'Cluster_Prominance':self.Cluster_Prominance,'Cluster_Shade':self.Cluster_Shade,'Dissimilarity':self.Dissimilarity, 
                'Homogenity_1':self.Homogenity_1,'Inverse_Different_Moment_Normalized':self.Inverse_Different_Moment_Normalized, 'Maximum_Probability':self.Maximum_Probability,
                'Information_Measure_Correlation_1':self.Information_Measure_Correlation_1,'Information_Measure_Correlation_2':self.Information_Measure_Correlation_2,'Inverse_Difference_Normalized':self.Inverse_Difference_Normalized }

class Haralick_Features():
    def __init__(self,image, mask = None, mask_val = 1, offsets = None, distance = 1, bins = 256, axis_range = None, normalization = True, save_glcm_matrices = True):      
        self.n_feats = 22
        self.image = SimpleITK.GetArrayFromImage(image) if isinstance(image, SimpleITK.Image) else image  
        if mask is None:
            self.mask = mask
            #self.image = image
        else:
            ## Crop the image, can save a lot of memory. Indices by pointer
            b = np.where(mask == mask_val)
            crop = [range(np.min(z), np.max(z)+1)  for z in b]
            self.mask = mask[np.ix_( *crop )]
            self.image = image[np.ix_( *crop )]
        self.mask_val = mask_val
        self.mask[self.mask != self.mask_val] = 0 #cleaning shit withinin the smaller version
        self.bins = bins
        self.normalization = normalization
        image_dim = image.ndim if isinstance(image, np.ndarray) else image.GetDimension()
        self.offsets = generate_default_offsets(dim = image_dim, distance = distance) if offsets is None else offsets
        self.n_offsets = len(self.offsets)        
        self.axis_range = axis_range
        self.save_glcms = save_glcm_matrices
        self.glcm_matrices = np.zeros([self.bins, self.bins, self.n_offsets]) if self.save_glcms else None
        self.features_matrix = np.zeros((self.n_offsets,self.n_feats))
        self.distance = distance
        
    def _compute_features_(self, offset):
        #print "Computing offset",offset
        g = Image_To_GLCM(self.image, offset, bins = self.bins, min_max = self.axis_range,
                          mask=self.mask,mask_val=self.mask_val, normalization=self.normalization)
        #start = time.time()
        g = g.glcm()
        #print 'glcm time',offset, time.time() - start
                          
        #start = time.time()              
        #TODO this just works for glcm with the same number of bins in each axes
        I, J = np.ogrid[0:self.bins, 0:self.bins]
        I = 1+ np.array(range(self.bins)).reshape((self.bins, 1))
        J = 1+ np.array(range(self.bins)).reshape((1, self.bins))
        IminusJ = I-J
        power_IminusJ = np.power(IminusJ,2)
        abs_IminusJ = np.abs(IminusJ)
        IplusJ = I+J
        
        mu_x = np.apply_over_axes(np.sum, (I * g), axes=(0, 1))[0, 0]
        mu_y = np.apply_over_axes(np.sum, (J * g), axes=(0, 1))[0, 0] 
        diff_i = I - mu_x
        diff_j = J - mu_y
        std_i = np.sqrt(np.apply_over_axes(np.sum, (g * (diff_i) ** 2),axes=(0, 1))[0, 0])
        std_j = np.sqrt(np.apply_over_axes(np.sum, (g * (diff_j) ** 2),axes=(0, 1))[0, 0])
        cov = np.apply_over_axes(np.sum, (g * (diff_i * diff_j)),axes=(0, 1))[0, 0]
        
        gxy = np.zeros(2*g.shape[0]-1)   ### g x+y
        gx_y = np.zeros(g.shape[0])  ### g x-y       
        for i in xrange(g.shape[0]):
            for j in xrange(g.shape[0]):
                gxy[i+j] += g[i,j]
                gx_y[np.abs(i-j)] += g[i,j]  
        mx_y = (gx_y*np.arange(len(gx_y))).sum()
        i,j = np.indices(g.shape)+1
        ii = np.arange(len(gxy))+2
        ii_ = np.arange(len(gx_y))
        
        features_vector = np.zeros((self.n_feats,))
        
        ### compute descriptors ###
        features_vector[0] = np.sum(np.power(g,2)) # Angular second moment. Energy = sqrt(ASM)
        features_vector[1] = np.apply_over_axes(np.sum, (g * power_IminusJ), axes=(0, 1))[0, 0] # Contrast
        if std_i>1e-15 and std_j>1e-15: # handle the special case of standard deviations near zero
            features_vector[2] = cov/(std_i*std_j)#v[2] = greycoprops(g,'correlation') # Correlation1
        else:
            features_vector[2] = 1
        features_vector[3] = np.apply_over_axes(np.sum, (g* (diff_i) ** 2),axes=(0, 1))[0, 0]# Sum of squares
        features_vector[4] = np.sum(np.divide(g,1+power_IminusJ)) #Homogenity 2
        features_vector[5] = (gxy*ii).sum() # Sum average
        features_vector[6] = ((ii-features_vector[5])*(ii-features_vector[5])*gxy).sum() # Sum variance
        features_vector[7] = -1*(gxy*np.log(gxy+ np.spacing(1))).sum() # Sum entropy
        features_vector[8] = -1*(g*np.log(g+np.spacing(1))).sum() # Entropy
        features_vector[9] = ((ii_-mx_y)*(ii_-mx_y)*gx_y).sum() # Difference variance
        features_vector[10] = -1*(gx_y*np.log(gx_y+np.spacing(1))).sum() # Difference entropy
        features_vector[11] = np.sum(I*J*g) #Autocorrelation
        features_vector[12] = (features_vector[11] - mu_x*mu_y)/(std_i*std_j) #Correlation2
        features_vector[13] = np.sum(np.power(IplusJ-mu_x-mu_y,4)*g) #Cluster prominence
        features_vector[14] = np.sum(np.power(IplusJ-mu_x-mu_y,3)*g) #Cluster shade
        features_vector[15] = np.sum(abs_IminusJ*g) #Dissimilarity
        features_vector[16] = np.sum(np.divide(g,1+abs_IminusJ)) #Homogenety 1
        features_vector[21] = np.sum(np.divide(g,1+(abs_IminusJ/ float(self.bins) )))  #Inverse difference normalized
        features_vector[17] = np.sum(np.divide(g,1+(power_IminusJ/float(self.bins) ))) # Inverse difference moment normalized
        features_vector[18] = np.max(g) #maximum probability 
        p_x = np.sum(g,axis=0); p_y = np.sum(g,axis=1) #marginals
        hx = -np.dot(p_x, np.log(p_x+np.spacing(1)))
        hy = -np.dot(p_y, np.log(p_y+np.spacing(1)))
        #hxy = Entropy
        hxy = features_vector[8]
        p_x = p_x.reshape((self.bins,1))
        p_y = p_y.reshape((self.bins,1))
        marginals_multi = p_x*p_y.T
        marginals_log = np.log(marginals_multi+np.spacing(1))
        hxy1 = -np.sum(g*marginals_log)
        hxy2 = -np.sum(np.multiply(marginals_multi,marginals_log))
        features_vector[19] = (hxy - hxy1)/max(hx,hy)  #Information measure of correlation 1
        #TO aovid sqrt of negative numbers its took the ||hxy2-hxy||. Not a clue about the best approach to this
        features_vector[20] = np.power(1 - np.exp(-2*(np.abs(hxy2-hxy))), 0.5 )  #Information measure of correlation 2
        return g, features_vector
        
    def compute_features2(self):
        for i,offset in enumerate(self.offsets):
            if self.save_glcms:
                self.glcm_matrices[:,:,i], self.features_matrix[i,:] = self._compute_features_(offset)
            else:
                _,self.features_matrix[i,:] = self._compute_features_(offset)
    
    def compute_features(self):
        pool = Pool()
        ds = [{_HARALICK :self, _OFFSETWA:offset} for offset in self.offsets]
        i = 0
        for res in pool.imap(compute_fake,ds):
            if self.save_glcms:
                self.glcm_matrices[:,:,i], self.features_matrix[i,:] = res
            else:
                self.features_matrix[i,:] = res[1]
            i +=1
        pool.close()
                
    def get_average_features(self):
        return np.nanmean(self.features_matrix, axis=0)
                
    def get_features_at_offset(self, n_offset = -1):
        return Texture_Features(self,n_offset)
    
    def get_features_all_offsets(self):
        """
        All offsets and average features within a list
        """
        return [self.get_features_at_offset(n_offset=offset_n).get_texture_feats_as_dicc() for offset_n in range(-1,self.n_offsets)]
            
        
    def get_glcm_at_offset(self,n_offset = -1, show_glcm = True):
        if -1 < n_offset < self.n_offsets:
            g = self.glcm_matrices[:,:,n_offset]
        else:
            g = np.mean(self.glcm_matrices, axis = 2)
        if show_glcm:
            plt.imshow(g, interpolation='nearest')
        return g

if __name__ == "__main__":
#    image2_test = [np.random.randint(-1024,1024, size = (10,10,10)).astype(dtype = np.int16) for i in range(10) ]
#    #image_test = np.random.randint(-1024,1024, size = (3,3,3))
#    start = time.time()
#    for i, image_test  in enumerate(image2_test):
#        #print i
#        har = Haralick_Features(image_test, bins=16, normalization=True, save_glcm_matrices=False)
#        har.compute_features()
#    print "map",time.time() - start
    
    image46_path = '/tmp/les_46.mhd'
    mask46_path = '/tmp/mask_46.mhd'
    image48_path = '/tmp/les_48.mhd'
    mask48_path = '/tmp/mask_48.mhd'
    image_46 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(image46_path))
    mask_46 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(mask46_path))
    image_48 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(image48_path))
    mask_48 = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(mask48_path))
    image46_feats = Haralick_Features(image_46, mask=mask_46, mask_val=46, normalization=True, bins=256,save_glcm_matrices=True)
    image48_feats = Haralick_Features(image_48, mask=mask_48, mask_val=48, normalization=True, bins=256,save_glcm_matrices=True)
    image46_feats.compute_features()
    image48_feats.compute_features()
"""
import time
image_test = np.random.randint(-1024,1024, size = (300,300,300))
#image_test = np.random.randint(0,high=4, size=(100,100,100))
mask = np.zeros((300,300,300))
mask[50:-50,50:-50,50:-50] = np.random.randint(0,2, size=(200,200,200))
har = Haralick_Features(image_test, bins=256, normalization=True,  mask=mask, save_glcm_matrices=True)
print har.image.shape
start = time.time()
har.compute_features()
print 'process time',time.time() - start
print har.get_average_features()


import time
start = time.time()
image2_test = [np.random.randint(-1024,1024, size = (3,3,3)).astype(dtype = np.int16) for i in range(49152) ]
for image_test  in image2_test:
    har = Haralick_Features(image_test, bins=256, normalization=True, save_glcm_matrices=False)
    har.compute_features()
print 'process time',time.time() - start


ogg = generate_default_offsets()
start = time.time()
image_test = np.random.randint(-1024,1024, size = (3,3,3)).astype(dtype = np.int16)
har = Haralick_Features(image_test, bins=256,offsets = ogg,  normalization=True, save_glcm_matrices=False)
har.compute_features()
print 'process time',time.time() - start

start = time.time()
for image_test  in image2_test:
    har2 = Haralick_Features(image_test, bins=256, normalization=True, save_glcm_matrices=False)
    har.compute_features2()
print "lineal",time.time() - start
"""

    

                

            
 
