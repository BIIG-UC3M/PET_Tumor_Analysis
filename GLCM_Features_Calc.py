# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:54:41 2017

@author: pmacias
"""
import GLCM_Filter_Histogram_based
import numpy as np

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
    def __init__(self,features_vector):
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
        self.Correlation1 = features_vector[12]
        self.Cluster_Prominance = features_vector[13]
        self.Cluster_Shade = features_vector[14]
        self.Dissimilarity = features_vector[15]
        self.Homogenity_1 = features_vector[16]
        self.Inverse_Different_Moment_Normalized = features_vector[17]
        self.Maximum_Probability = features_vector[18]
        self.Information_Measure_Correlation_1 = features_vector[19]
        self.Information_Measure_Correlation_2 = features_vector[20]
        self.Inverse_Difference_Normalized = features_vector[21]


def glcm_features_extraction(image,mk,dim, distances , angles  , levels , symmetry  , include_angles_average, normed = True,axis_range = None ):
    """
    2D images numpy matrices
    return a feature matrix [distances, angles, num_feats]
    
    output order of features:\n
    0. Energy/ASM
    1. Contrast
    2. Correlation1
    3. Sum of Squares
    4. Homogenity2
    5. Sum average
    6. Sum variance
    7. Sum entropy
    8. Entropy
    9. Difference Variance
    10. Difference entropy
    11. Autocorrelation
    12. Correlation2
    13. Cluster Prominence
    14. Cluster Shade
    15. Dissimilarity
    16. Homogenity 1 / inverse Diference
    17. Inverse Different Moment normalized
    18. Maximum probability 
    19. Information measure of correlation 1
    20. Information measre of correlation 2
    21. Inverse difference normalized
    """
    n_feats = 22
    ##This one returns a multidimension array [levels,levels,distances,angles]
    glcm_matrices=np.zeros([levels,levels,max(distances),angles])
    for dist in distances:
        for ang in range(angles):
            glcm_matrices[:,:,dist-1,ang]=GLCM_Filter_Histogram_based.create_GLCM(image,
                                            dist,dim,ang, bins = levels, axis_range = axis_range, mask=mk,normalization= normed)
 

    num_level,num_level2,diss,angls = glcm_matrices.shape
    features_matrix = np.zeros((diss,angls,n_feats))
    
    I, J = np.ogrid[0:num_level, 0:num_level]
    I = 1+ np.array(range(num_level)).reshape((num_level, 1))
    J = 1+ np.array(range(num_level)).reshape((1, num_level))
    IminusJ = I-J
    power_IminusJ = np.power(IminusJ,2)
    abs_IminusJ = np.abs(IminusJ)
    IplusJ = I+J
 
    for d in range(diss):
        for angle in range(angls):
            g = glcm_matrices[:,:,d,angle]
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
            
            ### compute descriptors ###
            
            features_matrix[d,angle,0] = np.sum(np.power(g,2)) # Angular second moment. Energy = sqrt(ASM)
            features_matrix[d,angle,1] = np.apply_over_axes(np.sum, (g * power_IminusJ), axes=(0, 1))[0, 0] # Contrast
            if std_i>1e-15 and std_j>1e-15: # handle the special case of standard deviations near zero
                features_matrix[d,angle,2] = cov/(std_i*std_j)#v[2] = greycoprops(g,'correlation') # Correlation1
            else:
                features_matrix[d,angle,2] = 1
            features_matrix[d,angle,3] = np.apply_over_axes(np.sum, (g* (diff_i) ** 2),axes=(0, 1))[0, 0]# Sum of squares
            features_matrix[d,angle,4] = np.sum(np.divide(g,1+power_IminusJ)) #Homogenity 2
            features_matrix[d,angle,5] = (gxy*ii).sum() # Sum average
            features_matrix[d,angle,6] = ((ii-features_matrix[d,angle,5])*(ii-features_matrix[d,angle,5])*gxy).sum() # Sum variance
            features_matrix[d,angle,7] = -1*(gxy*np.log(gxy+ np.spacing(1))).sum() # Sum entropy
            features_matrix[d,angle,8] = -1*(g*np.log(g+np.spacing(1))).sum() # Entropy
            features_matrix[d,angle,9] = ((ii_-mx_y)*(ii_-mx_y)*gx_y).sum() # Difference variance
            features_matrix[d,angle,10] = -1*(gx_y*np.log(gx_y++np.spacing(1))).sum() # Difference entropy
            features_matrix[d,angle,11] = np.sum(I*J*g) #Autocorrelation
            features_matrix[d,angle,12] = (features_matrix[d,angle,11] - mu_x*mu_y)/(std_i*std_j) #Correlation2
            features_matrix[d,angle,13] = np.sum(np.power(IplusJ-mu_x-mu_y,4)*g) #Cluster prominence
            features_matrix[d,angle,14] = np.sum(np.power(IplusJ-mu_x-mu_y,3)*g) #Cluster shade
            features_matrix[d,angle,15] = np.sum(abs_IminusJ*g) #Dissimilarity
            features_matrix[d,angle,16] = np.sum(np.divide(g,1+abs_IminusJ)) #Homogenety 1
            features_matrix[d,angle,21] = np.sum(np.divide(g,1+(abs_IminusJ/(levels+0.0))))  #Inverse difference normalized
            features_matrix[d,angle,17] = np.sum(np.divide(g,1+(power_IminusJ/(levels+0.0)))) # Inverse difference moment normalized
            features_matrix[d,angle,18] = np.max(g) #maximum probability 
            p_x = np.sum(g,axis=0); p_y = np.sum(g,axis=1) #marginals
            hx = -np.dot(p_x, np.log(p_x+np.spacing(1)))
            hy = -np.dot(p_y, np.log(p_y+np.spacing(1)))
            #hxy = Entropy
            hxy = features_matrix[d,angle,8]
            p_x = p_x.reshape((num_level,1))
            p_y = p_y.reshape((num_level2,1))
            marginals_multi = p_x*p_y.T
            marginals_log = np.log(marginals_multi+np.spacing(1))
            hxy1 = -np.sum(g*marginals_log)
            hxy2 = -np.sum(np.multiply(marginals_multi,marginals_log))
            features_matrix[d,angle,19] = (hxy - hxy1)/max(hx,hy)  #Information measure of correlation 1
            features_matrix[d,angle,20] = np.power(1 - np.exp(-2*(hxy2-hxy))  ,0.5 )  #Information measure of correlation 2
            
            features_averages_overAngles_matrix = None if include_angles_average == False else np.nanmean(features_matrix,axis=1)
    return features_matrix, features_averages_overAngles_matrix
    

