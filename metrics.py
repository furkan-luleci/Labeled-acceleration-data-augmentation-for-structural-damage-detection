from numpy import std
import numpy as np
import torch
from scipy.stats import ks_2samp
from scipy.spatial.distance import minkowski
from tqdm import tqdm

# Define FID score
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), std(act1)
	mu2, sigma2 = act2.mean(axis=0), std(act2)
	# calculate sum squared difference between means
	diff = (mu1 - mu2)**2.0
	# calculate sqrt of product between cov
	stdmean = sigma1*sigma2
	# calculate score
	fid = diff + sigma1**2 + sigma2**2 - 2.0 * stdmean
	return fid

def compute_similarity(ref_rec,input_rec,weightage=[0.33,0.33,0.33]):
    ## Time domain similarity
    ref_time = np.correlate(ref_rec,ref_rec)    
    inp_time = np.correlate(ref_rec,input_rec)
    diff_time = abs(ref_time-inp_time)

    ## Freq domain similarity
    ref_freq = np.correlate(np.fft.fft(ref_rec),np.fft.fft(ref_rec)) 
    inp_freq = np.correlate(np.fft.fft(ref_rec),np.fft.fft(input_rec))
    diff_freq = abs(ref_freq-inp_freq)

    ## Power similarity
    ref_power = np.sum(ref_rec**2)
    inp_power = np.sum(input_rec**2)
    diff_power = abs(ref_power-inp_power)

    return float(weightage[0]*diff_time+weightage[1]*diff_freq+weightage[2]*diff_power)

# Define the structural similarity index measure
def calculate_ssim(x,y): # x:real, y:fake
    mux, sigmax = x.mean(axis=0), std(x)
    muy, sigmay = y.mean(axis=0), std(y)
    cov = np.cov(x,y)[0][1]
    drange = 1
    c1 = 0.01 * drange
    c2 = 0.03 * drange
    ssim = ((2*mux*muy + c1)*(2*cov + c2))/((mux**2 + muy**2 + c1)*(sigmax**2 + sigmay**2 + c2))
    return ssim

# Define the likeness score
def gpu_LS(real,gen):
    # to torch tensors
    t_gen = torch.unsqueeze(torch.from_numpy(gen),1)
    t_real = torch.unsqueeze(torch.from_numpy(real),1)

    dist_real = torch.cdist(t_real, t_real)  # ICD 1
    dist_real = torch.flatten(torch.tril(dist_real, diagonal=-1))  # remove repeats
    dist_real_ = dist_real[dist_real.nonzero()].flatten()  # remove distance=0 for distances btw same data points

    dist_gen = torch.cdist(t_gen, t_gen)  # ICD 2
    dist_gen = torch.flatten(torch.tril(dist_gen, diagonal=-1))  # remove repeats
    dist_gen_ = dist_gen[dist_gen.nonzero()].flatten()  # remove distance=0 for distances btw same data points

    distbtw = torch.cdist(t_gen, t_real)  # BCD
    distbtw_ = torch.flatten(distbtw)

    D_Sep_1, _ = ks_2samp(dist_real_, distbtw_)
    D_Sep_2, _ = ks_2samp(dist_gen_, distbtw_)

    return (
        1- np.max([D_Sep_1, D_Sep_2]),  # LS=1-DSI
        )

# Define the likeness score (CPU)
#####################  LS CPU ver. ##################{

def dists(data):  # compute ICD
    num = data.shape[0]
    data = data.reshape((num, -1))
    dist = []
    for i in tqdm(range(0,num-1)):
        for j in range(i+1,num):
            dist.append(minkowski(data[i],data[j]))
            
    return np.array(dist)

def dist_btw(a,b):  # compute BCD
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    dist = []
    for i in tqdm(range(a.shape[0])):
        for j in range(b.shape[0]):
            dist.append(minkowski(a[i],b[j]))
            
    return np.array(dist)


def LS(real,gen):  # KS distance btw ICD and BCD
    dist_real = dists(real)  # ICD 1
    dist_gen = dists(gen)  # ICD 2
    distbtw = dist_btw(real, gen)  # BCD
    
    D_Sep_1, _ = ks_2samp(dist_real, distbtw)
    D_Sep_2, _ = ks_2samp(dist_gen, distbtw)

    return 1- np.max([D_Sep_1, D_Sep_2])  # LS=1-DSI
    
    
    
    
