import matplotlib.pyplot as plt
import numpy as np
import tifffile
# import glob
# from scipy import signal
import scipy
#import cv2
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import argparse
import rasterio
import pixel_repair_report
import os
from os import path
from affine import Affine

import csv
import datetime
from datetime import datetime, timezone, date

# from geotiff import GeoTiff

ShowPlot = False
#ShowPlot = True
bSaveLog = False
log_file = None
if bSaveLog:
    log_file=open('log_file.txt','w',buffering=1)

def transform_and_fill_new(F, mult_x=5.,mult_y=9,angle=15):
    tmp=F[0][0]
#     for _ in range(len(F.shape)-1):
#         tmp = tmp[0]
    
    a = 0
    if mult_x:
        a = 1/mult_x
    d = 0
    if mult_y:
        d = 1/mult_y
    
    M = np.array([
        [a, 0.],
        [0., d]
    ])

    phi = angle * np.pi / 180.
    Povorot = np.array([
        [ np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])

    itog = np.matmul(Povorot,M)
    
    a = itog[0][0]
    b = itog[0][1]
    c = itog[1][0]
    d = itog[1][1]
    
    y_range, x_range = F.shape[0:2]

    G = np.zeros(
        (max(min(int(F.shape[1]*c+F.shape[0]*d), F.shape[0]),0),
         max(min(int(F.shape[1]*a+F.shape[0]*b), F.shape[1]),0)
        ,F.shape[2]), dtype=type(tmp)
    )
    
    u_range, v_range = G.shape[:2]
    
    det_A = a*d-b*c
        
    rows, cols = F.shape[0:2]
    
    # Создаем сетку индексов
    u_indices, v_indices = np.meshgrid(np.arange(u_range), np.arange(v_range), indexing='ij')
    
    # Вычисляем новые индексы
    tmp_new_u_indices = np.round(( a * u_indices - c * v_indices) / det_A).astype(int)
    tmp_new_v_indices = np.round((-b * u_indices + d * v_indices) / det_A).astype(int)
    
    # Ограничиваем индексы, чтобы они не выходили за границы массива F
    new_u_indices = np.clip(tmp_new_u_indices, 0, rows - 1)
    new_v_indices = np.clip(tmp_new_v_indices, 0, cols - 1)
    
    # Формируем выходной массив G
#     G[u_indices, v_indices] = F[new_u_indices, new_v_indices, 0]
    G[u_indices, v_indices] = F[new_u_indices, new_v_indices]
        
    G[np.where(tmp_new_u_indices < 0)] = 0
    G[np.where(tmp_new_v_indices < 0)] = 0
    G[np.where(tmp_new_u_indices >= rows)] = 0
    G[np.where(tmp_new_v_indices >= cols)] = 0
    
    return G

def cross_correlate_2d(x, h):
    h = np.fft.ifftshift(np.fft.ifftshift(h, axes=0), axes=1)
    return np.fft.ifft2(np.fft.fft2(x,axes=(0,1)) * np.conj(np.fft.fft2(h,axes=(0,1))),axes=(0,1))

def ccf_repro_images(diff_crop,cropped_substrate,ncut):
    mcrop=np.zeros(cropped_substrate.shape)
    res=np.zeros(cropped_substrate.shape)
    
    cs=np.min(diff_crop.shape)//ncut

    crop_coords = []
    cropped_substrate_coords = []
    for i in range(ncut):
        for j in range(ncut):
            mcrop=mcrop*0
            xcyc=np.array(cropped_substrate.shape)//2-cs//2
            xc=xcyc[0]
            yc=xcyc[1]
            mcrop[xc:xc+cs,yc:yc+cs]=diff_crop[i*cs:(i+1)*cs,j*cs:(j+1)*cs]
            
            ccf = np.abs(cross_correlate_2d(mcrop, cropped_substrate))  # !!!!!!

            x, y = np.unravel_index(ccf.argmax(), ccf.shape)
            snr = np.max(ccf) / np.mean(ccf)
            print(i,j,x,y,snr)
            if snr>15:
                crop_coords.append((i*cs,j*cs))
                cropped_substrate_coords.append((xc*2+cs//2-x,yc*2+cs//2-y))
                try:
                    res[xc*2+cs//2-x:xc*2+cs//2-x + cs, yc*2+cs//2-y:yc*2+cs//2-y + cs]=diff_crop[i*cs:(i+1)*cs,j*cs:(j+1)*cs]
                except:
                    continue

    if ShowPlot:
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(np.abs(diff_crop))
        fig.add_subplot(1, 3, 2)
        plt.imshow(np.abs(res))
        fig.add_subplot(1, 3, 3)
        plt.imshow(np.abs(cropped_substrate))
        plt.show()

    return crop_coords, cropped_substrate_coords, cs


def ccf_repro_images_fullHD(diff_crop, cropped_substrate, ncut,method = 'rgb'):
    mcrop = np.zeros(cropped_substrate.shape,dtype=complex)
    res = np.zeros(cropped_substrate.shape,dtype=complex)

#    cs = np.min(diff_crop.shape) // ncut
    cs = np.array(diff_crop.shape) // ncut

    crop_coords = []
    cropped_substrate_coords = []
    #print('cs:',cs)
    for i in range(ncut):
        for j in range(ncut):
            mcrop = mcrop * 0
            #xcyc = np.array(cropped_substrate.shape) // 2 - cs // 2
            xc = cropped_substrate.shape[0] // 2 - cs[0]//2
            yc = cropped_substrate.shape[1] // 2 - cs[1]//2
            
            try:
                mcrop[xc:xc + cs[0], yc:yc + cs[1],:] = diff_crop[i * cs[0]:(i + 1) * cs[0], j * cs[1]:(j + 1) * cs[1],:]
            except:
                continue

            ccf = np.abs(cross_correlate_2d(mcrop, cropped_substrate))  # !!!!!!
            if method=='rgb':
                ccf = np.sum(ccf[:,:,:3],axis=2)
                x, y = np.unravel_index(ccf.argmax(), ccf.shape)
                snr = np.max(ccf) / np.mean(ccf)
            if method=='ir':
                # ccf = ccf[:,:,0]
                ccf = ccf[:,:,3]
                x, y = np.unravel_index(ccf.argmax(), ccf.shape)
                snr = np.max(ccf) / np.mean(ccf)

            #print(i, j, x, y, snr)
#            if snr > 10:
            if (snr > 8) and \
                np.abs(x + i * cs[0] - cropped_substrate.shape[0] + cs[0]//2)<cs[0] and \
                np.abs(y + j * cs[1] - cropped_substrate.shape[1] + cs[1]//2)<cs[1]:
                    try:
                        if ShowPlot:
                            res[xc * 2 + cs[0] // 2 - x:xc * 2 + cs[0] // 2 - x + cs[0],
                            yc * 2 + cs[1] // 2 - y:yc * 2 + cs[1] // 2 - y + cs[1],:] = diff_crop[i * cs[0]:(i + 1) * cs[0],
                                                                            j * cs[1]:(j + 1) * cs[1],:]
                        crop_coords.append((i * cs[0], j * cs[1]))
                        cropped_substrate_coords.append((xc * 2 + cs[0] // 2 - x, yc * 2 + cs[1] // 2 - y))
                        #print(snr,x + i * cs[0] - cropped_substrate.shape[0] + cs[0]//2, y + j * cs[1] - cropped_substrate.shape[1] + cs[1]//2)
                    except:
                        continue

    #plt.hist(np.abs(diff_crop).flatten(),bins=1000)
    #plt.show()
    if ShowPlot:
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(np.abs(diff_crop),vmax=3)
        fig.add_subplot(1, 3, 2)
        plt.imshow(np.abs(res),vmax=3)
        fig.add_subplot(1, 3, 3)
        plt.imshow(np.abs(cropped_substrate),vmax=np.mean(np.abs(cropped_substrate))+2*np.std(np.abs(cropped_substrate)))
        fig.suptitle('Кроп, миникропы, подложка')
        plt.show()

    return crop_coords, cropped_substrate_coords, cs

def cloud_filter(data0):
    data=data0*1
    data[1:,:]=np.where(np.abs(data[:-1,:])>0,data[1:,:],0)
    data[:,1:]=np.where(np.abs(data[:,:-1])>0,data[:,1:],0)
    data[:-1,:]=np.where(np.abs(data[1:,:])>0,data[:-1,:],0)
    data[:,:-1]=np.where(np.abs(data[:,1:])>0,data[:,:-1],0)
    data[2:,:]=np.where(np.abs(data[:-2,:])>0,data[2:,:],0)
    data[:,2:]=np.where(np.abs(data[:,:-2])>0,data[:,2:],0)
    data[:-2,:]=np.where(np.abs(data[2:,:])>0,data[:-2,:],0)
    data[:,:-2]=np.where(np.abs(data[:,2:])>0,data[:,:-2],0)
    return data

def smooth_(y, box_pts):
    box = np.ones(box_pts)
    box = box/np.sum(box)
    y_smooth = scipy.signal.convolve2d(y, box, mode='same')
    #y_smooth = np.convolve2d(y, box, mode='same')
    return y_smooth

def smooth(x, N, M):
    a = np.cumsum(x,axis=0) 
    a = (a[N:,:] - a[:-N,:]) / float(N)
    a = np.cumsum(a,axis=1) 
    a = (a[:,M:] - a[:,:-M]) / float(M)
    return a

# def affine_transform(matrix, params):
#     rows, cols = matrix.shape
#     # Define the affine transformation matrix
#     M = np.array([
#         [params[0], params[1], params[2]],
#         [params[3], params[4], params[5]]
#     ])

#     # Apply the affine transformation
#     transformed = cv2.warpAffine(matrix, M, (cols, rows))
#     return transformed

# def downscaling(F, multx, multy):
#     params = [1./multx, 0, 0, 0, 1./multy, 0]
#     return affine_transform(F, params) #[:,:int(F.shape[1]/multx)][:int(F.shape[0]/multy)]

def make_derivative(data0,mult_x,mult_y,result_type='x'):
#    if(mult_x*mult_y!=1):
#        data0=smooth(data00,int(mult_x),int(mult_y))
#    else:
#        data0=data00
    if True:
        if int(mult_x)!= mult_x or int(mult_y)!= mult_y:
            indices1=np.round(np.arange(0,data0.shape[1]-mult_y,mult_y)).astype(int)
            indices2=np.round(np.arange(0,data0.shape[0]-mult_x,mult_x)).astype(int)
            data = data0[:,indices1,:][indices2]
        else:
            data = data0[::int(mult_x),::int(mult_y),:] * 1.0

#    data = downscaling(data0,mult_y,mult_x)
    
    data_x = data * 1.0


    data_x[1:, :,:] = data[:-1, :,:]-data[1:, :,:]
#    data_x[1:, :] = np.where(np.abs(data[:-1, :]*data[1:, :])>0,data[:-1, :]-data[1:, :],0)
    data_x[0, :,:]=np.zeros((data.shape[1],data.shape[2]))


    data_y = data * 1.0
    data_y[:, 1:,:] = data[:, :-1,:]-data[:, 1:,:]
#    data_y[:, 1:] = np.where(np.abs(data[:, :-1]*data[:, 1:])>0,data[:, :-1]-data[:, 1:],0)
    data_y[:, 0,:]=np.zeros((data.shape[0],data.shape[2]))


    if result_type == 'x':
        return data_x
    if result_type == 'y':
        return data_y
    if result_type == 'complex':
        #cdata = np.zeros((2**(int(np.log2(data.shape[0]))+1),2**(int(np.log2(data.shape[1]))+1)),dtype=complex)
        #cdata[:data.shape[0],:data.shape[1]]=data_x + 1j*data_y
        cdata=data_x + 1j*data_y
        #cloud filter
        #cdata=cloud_filter(cdata)
        return cdata - np.mean(cdata)
        #return data_x + 1j*data_y
    if result_type == 'mcomplex':
        cdata=data_x + 1j*data_y
        return np.abs(cdata) - np.mean(np.abs(cdata))
    if result_type == 'mcomplex1':
        cdata=data_x + 1j*data_y
        return np.where(np.abs(cdata) - np.median(np.abs(cdata))>np.std(np.abs(cdata)),1,0)
    if result_type == 'none':
        return data - np.mean(data)
    return 'unknown data type'

def calc_maxcoin(coins):
    tolerance=5
    for i in range(coins.shape[0]):
        coin=0
        for j in range(coins.shape[0]):
            if i==j:
                continue
            if abs(coins[i][0]-coins[j][0])+abs(coins[i][1]-coins[j][1])<tolerance:
                coin+=1
        coins[i,3]=coin
    return np.argmax(coins[:,3]),coins


def argmax_ignore_bounds(array, stepx, stepy):
    """
    Find the argmax index in a 2D array ignoring boundary regions.

    Parameters:
    array (np.ndarray): 2D array in which to find the argmax.
    stepx (int): Number of rows to ignore from the start and end.
    stepy (int): Number of columns to ignore from the start and end.

    Returns:
    tuple: Tuple representing the argmax index within the valid region of the array.
    """

    # Slice the array to exclude the boundaries
    valid_region = array[stepx:-stepx, stepy:-stepy]

    # Find the index of the maximum value in the valid region
    local_argmax = np.unravel_index(np.argmax(valid_region), valid_region.shape)

    return local_argmax[0] + stepx, local_argmax[1] + stepy



def calc_for_mults_new(diff_crop,substrate,mult_i,mult_j,deriv_type,return_type='snr',find_rotation=False,method='rgb'):
#    diff_substrate = substrate[::mult_i, ::mult_j] * 1.0

#    diff_substrate[1:,:]=diff_substrate[:-1,:]-diff_substrate[1:,:]

#    diff_substrate[0,:]=np.zeros(diff_substrate.shape[1])
#    print(mult_j,mult_i)
#     print(substrate.shape)
    if find_rotation:
        angles=np.arange(-1.5,1.5,0.2)
    else:
        angles=[0]
    #angles=[0]
    result=(0,0,0,0,0,0,0)
    #diff_substrate=make_derivative(substrate,mult_i,mult_j,deriv_type)
    #plt.imshow(diff_substrate)
    #plt.show()
    for angl in angles:
        if find_rotation:
            transf_sub=transform_and_fill_new(substrate,mult_j,mult_i,angle=angl)
            diff_substrate=make_derivative(transf_sub,1,1,deriv_type)
        else:
            diff_substrate=make_derivative(substrate,mult_i,mult_j,deriv_type)
        #diff_crop=transform_and_fill_new(diff_crop0,1,1,angle=angl)
        # print(diff_substrate.shape)
        # exit(0)

        ix, iy = diff_substrate.shape[:2]
        if deriv_type=='complex':
            im1 = np.zeros(diff_substrate.shape,dtype=complex)
        else:
            im1 = np.zeros(diff_substrate.shape)
        i1mx, i1my = diff_crop.shape[:2]
#        print(im1[ix // 2:ix // 2 + i1mx, iy // 2:iy // 2 + i1my].shape,diff_crop.shape)
        im1[(ix- i1mx) // 2:(ix- i1mx) // 2 + i1mx, (iy-i1my) // 2:(iy-i1my) // 2 + i1my,:] = diff_crop
        ccf = np.abs(cross_correlate_2d(im1, diff_substrate))  # !!!!!!
        coins=np.zeros((ccf.shape[2],4))
        for color in range(ccf.shape[2]):
            x, y = np.unravel_index(ccf[:,:,color].argmax(), ccf.shape[:2])
#            x, y = argmax_ignore_bounds(ccf[:,:,color],i1mx//2,i1my//2)
            snr = np.max(ccf[:,:,color]) / np.mean(ccf[:,:,color])
            coins[color,0] = x
            coins[color,1] = y
            coins[color,2] = snr

         #   print(color,x,y,snr)

    #    ccf = np.sum(ccf,axis=2)
        maxcoin_arg,coins = calc_maxcoin(coins)
        if method=='rgb':
            maxcoin=coins[maxcoin_arg,3]
            ccf = np.sum(ccf[:,:,:3],axis=2)
            #x = int(coins[maxcoin_arg,0])
            #y = int(coins[maxcoin_arg,1])
            x, y = np.unravel_index(ccf.argmax(), ccf.shape)
            maxcoin=0
        if method=='ir':
            maxcoin = 0
            # ccf = ccf[:,:,0]
            ccf = ccf[:,:,3]
    #    plt.imshow(ccf)
    #    plt.show()
            x, y = np.unravel_index(ccf.argmax(), ccf.shape)
            #x, y = argmax_ignore_bounds(ccf,i1my//2,i1mx//2)
        snr = np.max(ccf) / np.mean(ccf)
        #print('sum snr :',snr,mult_i,mult_j,x,y)

#        if result[0]<snr:
#            result=(snr,mult_i,mult_j,x,y,angl,maxcoin)

        if result[6]<=maxcoin:
            if result[0]<snr:
                result=(snr,mult_i,mult_j,x,y,angl,maxcoin)
#                result=(snr,mult_i,mult_j,x,y,angl,maxcoin)


    if return_type=='snr':
        #print((snr,mult_i,mult_j))
        #return (snr,mult_i,mult_j,x,y)
        return result
    else:
        return ccf


def calc_for_mults(diff_crop,substrate,mult_i,mult_j,deriv_type,return_type='snr',tmp=False):
#    diff_substrate = substrate[::mult_i, ::mult_j] * 1.0

#    diff_substrate[1:,:]=diff_substrate[:-1,:]-diff_substrate[1:,:]

#    diff_substrate[0,:]=np.zeros(diff_substrate.shape[1])
#    print(mult_j,mult_i)
#     print(substrate.shape)
    diff_substrate=make_derivative(substrate,mult_i,mult_j,deriv_type)
    # print(diff_substrate.shape)
    # exit(0)

    ix, iy = diff_substrate.shape[:2]
    if deriv_type=='complex':
        im1 = np.zeros(diff_substrate.shape,dtype=complex)
    else:
        im1 = np.zeros(diff_substrate.shape)
    i1mx, i1my = diff_crop.shape[:2]
    # print(im1[ix // 2:ix // 2 + i1mx, iy // 2:iy // 2 + i1my].shape,diff_crop.shape)
    im1[(ix- i1mx) // 2:(ix- i1mx) // 2 + i1mx, (iy-i1my) // 2:(iy-i1my) // 2 + i1my,:] = diff_crop


    ccf = np.abs(cross_correlate_2d(im1, diff_substrate))  # !!!!!!
    #plt.imshow(ccf,vmax=4*np.mean(np.abs(ccf)))
    #plt.show()
    for color in range(ccf.shape[2]):
        x, y = np.unravel_index(ccf[:,:,color].argmax(), ccf.shape[:2])
        snr = np.max(ccf[:,:,color]) / np.mean(ccf[:,:,color])
        print(color,x,y,snr)

    ccf = np.sum(ccf,axis=2)
    #ccf = np.sum(ccf[:,:,:3],axis=2)
    #ccf = ccf[:,:,3]

#    plt.imshow(ccf)
#    plt.show()
    x, y = np.unravel_index(ccf.argmax(), ccf.shape)
    snr = np.max(ccf) / np.mean(ccf)
    print('sum snr :',snr,mult_i,mult_j,x,y)

    if return_type=='snr':
        #print((snr,mult_i,mult_j))
        return (snr,mult_i,mult_j,x,y)
    else:
        return ccf



def initial_search(diff_crop, substrate, mults,deriv_type,find_rotation=False,method='ir'):
    #best_ccf=np.zeros(diff_crop.shape)
    import time

    start = time.time()
    nbest=mults.shape[0]*mults.shape[1]
    #best_results=np.zeros((nbest,7))
    #best_snr=0
    #best_coin=0
    optm=(0,0)
    parlist=[]
    for mult_i in mults[0]:
        for mult_j in mults[1]:
            parlist.append((diff_crop,substrate,mult_i,mult_j,deriv_type,'snr',find_rotation,method))

    snrs=Parallel(n_jobs=16)(delayed(calc_for_mults_new)(*i) for i in parlist)
#    snrs=Parallel(n_jobs=1)(delayed(calc_for_mults)(*i) for i in parlist)

    snrs=np.array(snrs)
    snrs[:,-1]=snrs[:,-1]*1000+snrs[:,0]
    snrs=snrs[np.argsort(snrs[:, -1])][::-1,:]
    #exit(0)

    #print("hello")
    if not find_rotation:
        print(snrs[:7,1:])
    end = time.time()
    print('time:',end-start)
    if bSaveLog:
        log_file.write('initial_search best SNR:{}\n'.format(snrs[0,0]))
    return snrs

def angle_test_fullHD(diff_crop, substrate, angls, mult_i,mult_j, deriv_type):
    #best_ccf=np.zeros(diff_crop.shape)
    import time

    start = time.time()
    best_snr=0
    optm=(0,0)
    parlist=[]
    for angl in angls:
        parlist.append((diff_crop,transform_and_fill_new(substrate, 1,1,angl),mult_i,mult_j,deriv_type))

    snrs=Parallel(n_jobs=1)(delayed(calc_for_mults)(*i) for i in parlist)
    ii=0
    for angl in angls:
        snr=snrs[ii][0]
        print("angl: " + str(angl), "SNR: " + str(snr))
        if snr > best_snr:
            optm = (snrs[ii][1], snrs[ii][2],snrs[ii][3],snrs[ii][4])
            best_snr = snr
        ii += 1
    #print("hello")
    end = time.time()
    print('time:',end-start)
    if bSaveLog:
        log_file.write('angle_test_fullHD best SNR:{}\n'.format(best_snr))
    return optm,best_snr

#def extrapolate_crop(crop,mx,my):
def scale_image(image, multx, multy):
    # Step 1: Perform 2D FFT to transform the image to the frequency domain
    f_transform = np.fft.fft2(image,axes=(0,1))

    # Step 2: Shift the zero-frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform,axes=(0,1))

    # Step 3: Get the dimensions of the original image
    original_shape = image.shape

    # Step 4: Calculate the new shape based on scaling factors
    new_shape = (int(original_shape[0] * multy), int(original_shape[1] * multx),original_shape[2])

    # Step 5: Create a new array with the new shape and insert the frequency domain data
    f_transform_resized = np.zeros(new_shape, dtype=complex)

    # Calculate the min indices for cropping
    min_x = min(original_shape[1], new_shape[1])
    min_y = min(original_shape[0], new_shape[0])

    # Centering the transform
    center_x = new_shape[1] // 2
    center_y = new_shape[0] // 2
    center_x_orig = original_shape[1] // 2
    center_y_orig = original_shape[0] // 2

    f_transform_resized[center_y - min_y // 2:center_y + min_y // 2, center_x - min_x // 2:center_x + min_x // 2] = \
        f_transform_shifted[center_y_orig - min_y // 2:center_y_orig + min_y // 2,
        center_x_orig - min_x // 2:center_x_orig + min_x // 2]

    # Step 6: Shift back the zero-frequency component to the original place
    f_transform_resized_shifted_back = np.fft.ifftshift(f_transform_resized,axes=(0,1))

    # Step 7: Perform the inverse 2D FFT to transform back to the spatial domain
    scaled_image = np.fft.ifft2(f_transform_resized_shifted_back,axes=(0,1))

    return scaled_image


#def make_cropped_substate():


def process_crop(crop, crop_file_name, substrate, mults, refined_mults, method='rgb'):
    # if method == 'ir':
    #     substrate = substrate[:,:,3:] * 1.0
    #     crop = crop[:,:,3:] * 1.0
#    deriv_type='x'
    deriv_type='complex'
#    deriv_type='mcomplex1'
#    deriv_type='mcomplex'
#    deriv_type='none'
#    med_crop = np.sum(crop,axis=2)
#    med_crop = np.median(crop,axis=2)
    med_crop = crop*1.0
#    med_crop = crop[:,:,3]

    #diff_crop = diff_crop - np.mean(diff_crop) * 1.0
    #diff_crop[1:, :] = diff_crop[:-1, :]-diff_crop[1:, :]
    #diff_crop[0, :]=np.zeros(diff_crop.shape[1])
    diff_crop = make_derivative(med_crop,1,1,deriv_type)

    ISmult=2
    med_crop_sm = smooth(med_crop,ISmult,ISmult)
    diff_crop_sm = make_derivative(med_crop_sm,ISmult,ISmult,deriv_type)
    substrate_sm0 = smooth(substrate,ISmult*6,ISmult*6)
    substrate_sm = substrate_sm0[::ISmult,::ISmult]
    #try:
    search_results = initial_search(diff_crop_sm, substrate_sm, mults, deriv_type,find_rotation=False,method=method)
#    optm,best_snr = initial_search(diff_crop_sm, substrate_sm, mults, deriv_type,find_rotation=False,method=method)
#    optm,best_snr = initial_search(diff_crop, substrate_sm, mults, deriv_type)
    #except:
    #    print('Initial search failed')
    #    return [],[],[],0,0
    #exit(0)
#    best_ccf = calc_for_mults(diff_crop, substrate, optm[0],optm[1], deriv_type,return_type='ccf')
    max_snrs=7
    for is_r in range(max_snrs):
        if is_r==max_snrs-1:
            print('snr search failed')
            return [],[],[],0,0

        best_snr=search_results[is_r,0]
        optm=search_results[is_r,1:]
        print(crop_file_name,' SNR:{:.1f}'.format(best_snr),' mults:',optm)
        x, y = int(optm[2]*ISmult),int(optm[3]*ISmult)
        #exit(0)
        
        
        #print('x,y:',x,y)
        #x,y = 1076, 2024
        #print('x,y:',x,y)
        #optm = (9,5,1076, 2024)
        diff_substrate = make_derivative(substrate_sm0,optm[0], optm[1],deriv_type)
        
        ix, iy = diff_substrate.shape[:2]
    #    im1 = np.zeros(diff_substrate.shape)
    #    if deriv_type=='complex':
    #        im1 = np.zeros(diff_substrate.shape,dtype=complex)
        i1mx, i1my = diff_crop.shape[:2]
        print('sub, crop shape:',diff_substrate.shape[:2],diff_crop.shape[:2])
    #    im1[ix // 2 - i1mx//2:ix // 2 - i1mx//2  + i1mx, iy // 2  - i1my//2:iy // 2 + i1my - i1my//2,:] = diff_crop

        #print(optm[0], optm[1], np.unravel_index(best_ccf.argmax(), best_ccf.shape), ix - x, iy - y)
        #print('ccf_max=', np.max(best_ccf) / np.mean(best_ccf))
        delta = 10
        cropped_substrate = diff_substrate[max(ix - x - i1mx//2 - delta,0):min(ix - x - i1mx//2 + i1mx + delta,ix), max(iy - y - i1my//2 - delta,0):min(iy - y - i1my//2 + i1my + delta, iy),:]
        
        if ShowPlot:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(np.sum(np.abs(diff_crop),axis=2))
            fig.add_subplot(1, 2, 2)
            plt.imshow(np.sum(np.abs(cropped_substrate),axis=2))
            fig.suptitle('Метод производной для загрубленной подложки 1')
            plt.show()

        delta = 100
        # cropped_substrateHD = substrate[max(ix - x - delta,0)*optm[0]:min(ix - x + i1mx + delta,ix)*optm[0], max(iy - y - delta,0)*optm[1]:min(iy - y + i1my + delta, iy)*optm[1]]

        kek1 = int(max((ix - x - i1mx//2) * optm[0] - delta, 0))
        kek2 = int(max((iy - y - i1my//2) * optm[1] - delta, 0))
        cropped_substrateHD = substrate[
                            kek1:
                            int(min((ix - x - i1mx//2)* optm[0] + i1mx* optm[0] + delta, ix * optm[0])),
                            kek2:
                            int(min((iy - y - i1my//2) * optm[1] + i1my * optm[1] + delta, iy * optm[1])),:]
        #angls = np.arange(-5,5,0.5)
        #opt_ang = angle_test_fullHD(diff_crop, cropped_substrateHD, angls, optm[0], optm[1], deriv_type)
        #exit(0)
        ixHD, iyHD = cropped_substrateHD.shape[:2]

        new_mults=refined_mults+np.array([optm[0],optm[1]])[:, np.newaxis]
    #    new_mults = [np.arange(optm[0] - 0.6, optm[0] + 0.6, 0.1), np.arange(optm[1] - 0.6, optm[1] + 0.6, 0.1)]
    #    new_mults=[np.arange(optm[0]-0.1,optm[0]+0.1,0.1),np.arange(optm[1]-0.1,optm[1]+0.1,0.1)]

        #new_mults=[[optm[0]],[optm[1]]]
        try:
            #optm, snr_refined
            refined_results = initial_search(diff_crop, cropped_substrateHD, new_mults, deriv_type,find_rotation=True,method=method)
        except:
            print('refined search failed')
            continue
            #print('refined search failed')
            #return [],[],[],0,0
        snr_refined=refined_results[0,0]
        optm=refined_results[0,1:]
        x, y = int(optm[2]),int(optm[3])

        print('optm1:',optm, ' SNR_refined:',snr_refined, ' initial SNR:',best_snr)
        if snr_refined <= best_snr+0.5:
            continue
            #return [],[],[],0,0
        else:
            break
       
    cropped_substrateHD = make_derivative(cropped_substrateHD,1,1,deriv_type)
    cropped_substrate = make_derivative(cropped_substrateHD,optm[0], optm[1],deriv_type)
    
    crop_HD = scale_image(med_crop,optm[1],optm[0])
    crop_HD = make_derivative(crop_HD,1,1,deriv_type)
    i1mxHD, i1myHD = crop_HD.shape[:2]
    
    if ShowPlot:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.abs(diff_crop))
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.abs(cropped_substrate)/np.max(np.abs(cropped_substrate))*100)
        fig.suptitle('Метод производной для загрубленной подложки')
        plt.show()

    if False and ShowPlot:
        #tmp=np.abs(cropped_substrateHD)*1.0
        #plt.hist(tmp.flatten(),bins=1000)
        #plt.show()
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.abs(crop_HD))
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.abs(cropped_substrateHD),vmax=4*np.median(np.abs(cropped_substrateHD)))
        fig.suptitle('Метод производной для подложки full HD')
        plt.show()
    
#    crop_coords, cropped_substrate_coords, cs = ccf_repro_images(diff_crop, cropped_substrate, ncut=4)
    crop_coords, cropped_substrate_coords, cs = ccf_repro_images_fullHD(crop_HD, cropped_substrateHD, ncut=4,method=method)



    
#    substrate_coords= [((tmp_cord[0] + max(ix - x - delta,0))*optm[0], (tmp_cord[1] + max(iy - y - delta,0))*optm[1]) for tmp_cord in cropped_substrate_coords]
    substrate_coords= [((tmp_cord[0] + kek1),
                        (tmp_cord[1] + kek2)) for tmp_cord in cropped_substrate_coords]

#    return crop_coords, substrate_coords, optm,kek1 + (ixHD - x + i1mxHD), kek2 + (iyHD - y + i1myHD)
#    return crop_coords, substrate_coords, optm,kek1 + i1mx//2, kek2 + i1my//2
    
    # return crop_coords, substrate_coords, optm,kek1 + (ixHD - x - i1mxHD), kek2 + (iyHD - y - i1myHD)
    
    # crop_coords = []

    x_sdvig = (iyHD - y - i1myHD)
    y_sdvig = (ixHD - x - i1mxHD)

    # print("do:", (x_sdvig, y_sdvig))
    # print("params", (optm[1], optm[0], optm[4]))
    a, b, c, d, x_tmp_0, y_tmp_0 = get_abcd_from_mults_angl_xy0(optm[1], optm[0], optm[4], 0, 0)
    det_A = a*d-b*c
    y_sdvig, x_sdvig = ((-c/det_A*x_sdvig + a/det_A*y_sdvig),
     ( d/det_A*x_sdvig - b/det_A*y_sdvig))
    # print("posle:", (x_sdvig, y_sdvig))
    
    return crop_coords, substrate_coords, optm,kek1 + y_sdvig, kek2 + x_sdvig
    

def get_abcd_from_mults_angl_xy0(mult_x,mult_y,angle, x0, y0):
#     a = 0
#     if mult_x:
    a = mult_x
#     d = 0
#     if mult_y:
    d = mult_y
    
    M = np.array([
        [ a, 0., 0.],
        [0., d , 0.],
        [0., 0., 1.]
    ])
    
    M_inv = np.array([
        [ 1./a, 0., 0.],
        [0., 1./d , 0.],
        [0., 0., 1.]
    ])
    
    
    X_Y_vec = np.array([
        [x0],
        [y0],
        [1.]
    ])
    
    x0,y0 = (M_inv@X_Y_vec).T[0][:2]
#     print (x0,y0)
    
    M_pr = np.array([
        [1., 0., x0],
        [0., 1., y0],
        [0., 0., 1.]
    ])

    phi = angle * np.pi / 180.
    Povorot = np.array([
        [ np.cos(phi), np.sin(phi), 0.],
        [-np.sin(phi), np.cos(phi), 0.],
        [0.,       0.,              1.]
    ])

    M_obr = np.array([
        [1., 0., -x0],
        [0., 1., -y0],
        [0., 0.,  1.]
    ])
    

#     itog = M_pr@M@M_obr@M_pr@Povorot@M_obr
    itog = M@M_pr@Povorot@M_obr
    
    a  = itog[0][0]
    b  = itog[0][1]
    c  = itog[1][0]
    d  = itog[1][1]
    x0 = itog[0][2]
    y0 = itog[1][2]
    
    return (a, b, c, d, x0, y0)

# вращение вокруг центра с почти допиленным обрезанием
def transform_and_fill_new_2(F, mult_x=5.,mult_y=9,angle=15):
    y_range, x_range = F.shape[0:2]
    
    a, b, c, d, x0, y0 = get_abcd_from_mults_angl_xy0(mult_x, mult_y, angle, x_range//2, y_range//2)
#     a, b, c, d, x0, y0 = get_abcd_from_mults_angl_xy0(mult_x, mult_y, angle, 0, 0)
#     print(a, b, c, d, x0, y0)
    
    det_A = a*d-b*c
        
    tmp=F[0][0]
    G = np.zeros(
#         (F.shape[0],
#         F.shape[1])
        (max(min(int(-min(c/det_A,0)*x_range + max(a/det_A,0)*y_range - (-c/det_A*x0 + a/det_A*y0)), y_range),1),
         max(min(int( max(d/det_A,0)*x_range - min(b/det_A,0)*y_range - ( d/det_A*x0 - b/det_A*y0)), x_range),1))
#         (int(-min(c/det_A,0)*sh1 + max(a/det_A,0)*sh0 - (-c/det_A*x0 +a/det_A*y0)),
#         int(max(d/det_A, 0)*sh1 - min(b/det_A,0)*sh0 - (d/det_A*x0 -b/det_A*y0)))
        , dtype=type(tmp)
#         (max(min(int(F.shape[0]*a/det_A+F.shape[1]*b/det_A + x0), F.shape[0]),0),
#          max(min(int(F.shape[0]*c/det_A+F.shape[1]*d/det_A + y0), F.shape[1]),0)), dtype=type(tmp)
    )
    
    u_range, v_range = G.shape
    
    rows, cols = F.shape[0:2]
    
    # Создаем сетку индексов
    u_indices, v_indices = np.meshgrid(np.arange(u_range), np.arange(v_range), indexing='ij')
    
    # Вычисляем новые индексы
#     tmp_new_u_indices = np.round(( a * (u_indices + x0) + b * (v_indices + y0)) / det_A ).astype(int)
#     tmp_new_v_indices = np.round(( c * (u_indices + x0) + d * (v_indices + y0)) / det_A ).astype(int)
    tmp_new_v_indices = np.round( a * v_indices + b * u_indices + x0).astype(int)
    tmp_new_u_indices = np.round( c * v_indices + d * u_indices + y0).astype(int)
    
    # Ограничиваем индексы, чтобы они не выходили за границы массива F
    new_v_indices = np.clip(tmp_new_v_indices, 0, cols - 1)
    new_u_indices = np.clip(tmp_new_u_indices, 0, rows - 1)
    
    # Формируем выходной массив G
#     G[u_indices, v_indices] = F[new_u_indices, new_v_indices, 0]
    G[u_indices, v_indices] = F[new_u_indices, new_v_indices]
        
    G[np.where(tmp_new_u_indices < 0)] = 0
    G[np.where(tmp_new_v_indices < 0)] = 0
    G[np.where(tmp_new_u_indices >= rows)] = 0
    G[np.where(tmp_new_v_indices >= cols)] = 0
    
    return G

def prepare_substrate(substrate_path):
    if not os.path.exists('1_20_geotiff'):
        os.makedirs('1_20_geotiff')

    if not os.path.exists('pic'):
        os.makedirs('pic')
    substrate_orig = tifffile.imread(substrate_path)
    substrate=substrate_orig
    if ('layout_2021-06-15.tif' in substrate_path):
        substrate=np.where(np.abs(substrate)>2000,0,substrate)
    else:
        substrate=np.where(np.abs(substrate)>12500,0,substrate)
    
    for i in range(4):
        sub_mean=np.mean(substrate_orig[:,:,i])
        sub_std=np.std(substrate_orig[:,:,i])
        sub_max=np.max(substrate_orig[:,:,i])
        sub_max=12500
        maxcolor=1000
        substrate[:,:,i]=substrate[:,:,i]/sub_max*maxcolor
    
    with rasterio.open(substrate_path) as src:
        transform = src.transform
    super_string_partial_name_of_substrate, substrate_suffix = path.splitext(os.path.basename(substrate_path))
    
    file_coord = open('coordinates_' + super_string_partial_name_of_substrate + '_.dat', 'w')
    print('coordinates_' + super_string_partial_name_of_substrate + '_.dat')
            
    print(substrate[-1,-1,:])
    if ShowPlot:
        plt.imshow(substrate[:,:,0])
        plt.show()
    
    mults=np.array([np.arange(5,10.5,0.2),np.arange(5,10.5,0.2)])
    addmult=0.15
    addmult_step=0.05
    refined_mults=[np.arange(-addmult,+addmult,addmult_step),np.arange(-addmult,+addmult,addmult_step)]

    if bSaveLog:
        log_file.write('Layout: {}\n'.format(substrate_path))
    
    return substrate, mults, refined_mults, file_coord, transform, super_string_partial_name_of_substrate

def main_process_func(substrate_path, crop_file_name_0, outputname):
    start_time = datetime.now(timezone.utc)
    
    substrate, mults, refined_mults, file_coord, transform, super_string_partial_name_of_substrate = prepare_substrate(substrate_path)
    result_data = []
    result = new_process_crop(substrate_path, substrate, mults, refined_mults, crop_file_name_0, start_time, file_coord, transform, super_string_partial_name_of_substrate, outputname)
    end_time = datetime.now(timezone.utc)
    result["end"] = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    result_data.append(result)
    
    # with open('coords_' + outputname, 'w', newline='') as f:
    with open(outputname, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, delimiter=';', fieldnames=list(result_data[0].keys()),quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for d in result_data:
            writer.writerow(d)
    
    if bSaveLog:
        log_file.close()
    
    return result

def new_process_crop(substrate_path, substrate, mults, refined_mults, crop_file_name_0, start_time, file_coord, transform, super_string_partial_name_of_substrate, outputname):
    super_result = {}
    # «layout_name» имя подложки,
    # «crop_name» имя снимка,  
    # «ul», «ur», «br», «bl», где лево-верх, право-верх, право-низ, лево-низ координаты, 
    # «crs» координатная система в формате «EPSG:{12345}», 
    # «start» и «end»
    super_result["layout_name"] = os.path.basename(substrate_path)
    super_result["crop_name"] = os.path.basename(crop_file_name_0)
    
    stem, suffix = path.splitext(crop_file_name_0)
    crop_file_name=stem + '_corr' + suffix
    pixel_repair_report.process_image_file(crop_file_name_0,crop_file_name)
    if bSaveLog:
        log_file.write('crop_file_name={}\n'.format(crop_file_name))
    crop = tifffile.imread(crop_file_name)
    method='ir'
    crop_coords, substrate_coords, optm,x_,y_ = process_crop(crop, crop_file_name, substrate, mults,refined_mults,method=method)
    if(abs(x_)+abs(y_)==0):
        method='rgb'
        crop_coords, substrate_coords, optm,x_,y_ = process_crop(crop, crop_file_name, substrate, mults,refined_mults,method=method)
        if(abs(x_)+abs(y_)==0):
            # continue
            # pass
            super_result["ul"] = str(0) + '_' + str(0)
            super_result["ur"] = str(0) + '_' + str(0)
            super_result["br"] = str(0) + '_' + str(0)
            super_result["bl"] = str(0) + '_' + str(0)
            
            super_result["crs"] = 'EPSG:32637'
            
            super_result["start"] = start_time.strftime("%Y-%m-%dT%H:%M:%S")      
            super_result["start_time"] = start_time      
                  
            return super_result
            # TODO
    if(len(crop_coords)<3):
        coef_a = 1
        coef_b = 0
        coef_c = 0
        coef_d = 1
        x_0 = optm[2]+x_
        y_0 = optm[3]+y_
        
        # print(coef_a,coef_b,coef_c,coef_d, x_0, y_0)

        # a, b, c, d, x_tmp_0, y_tmp_0 = get_abcd_from_mults_angl_xy0(optm[1], optm[0], optm[4], optm[3], optm[2])

        # print(a/optm[1], b/optm[1], c/optm[0], d/optm[0], x_tmp_0, y_tmp_0)

        # coef_a = d/optm[0]
        # coef_b = b/optm[0]
        # coef_c = c/optm[1]
        # coef_d = a/optm[1]
        # x_0 = optm[2]+x_+y_tmp_0
        # y_0 = optm[3]+y_+x_tmp_0
    else:
        x_old, y_old = np.array(crop_coords)[:,0], np.array(crop_coords)[:,1]
        x = np.array(substrate_coords)[:,0]
        y = np.array(substrate_coords)[:,1]

        X = np.transpose(np.array([x_old,y_old]))
        Y = np.transpose(np.array([x,y]))
        weights = np.ones(Y.shape[0])
        print('old x_0, y_0:',x_,y_)
        for iIter in range(10):
            model = LinearRegression().fit(X, Y,sample_weight=weights)
            x_0,y_0 = model.intercept_
            Y1 = model.predict(X)

            weights = 1/(1+np.sum(np.abs(Y1-Y),axis=1)**2)
        print('new x_0, y_0:',x_0,y_0)
        
        print(model.intercept_)
        print('coef:', model.coef_)
        coef_a = model.coef_[0][0]
        coef_b = model.coef_[0][1]
        coef_c = model.coef_[1][0]
        coef_d = model.coef_[1][1]

    print ('a:{:.1f}, d:{:.1f}'.format(coef_a, coef_d))
    print(optm)
    if bSaveLog:
        log_file.write('a:{:.6f}, d:{:.6f}, opt:{},{}\n'.format(coef_a, coef_d, optm[0],optm[1]))
        log_file.write('b:{:.6f}, c:{:.6f}\n\n'.format(coef_b, coef_c))

    y1,x1 = (x_0 + (coef_a*0+coef_b*0)*optm[0]),                         (y_0 + (coef_c*0+coef_d*0)*optm[1])
    y2,x2 = (x_0 + (coef_a*(crop.shape[0] - 1)+coef_b*0)*optm[0]),             (y_0 + (coef_c*(crop.shape[0] - 1)+coef_d*0)*optm[1])
    y3,x3 = (x_0 + (coef_a*(crop.shape[0] - 1)+coef_b*(crop.shape[1] - 1))*optm[0]), (y_0 + (coef_c*(crop.shape[0] - 1)+coef_d*(crop.shape[1] - 1))*optm[1])
    y4,x4 = (x_0 + (coef_a*0+coef_b*(crop.shape[1] - 1))*optm[0]) ,             (y_0 + (coef_c*0+coef_d*(crop.shape[1] - 1))*optm[1])
    
    print("******")
    print(x1,y1)
    print(x2,y2)
    print(x3,y3)
    print(x4,y4)
    print("_____")

    sub0= substrate*1.0
    
    if ShowPlot:
        minix = sub0.shape[0]
        maxix = 0
        miniy = sub0.shape[1]
        maxiy = 0

        for ii in range(crop.shape[0]):
            for jj in range(crop.shape[1]):
                x = int(x_0 +(coef_a*ii + coef_b * jj)*optm[0])
                y = int(y_0 +(coef_c*ii + coef_d * jj)*optm[1])
        
                minix = min(minix, x)
                maxix = max(maxix, x)
                miniy = min(miniy, y)
                maxiy = max(maxiy, y)
        
    pixels = [(x1, y1, 'ul'), (x2, y2, 'll'), (x3, y3, 'lr'), (x4, y4, 'ur')]
    
    coords = []
    
    file_coord.write(stem + '\n')
    for pixel in pixels:
        spatial_coordinate = rasterio.transform.xy(transform, pixel[1], pixel[0], offset=pixel[2])
        
        coords.append(spatial_coordinate)
        print("spatial_coordinate:", spatial_coordinate)

        file_coord.write(f"{spatial_coordinate[0]} {spatial_coordinate[1]}\n")
        file_coord.flush()
    
    super_result["ul"] = str(int(coords[0][0]*1000)/1000) + '_' + str(int(coords[0][1]*1000)/1000)
    super_result["ur"] = str(int(coords[3][0]*1000)/1000) + '_' + str(int(coords[3][1]*1000)/1000)
    super_result["br"] = str(int(coords[2][0]*1000)/1000) + '_' + str(int(coords[2][1]*1000)/1000)
    super_result["bl"] = str(int(coords[1][0]*1000)/1000) + '_' + str(int(coords[1][1]*1000)/1000)
    
    super_result["crs"] = 'EPSG:32637'
    
    super_result["start"] = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    super_result["start_time"] = start_time      
    
    optimal_params = np.array([coef_d*optm[1], -coef_b*optm[0], y_0, -coef_c*optm[1], coef_a*optm[0], x_0])*1.0
    
    M = np.array([
        [optimal_params[0], optimal_params[1], optimal_params[2]],
        [optimal_params[3], optimal_params[4], optimal_params[5]]
    ])
    
    new_transform = Affine(M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2])
    
    new_transform = transform*(new_transform)
    
    # src = rasterio.open(crop_file_name_0)
    src = rasterio.open(crop_file_name)
    data = src.read()            
    num_bands = src.count
    profile = src.profile
    profile.update({
        'driver': 'GTiff',
        'crs': 'EPSG:32637',
        'transform': new_transform,
        'count': num_bands,
        'width':crop.shape[1],
        'height':crop.shape[0]
    })
    
    stem_out, suffix_out = path.splitext(outputname)
    
    if not os.path.exists(os.path.join('1_20_geotiff', stem_out)):
        os.makedirs(os.path.join('1_20_geotiff', stem_out))

    tile_path = os.path.join('1_20_geotiff', stem_out, os.path.basename(crop_file_name))
    with rasterio.open(tile_path, 'w', **profile) as dst:
        dst.write(data)
        print(f"Saved tile: {tile_path}")
    
    if ShowPlot:
        fig = plt.figure(figsize=(10, 8))
        fig.add_subplot(1, 2, 1)
        plt.imshow(crop[:,:,0],vmax=1000)
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.abs(sub0),vmax=1000)
        plt.ylim([maxix,minix])
        plt.xlim([miniy,maxiy])
        fig.suptitle('Сравнение кропа и преобразованной подложки')
        fig.savefig('pic/'+super_string_partial_name_of_substrate+stem+'.png'.format(i,j), bbox_inches = 'tight', pad_inches = 0)
        if ShowPlot:
            plt.show()
    return super_result


if __name__ == "__main__":
    # substrate_orig = tifffile.imread('layouts/layout_2021-06-15.tif')   #
    # substrate_orig = tifffile.imread('layouts/layout_2021-08-16.tif')   #original
    # substrate_orig = tifffile.imread('layouts/layout_2021-10-10.tif')
    # substrate_orig = tifffile.imread('layouts/layout_2022-03-17.tif')
    
    start_time = datetime.now(timezone.utc)
    
    result_data = []
    
    parser = argparse.ArgumentParser(description="Get name of substrate")
    parser.add_argument('substrate_path', type=str, help ='Path to the substrate file')
    args = parser.parse_args()
    substrate_path = args.substrate_path
    unix_time = datetime.now(timezone.utc).timestamp()*1000
    
    outputname = str(unix_time) + '.csv'
    print(substrate_path)
    
    substrate, mults, refined_mults, file_coord, transform, super_string_partial_name_of_substrate = prepare_substrate(substrate_path)
    for i in range(0,5):
        for j in range(0,4):
            crop_file_name_0='1_20/crop_{}_{}_0000.tif'.format(i,j)
            result = new_process_crop(substrate_path, substrate, mults, refined_mults, crop_file_name_0, start_time, file_coord, transform, super_string_partial_name_of_substrate, outputname)
            end_time = datetime.now(timezone.utc)
            result["end"] = end_time.strftime("%Y-%m-%dT%H:%M:%S")
            start_time = result["start_time"]
            # print(type(end_time), type(start_time))
            print('time', (end_time - start_time).total_seconds())

            result_data.append(result)
    
    with open('coords_' + outputname, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, delimiter=';', fieldnames=list(result_data[0].keys()),quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for d in result_data:
            writer.writerow(d)
    
    if bSaveLog:
        log_file.close()