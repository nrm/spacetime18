import matplotlib.pyplot as plt
import numpy as np
import tifffile
import glob
from scipy import signal
import scipy
import cv2
from sklearn.linear_model import LinearRegression

from geotiff import GeoTiff

def cross_correlate_2d(x, h):
    h = np.fft.ifftshift(np.fft.ifftshift(h, axes=0), axes=1)
    return np.fft.ifft2(np.fft.fft2(x) * np.conj(np.fft.fft2(h)))

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
            if snr>7:
                crop_coords.append((i*cs,j*cs))
                cropped_substrate_coords.append((xc*2+cs//2-x,yc*2+cs//2-y))
                try:
                    res[xc*2+cs//2-x:xc*2+cs//2-x + cs, yc*2+cs//2-y:yc*2+cs//2-y + cs]=diff_crop[i*cs:(i+1)*cs,j*cs:(j+1)*cs]
                except:
                    continue

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(diff_crop)
    fig.add_subplot(1, 3, 2)
    plt.imshow(res)
    fig.add_subplot(1, 3, 3)
    plt.imshow(cropped_substrate)
    plt.show()

    return crop_coords, cropped_substrate_coords, cs

def calc_for_mults(diff_crop,substrate,mult_i,mult_j):
    diff_substrate = substrate[::mult_i, ::mult_j] * 1.0

    diff_substrate[1:,:]=diff_substrate[:-1,:]-diff_substrate[1:,:]

    diff_substrate[0,:]=np.zeros(diff_substrate.shape[1])
    
    ix, iy = diff_substrate.shape
    im1 = np.zeros(diff_substrate.shape)
    i1mx, i1my = diff_crop.shape

    im1[ix // 2:ix // 2 + i1mx, iy // 2:iy // 2 + i1my] = diff_crop
    ccf = np.abs(cross_correlate_2d(im1, diff_substrate))  # !!!!!!
    return ccf


def initial_search(diff_crop, substrate, mults):
    best_ccf=np.zeros(diff_crop.shape)
    best_snr=0
    optm=(0,0)
    
    for mult_i in mults:
        for mult_j in mults:
            ccf=calc_for_mults(diff_crop,substrate,mult_i,mult_j)
            snr = np.max(ccf) / np.mean(ccf)
            if snr > best_snr:
                optm = (mult_i, mult_j)
                best_ccf = ccf
                best_snr = snr
    return best_ccf, optm

def process_crop(crop, crop_file_name, substrate, mults):
    diff_crop = np.median(crop,axis=2)
    diff_crop = diff_crop - np.mean(diff_crop) * 1.0
    diff_crop[1:, :] = diff_crop[:-1, :]-diff_crop[1:, :]
    diff_crop[0, :]=np.zeros(diff_crop.shape[1])

    best_ccf, optm = initial_search(diff_crop, substrate, mults)

    print(crop_file_name,' SNR:{:.1f}'.format(np.max(best_ccf) / np.mean(best_ccf)),' mults:',optm)
    x, y = np.unravel_index(best_ccf.argmax(), best_ccf.shape)
    
    diff_substrate = substrate[::optm[0], ::optm[1]] * 1.0
    diff_substrate[1:,:]=diff_substrate[:-1,:]-diff_substrate[1:,:]
    diff_substrate[0,:]=np.zeros(diff_substrate.shape[1])
    
    ix, iy = diff_substrate.shape
    im1 = np.zeros(diff_substrate.shape)
    i1mx, i1my = diff_crop.shape
    im1[ix // 2:ix // 2 + i1mx, iy // 2:iy // 2 + i1my] = diff_crop
    print(optm[0], optm[1], np.unravel_index(best_ccf.argmax(), best_ccf.shape), ix - x, iy - y)
    print('ccf_max=', np.max(best_ccf) / np.mean(best_ccf))
    delta = 10
    cropped_substrate = diff_substrate[max(ix - x - delta,0):min(ix - x + i1mx + delta,ix), max(iy - y - delta,0):min(iy - y + i1my + delta, iy)]
    
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(diff_crop)
    fig.add_subplot(1, 2, 2)
    plt.imshow(cropped_substrate)
    plt.show()

    crop_coords, cropped_substrate_coords, cs = ccf_repro_images(diff_crop, cropped_substrate, ncut=4)
    
    substrate_coords= [((tmp_cord[0] + max(ix - x - delta,0))*optm[0], (tmp_cord[1] + max(iy - y - delta,0))*optm[1]) for tmp_cord in cropped_substrate_coords]
        
    return crop_coords, substrate_coords, optm, crop

if __name__ == "__main__":
    #substrate = tifffile.imread('layouts/layout_2021-06-15.tif')
    substrate_orig = tifffile.imread('layouts/layout_2021-08-16.tif')
    #substrate = tifffile.imread('layouts/layout_2021-10-10.tif')
    #substrate = tifffile.imread('layouts/layout_2022-03-17.tif')
    
    substrate=np.median(substrate_orig,axis=2)
    
    substrate=(substrate-np.mean(substrate))*1.0
    
    substrate=np.where(np.abs(substrate)>10000,0,substrate)
    # substrate=np.where(np.abs(substrate)>10000,10000,substrate)
    
    plt.imshow(substrate)
    plt.show()
    
    mults=[5,6,7,8,9,10]
    for i in range(0,5):
        for j in range(0,4):
            crop_file_name='1_20/crop_{}_{}_0000.tif'.format(i,j)
            crop = tifffile.imread(crop_file_name)
            crop_coords, substrate_coords, optm, crop = process_crop(crop, crop_file_name, substrate, mults)

            # print(crop_coords)
            # print(substrate_coords)
            x_0, y_0 = substrate_coords[0][0], substrate_coords[0][1]
            x_old, y_old = np.array(crop_coords)[:,0], np.array(crop_coords)[:,1]
            x = np.array(substrate_coords)[:,0] - x_0
            y = np.array(substrate_coords)[:,1] - y_0
            model = LinearRegression().fit(np.transpose(np.array([x_old,y_old])), np.transpose(np.array([x,y])))
            # print('coef:', model.coef_)
            coef_a = model.coef_[0][0]
            coef_b = model.coef_[0][1]
            coef_c = model.coef_[1][0]
            coef_d = model.coef_[1][1]
            # print ('a:{:.1f}, d:{:.1f}'.format(coef_a, coef_d))
            # print(max(coef_a, coef_d) / min(coef_a, coef_d))
            # print(max(optm) / min(optm))
            # print(max(crop.shape[0:2]) / min(crop.shape[0:2]))

            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(crop[:,:,0])
            fig.add_subplot(1, 2, 2)
            len_a = int(coef_a*crop.shape[0]+coef_b*crop.shape[1])
            len_b = int(coef_c*crop.shape[0]+coef_d*crop.shape[1])
            plt.imshow(substrate_orig[x_0:x_0+len_a, y_0:y_0+len_b,0])
            plt.show()
            exit(0)