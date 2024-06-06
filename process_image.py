import matplotlib.pyplot as plt
import numpy as np
import tifffile
import glob
from scipy import signal
import scipy
import cv2

from geotiff import GeoTiff

def cross_correlate_2d(x, h):
    h = np.fft.ifftshift(np.fft.ifftshift(h, axes=0), axes=1)
    return np.fft.ifft2(np.fft.fft2(x) * np.conj(np.fft.fft2(h)))

def ccf_repro_images(crop,lay,ncut):
    mcrop=np.zeros(lay.shape)
    res=np.zeros(lay.shape)
    
    cs=np.min(crop.shape)//ncut
    for i in range(ncut):
        for j in range(ncut):
            mcrop=mcrop*0
            xcyc=np.array(lay.shape)//2-cs//2
            xc=xcyc[0]
            yc=xcyc[1]
            mcrop[xc:xc+cs,yc:yc+cs]=crop[i*cs:(i+1)*cs,j*cs:(j+1)*cs]-np.mean(crop[i*cs:(i+1)*cs,j*cs:(j+1)*cs])

            ccf = np.abs(cross_correlate_2d(mcrop, lay))  # !!!!!!
            
            x, y = np.unravel_index(ccf.argmax(), ccf.shape)
            snr = np.max(ccf) / np.mean(ccf)
            print(i,j,x,y,snr)
            try:
                if snr>7:
                    res[xc*2-x:xc*2-x + cs, yc*2-y:yc*2-y + cs]=crop[i*cs:(i+1)*cs,j*cs:(j+1)*cs]
            except:
                continue

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(crop)
    fig.add_subplot(1, 3, 2)
    plt.imshow(res)
    fig.add_subplot(1, 3, 3)
    plt.imshow(lay)
    plt.show()

    return 0

def calc_for_mults(crop,lay,im,jm,bPlot=False):
    image1 = np.median(crop,axis=2)
    image1[1:, :] = image1[:-1, :]-image1[1:, :]
    image1[0, :]=np.zeros(len(image1[0, :]))
    im2 = lay[::im, ::jm] * 1.0

    im2[1:,:]=im2[:-1,:]-im2[1:,:]

    im2[0,:]=np.zeros(im2.shape[1])

    ix = im2.shape
    im1 = np.zeros((ix[0], ix[1]))
    med1 = np.mean(image1)
    i1mx = image1.shape[0]
    i1my = image1.shape[1]

    im1[ix[0] // 2:ix[0] // 2 + i1mx, ix[1] // 2:ix[1] // 2 + i1my] = image1 * 1.0 - med1
    if bPlot:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(image1 * 1.0 - med1)
        fig.add_subplot(1, 2, 2)
    ccf = np.abs(cross_correlate_2d(im1, im2))  # !!!!!!
    x, y = np.unravel_index(ccf.argmax(), ccf.shape)
    snr = np.max(ccf) / np.mean(ccf)
    if bPlot:
        print(im, jm, np.unravel_index(ccf.argmax(), ccf.shape), ix - x, ix - y)
        print('ccf_max=', np.max(ccf) / np.mean(ccf))
        delta = 10
        im2__ = im2[ix[0] - x - delta:ix[0] - x + i1mx + delta, ix[1] - y - delta:ix[1] - y + i1my + delta] * 1.0 - med
        plt.imshow(im2__)
        plt.show()
        ccf_repro_images(image1 * 1.0 - med1, im2__, 4)
    return snr


def initial_search(crop,lay,mults,fn):
    bPlot=False
    best=0
    optm=(0,0)
    for im in mults:
        for jm in mults:
            snr=calc_for_mults(crop,lay,im,jm)
            if snr > best:
                optm = (im, jm)
                best = snr

    print(fn,' SNR:{:.1f}'.format(best),' mults:',optm)
    calc_for_mults(crop, lay, optm[0], optm[1],bPlot=True)
    return 0

if __name__ == "__main__":
    #substrate = tifffile.imread('layouts/layout_2021-06-15.tif')
    substrate_orig = tifffile.imread('layouts/layout_2021-08-16.tif')
    #substrate = tifffile.imread('layouts/layout_2021-10-10.tif')
    #substrate = tifffile.imread('layouts/layout_2022-03-17.tif')
    
    substrate=np.median(substrate_orig,axis=2)
    
    med=np.median(substrate)
    im2=(substrate-med)*1.0
    ix=im2.shape[0]
    
    im2=np.where(np.abs(im2)>10000,0,im2)
    med=np.mean(im2)
    im2=im2-med
    
    plt.imshow(im2)
    plt.show()
    
    # exit(0)
    
    mults=[5,6,7,8,9,10]
    for i in range(0,5):
        for j in range(0,4):
            fn='1_20/crop_{}_{}_0000.tif'.format(i,j)
            image1 = tifffile.imread(fn)
            initial_search(image1,im2,mults,fn)