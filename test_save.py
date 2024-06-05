import matplotlib.pyplot as plt
import numpy as np
import tifffile
import glob
from scipy import signal
import scipy


def cross_correlate_2d(x, h):
    h = np.fft.ifftshift(np.fft.ifftshift(h, axes=0), axes=1)
    return np.fft.ifft2(np.fft.fft2(x) * np.conj(np.fft.fft2(h)))



def ccf_repro
    
#image2 = tifffile.imread('layouts/layout_2021-06-15.tif')
image2 = tifffile.imread('layouts/layout_2021-08-16.tif')
#image2 = tifffile.imread('layouts/layout_2021-10-10.tif')
#image2 = tifffile.imread('layouts/layout_2022-03-17.tif')

print(image2.shape)

mm=7

layer=1

#ix=image2.shape[0]//mm
#print(ix)

#plt.imshow(image[::5,::5,1])
#plt.show()
#med=np.mean(image2[::mm,::mm,1])
med=np.median(image2[::mm,::mm,layer])
sp2=np.fft.fft2(image2[::mm,::mm,layer]-med)
im2=(image2[::mm,::mm,1]-med)*1.0
ix=im2.shape[0]

im2=np.where(np.abs(im2)>10000,0,im2)
med=np.mean(im2)
im2=im2-med
plt.hist(im2.flatten(),bins=100)
plt.show()
plt.imshow(im2)
plt.show()


for i in range(0,5):
    for j in range(3,4):
        fn='1_20/crop_{}_{}_0000.tif'.format(i,j)
        image1 = tifffile.imread(fn)
        im1=np.zeros((ix,ix))
        med1=np.mean(image1[:,:,layer])
        i1mx=image1.shape[0]
        i1my=image1.shape[1]
        modposx=i*(im2.shape[0]//10)
        modposy=j*(im2.shape[1]//10)

        im1[ix//2:ix//2+i1mx,ix//2:ix//2+i1my]=image1[:,:,layer]*1.0 - med1
              
#        im1[:image1.shape[0],:image1.shape[1]]=image1[:,:,1]*1.0 - med1
        med1=np.mean(im2[modposx:modposx+i1mx,modposy:modposy+i1my]*1.0)
#        im1[ix//2:ix//2+i1mx,ix//2:ix//2+i1my]=im2[modposx:modposx+i1mx,modposy:modposy+i1my]*1.0 - med1
        #plt.figure()
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(image1[:,:,layer]*1.0 - med1)
        fig.add_subplot(1,2,2)
        #plt.imshow(im1)
        #plt.show()
        
        im2_=np.zeros((ix,ix))
        im2_[modposx:modposx+i1mx,modposy:modposy+i1my]=im2[modposx:modposx+i1mx,modposy:modposy+i1my]*1.0 - med
#        im2_[modposy:modposy+i1mx,modposx:modposx+i1my]=im2[modposx:modposx+i1mx,modposy:modposy+i1my]*1.0 - med
#        im1=(im2[modposx:modposx+i1mx,modposy:modposy+i1my]*1.0 - med)*1.0
        
#        im1[:image1.shape[0],:image1.shape[1]]=image2[i*600:i*600+image1.shape[0],j*600:j*600+image1.shape[1],1]*1.0
        
#        sp1=np.fft.fft2(im1)

        #ccf=np.fft.ifft2(sp1*np.conj(sp2))
#        ccf=np.fft.ifft2(np.dot(sp1,np.conj(sp2)))
#        ccf=np.abs(ccf)
#        ccf=np.fft.fftshift(ccf)
        
#        corr = signal.correlate2d(image2[::mm,::mm,1]-med, image2[i*600:i*600+image1.shape[0]*mm:mm,j*600:j*600+image1.shape[1]*mm:mm,1]*1.0 - med, boundary='symm', mode='same')
#        ccf = signal.correlate2d(im1,im2, boundary='symm', mode='same')
#        ccf=scipy.signal.fftconvolve(im1, im2_[::-1, ::-1])
#        ccf=np.abs(ccf)
#        print(im1.shape,im2.shape)
#        ccf=np.abs(cross_correlate_2d(im1,im2_[::-1, ::-1])) # work for simple model
#        ccf=np.abs(cross_correlate_2d(im1,im2[::-1, ::-1])) # work for simple model
        ccf=np.abs(cross_correlate_2d(im1,im2))  #!!!!!!
#        y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
#        print(i,j,np.unravel_index(np.argmax(corr), corr.shape))
        x,y=np.unravel_index(ccf.argmax(), ccf.shape)
        print(i,j,np.unravel_index(ccf.argmax(), ccf.shape),modposx,modposy,ix-x,ix-y)
        im2__=im2[ix-x:ix-x+i1mx,ix-y:ix-y+i1my]*1.0 - med
        plt.imshow(im2__)
        plt.show()
        
#        plt.figure()
        plt.imshow(ccf)
#        plt.imshow(corr)
        plt.show()


exit(0)





image1 = tifffile.imread('1_20/crop_0_0_0000.tif')
#image1 = tifffile.imread('1_20/crop_0_1_0000.tif')
#plt.imshow(image[:,:,1])
#plt.show()
image2 = tifffile.imread('layouts/layout_2021-08-16.tif')
print(image2.shape)

mm=10
ix=image2.shape[0]//mm
print(ix)

#plt.imshow(image[::5,::5,1])
#plt.show()
med=np.mean(image2[::mm,::mm,1])
sp2=np.fft.fft2(image2[::mm,::mm,1]-med)
#print(np.abs(image2))
#plt.imshow(np.abs(sp2))
#plt.show()

im1=np.zeros((ix,ix))
im1[:image1.shape[0],:image1.shape[1]]=image1[:,:,1]*1.0


#plt.imshow(im1)
#plt.show()
med1=np.mean(im1)
sp1=np.fft.fft2(im1-med1)

ccf=np.fft.ifft2(sp1*np.conj(sp2))

plt.imshow(np.fft.fftshift(np.abs(ccf)))
plt.show()
