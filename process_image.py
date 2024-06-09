import matplotlib.pyplot as plt
import numpy as np
import tifffile
import glob
from scipy import signal
import scipy
import cv2
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import argparse
import rasterio


from geotiff import GeoTiff

ShowPlot = False

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
            if snr>10:
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


def ccf_repro_images_fullHD(diff_crop, cropped_substrate, ncut):
    mcrop = np.zeros(cropped_substrate.shape,dtype=complex)
    res = np.zeros(cropped_substrate.shape,dtype=complex)

#    cs = np.min(diff_crop.shape) // ncut
    cs = np.array(diff_crop.shape) // ncut

    crop_coords = []
    cropped_substrate_coords = []
    for i in range(ncut):
        for j in range(ncut):
            mcrop = mcrop * 0
            #xcyc = np.array(cropped_substrate.shape) // 2 - cs // 2
            xc = cropped_substrate.shape[0] // 2 - cs[0]//2
            yc = cropped_substrate.shape[1] // 2 - cs[1]//2
            
            try:
                mcrop[xc:xc + cs[0], yc:yc + cs[1]] = diff_crop[i * cs[0]:(i + 1) * cs[0], j * cs[1]:(j + 1) * cs[1]]
            except:
                continue

            ccf = np.abs(cross_correlate_2d(mcrop, cropped_substrate))  # !!!!!!

            x, y = np.unravel_index(ccf.argmax(), ccf.shape)
            snr = np.max(ccf) / np.mean(ccf)
            print(i, j, x, y, snr)
            if snr > 15:
                try:
                    res[xc * 2 + cs[0] // 2 - x:xc * 2 + cs[0] // 2 - x + cs[0],
                    yc * 2 + cs[1] // 2 - y:yc * 2 + cs[1] // 2 - y + cs[1]] = diff_crop[i * cs[0]:(i + 1) * cs[0],
                                                                      j * cs[1]:(j + 1) * cs[1]]
                    crop_coords.append((i * cs[0], j * cs[1]))
                    cropped_substrate_coords.append((xc * 2 + cs[0] // 2 - x, yc * 2 + cs[1] // 2 - y))

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
        plt.imshow(np.abs(cropped_substrate),vmax=500)
        fig.suptitle('Кроп, миникропы, подложка')
        plt.show()

    return crop_coords, cropped_substrate_coords, cs


def make_derivative(data0,mult_x,mult_y,result_type='x'):
    data = data0[::mult_x,::mult_y] * 1.0
    data_x = data * 1.0
    data_x[1:, :] = data[:-1, :]-data[1:, :]
    data_x[0, :]=np.zeros(data.shape[1])
    data_y = data * 1.0
    data_y[:, 1:] = data[:, :-1]-data[:, 1:]
    data_y[:, 0]=np.zeros(data.shape[0])
    if result_type == 'x':
        return data_x
    if result_type == 'y':
        return data_y
    if result_type == 'complex':
        return data_x + 1j*data_y
    if result_type == 'mcomplex':
        return np.abs(data_x + 1j*data_y)

    return 'unknown data type'


def calc_for_mults(diff_crop,substrate,mult_i,mult_j,deriv_type,return_type='snr'):
#    diff_substrate = substrate[::mult_i, ::mult_j] * 1.0

#    diff_substrate[1:,:]=diff_substrate[:-1,:]-diff_substrate[1:,:]

#    diff_substrate[0,:]=np.zeros(diff_substrate.shape[1])
    #print(mult_j,mult_i)
    diff_substrate=make_derivative(substrate,mult_i,mult_j,deriv_type)
    
    ix, iy = diff_substrate.shape
    if deriv_type=='complex':
        im1 = np.zeros(diff_substrate.shape,dtype=complex)
    else:
        im1 = np.zeros(diff_substrate.shape)
    i1mx, i1my = diff_crop.shape

    im1[ix // 2:ix // 2 + i1mx, iy // 2:iy // 2 + i1my] = diff_crop
    ccf = np.abs(cross_correlate_2d(im1, diff_substrate))  # !!!!!!
    snr = np.max(ccf) / np.mean(ccf)
    if return_type=='snr':
        print((snr,mult_i,mult_j))
        return (snr,mult_i,mult_j)
    else:
        return ccf


def initial_search(diff_crop, substrate, mults,deriv_type):
    #best_ccf=np.zeros(diff_crop.shape)
    import time

    start = time.time()
    best_snr=0
    optm=(0,0)
    parlist=[]
    for mult_i in mults:
        for mult_j in mults:
            parlist.append((diff_crop,substrate,mult_i,mult_j,deriv_type))

    snrs=Parallel(n_jobs=8)(delayed(calc_for_mults)(*i) for i in parlist)
    ii=0
    for mult_i in mults:
        for mult_j in mults:
            snr=snrs[ii][0]
            if snr > best_snr:
                optm = (snrs[ii][1], snrs[ii][2])
                best_snr = snr
            ii += 1
    print("hello")
    end = time.time()
    print('time:',end-start)
    return optm

#def extrapolate_crop(crop,mx,my):
def scale_image(image, multx, multy):
    # Step 1: Perform 2D FFT to transform the image to the frequency domain
    f_transform = np.fft.fft2(image)

    # Step 2: Shift the zero-frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Step 3: Get the dimensions of the original image
    original_shape = image.shape

    # Step 4: Calculate the new shape based on scaling factors
    new_shape = (int(original_shape[0] * multy), int(original_shape[1] * multx))

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
    f_transform_resized_shifted_back = np.fft.ifftshift(f_transform_resized)

    # Step 7: Perform the inverse 2D FFT to transform back to the spatial domain
    scaled_image = np.fft.ifft2(f_transform_resized_shifted_back)

    return scaled_image



def process_crop(crop, crop_file_name, substrate, mults):
    deriv_type='complex'
    med_crop = np.median(crop,axis=2)

    #diff_crop = diff_crop - np.mean(diff_crop) * 1.0
    #diff_crop[1:, :] = diff_crop[:-1, :]-diff_crop[1:, :]
    #diff_crop[0, :]=np.zeros(diff_crop.shape[1])
    diff_crop = make_derivative(med_crop,1,1,deriv_type)

    optm = initial_search(diff_crop, substrate, mults, deriv_type)
    #exit(0)
    best_ccf = calc_for_mults(diff_crop, substrate, optm[0],optm[1], deriv_type,return_type='ccf')

    print(crop_file_name,' SNR:{:.1f}'.format(np.max(best_ccf) / np.mean(best_ccf)),' mults:',optm)
    x, y = np.unravel_index(best_ccf.argmax(), best_ccf.shape)
    
    #diff_substrate = substrate[::optm[0], ::optm[1]] * 1.0
    #diff_substrate[1:,:]=diff_substrate[:-1,:]-diff_substrate[1:,:]
    #diff_substrate[0,:]=np.zeros(diff_substrate.shape[1])
    diff_substrate = make_derivative(substrate,optm[0], optm[1],deriv_type)
    
    ix, iy = diff_substrate.shape
    im1 = np.zeros(diff_substrate.shape)
    if deriv_type=='complex':
        im1 = np.zeros(diff_substrate.shape,dtype=complex)
    i1mx, i1my = diff_crop.shape
    im1[ix // 2:ix // 2 + i1mx, iy // 2:iy // 2 + i1my] = diff_crop
    print(optm[0], optm[1], np.unravel_index(best_ccf.argmax(), best_ccf.shape), ix - x, iy - y)
    print('ccf_max=', np.max(best_ccf) / np.mean(best_ccf))
    delta = 10
    cropped_substrate = diff_substrate[max(ix - x - delta,0):min(ix - x + i1mx + delta,ix), max(iy - y - delta,0):min(iy - y + i1my + delta, iy)]
    delta = 10
    cropped_substrateHD = substrate[max(ix - x - delta,0)*optm[0]:min(ix - x + i1mx + delta,ix)*optm[0], max(iy - y - delta,0)*optm[1]:min(iy - y + i1my + delta, iy)*optm[1]]
    cropped_substrateHD = make_derivative(cropped_substrateHD,1,1,deriv_type)

    crop_HD = scale_image(med_crop,optm[1],optm[0])
    crop_HD = make_derivative(crop_HD,1,1,deriv_type)
    
    if ShowPlot:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.abs(diff_crop))
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.abs(cropped_substrate))
        fig.suptitle('Метод производной для загрубленной подложки')
        plt.show()

    if ShowPlot:
        #tmp=np.abs(cropped_substrateHD)*1.0
        #plt.hist(tmp.flatten(),bins=1000)
        #plt.show()
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.abs(crop_HD))
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.abs(cropped_substrateHD),vmax=500)
        fig.suptitle('Метод производной для подложки full HD')
        plt.show()


#    crop_coords, cropped_substrate_coords, cs = ccf_repro_images(diff_crop, cropped_substrate, ncut=4)
    crop_coords, cropped_substrate_coords, cs = ccf_repro_images_fullHD(crop_HD, cropped_substrateHD, ncut=4)



    
#    substrate_coords= [((tmp_cord[0] + max(ix - x - delta,0))*optm[0], (tmp_cord[1] + max(iy - y - delta,0))*optm[1]) for tmp_cord in cropped_substrate_coords]
    substrate_coords= [((tmp_cord[0] + max(ix - x - delta,0)*optm[0]), (tmp_cord[1] + max(iy - y - delta,0)*optm[1])) for tmp_cord in cropped_substrate_coords]

    return crop_coords, substrate_coords, optm

if __name__ == "__main__":
    # substrate_orig = tifffile.imread('layouts/layout_2021-06-15.tif')   #
    # substrate_orig = tifffile.imread('layouts/layout_2021-08-16.tif')   #original
    # substrate_orig = tifffile.imread('layouts/layout_2021-10-10.tif')
    # substrate_orig = tifffile.imread('layouts/layout_2022-03-17.tif')
    
    parser = argparse.ArgumentParser(description="Get name of substrate")
    parser.add_argument('substrate_path', type=str, help ='Path to the substrate file')
    args = parser.parse_args()
    substrate_orig = tifffile.imread(args.substrate_path)
    
    substrate=np.median(substrate_orig,axis=2)
    
    substrate=(substrate-np.median(substrate))*1.0
    if (args.substrate_path == 'layouts/layout_2021-06-15.tif'):
        substrate=np.where(np.abs(substrate)>3000,0,substrate)
    else:
        substrate=np.where(np.abs(substrate)>10000,0,substrate)
    # substrate=np.where(np.abs(substrate)>10000,10000,substrate)
    
    with rasterio.open(args.substrate_path) as src:
        transform = src.transform
    file_coord = open('coordinates_' + args.substrate_path[8:len(args.substrate_path)-4] + '_.dat', 'w')
    print('coordinates_' + args.substrate_path[8:len(args.substrate_path)-4] + '_.dat')
            
    
    if ShowPlot:
        plt.imshow(substrate)
        plt.show()
    
    mults=[5,6,7,8,9,10]
    # for i in range(0,5):
    #     for j in range(0,4):
    for i in range(0,8):
        for j in range(0,5):
            # crop_file_name='1_20/crop_{}_{}_0000.tif'.format(i,j)
            crop_file_name='2_40/tile_{}_{}.tif'.format(i,j)
            crop = tifffile.imread(crop_file_name)
            crop_coords, substrate_coords, optm = process_crop(crop, crop_file_name, substrate, mults)

            # print(crop_coords)
            # print(substrate_coords)
#            x_0, y_0 = substrate_coords[0][0], substrate_coords[0][1]
            if crop_coords == []:
                continue
            x_old, y_old = np.array(crop_coords)[:,0], np.array(crop_coords)[:,1]
            x = np.array(substrate_coords)[:,0]# - x_0
            y = np.array(substrate_coords)[:,1]# - y_0
#            print(x_0,y_0)
            model = LinearRegression().fit(np.transpose(np.array([x_old,y_old])), np.transpose(np.array([x,y])))
            x_0,y_0 = model.intercept_
            print(model.intercept_)
            print('coef:', model.coef_)
            coef_a = model.coef_[0][0]
            coef_b = model.coef_[0][1]
            coef_c = model.coef_[1][0]
            coef_d = model.coef_[1][1]
            print ('a:{:.1f}, d:{:.1f}'.format(coef_a, coef_d))
            print(optm)
            # print(max(coef_a, coef_d) / min(coef_a, coef_d))
            # print(max(optm) / min(optm))
            # print(max(crop.shape[0:2]) / min(crop.shape[0:2]))
            
            y1,x1 = int(x_0 + (coef_a*0+coef_b*0)*optm[0]),                         int(y_0 + (coef_c*0+coef_d*0)*optm[1])
            y2,x2 = int(x_0 + (coef_a*crop.shape[0]+coef_b*0)*optm[0]),             int(y_0 + (coef_c*crop.shape[0]+coef_d*0)*optm[1])
            y3,x3 = int(x_0 + (coef_a*crop.shape[0]+coef_b*crop.shape[1])*optm[0]), int(y_0 + (coef_c*crop.shape[0]+coef_d*crop.shape[1])*optm[1])
            y4,x4 = int(x_0 + (coef_a*0+coef_b*crop.shape[1])*optm[0]),             int(y_0 + (coef_c*0+coef_d*crop.shape[1])*optm[1])
            
            print("******")
            print(x1,y1)
            print(x2,y2)
            print(x3,y3)
            print(x4,y4)
            print("_____")

            # len_a = int(coef_a*crop.shape[0]+coef_b*crop.shape[1])
            # len_b = int(coef_c*crop.shape[0]+coef_d*crop.shape[1])
            #sub0= make_derivative(substrate,1,1,'complex')
            sub0= substrate*1.0

            minix = sub0.shape[0]
            maxix = 0
            miniy = sub0.shape[1]
            maxiy = 0

            for ii in range(crop.shape[0]):
                for jj in range(crop.shape[1]):
                    x = int(x_0 +(coef_a*ii + coef_b * jj)*optm[0])
                    y = int(y_0 +(coef_c*ii + coef_d * jj)*optm[1])
                    sub0[x, y] = 3000

                    minix = min(minix, x)
                    maxix = max(maxix, x)
                    miniy = min(miniy, y)
                    maxiy = max(maxiy, y)
                
            pixels = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

            file_coord.write('crop_{}_{}_0000\n'.format(i,j))
            for pixel in pixels:
                spatial_coordinate = transform * (pixel[0], pixel[1])
                file_coord.write(f"{spatial_coordinate[0]} {spatial_coordinate[1]}\n")
                file_coord.flush()
            
            fig = plt.figure(figsize=(10, 8))
            fig.add_subplot(1, 2, 1)
            plt.imshow(crop[:,:,0],vmax=1000)
            fig.add_subplot(1, 2, 2)
#            plt.imshow(substrate_orig[x_0:x_0+len_a, y_0:y_0+len_b,0],vmax=1000)
            # plt.imshow(np.abs(sub0[minix:maxix,miniy:maxiy]),vmax=1000)
            # plt.imshow(np.abs(sub0[y1:y3,x1:x3]),vmax=1000)
            plt.imshow(np.abs(sub0),vmax=1000)
            plt.ylim([maxix,minix])
            plt.xlim([miniy,maxiy])
            fig.suptitle('Сравнение кропа и преобразованной подложки')
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            fig.savefig('pic/'+args.substrate_path[8:len(args.substrate_path)-4]+'_crop_{}_{}_0000.png'.format(i,j), bbox_inches = 'tight', pad_inches = 0)
            if ShowPlot:
                plt.show()
