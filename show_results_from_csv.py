import numpy as np
import matplotlib.pyplot as plt
from geotiff import GeoTiff
import rasterio.transform
import tifffile
import rasterio
import glob
import pandas as pd

tiff_files = ['layouts/layout_2021-06-15.tif',
              'layouts/layout_2021-08-16.tif',
              'layouts/layout_2021-10-10.tif',
              'layouts/layout_2022-03-17.tif']

#tiff_files=glob.glob('2_40/*.tif')
#tiff_files=glob.glob('1_20/*.tif')
print(tiff_files)
#exit(0)

results_csv_path='res/7/combined_result.csv'

df=pd.read_csv(results_csv_path,delimiter=';')

print(df)
#exit(0)
fig = plt.figure()
ii=0

for tiff_file in tiff_files:
    ii+=1
    lname=tiff_file.split('/')[-1]
    d=df[df['layout_name']==lname]
    print(d)
    #exit(0)
    if True:
        geo_tiff = GeoTiff(tiff_file)

        print(geo_tiff)

        # the original crs code
        print(geo_tiff.crs_code)
        # the current crs code
        print(geo_tiff.as_crs)
        # the shape of the tiff
        print(geo_tiff.tif_shape)
        # the bounding box in the as_crs CRS
        print(geo_tiff.tif_bBox)
        # the bounding box as WGS 84
        print(geo_tiff.tif_bBox_wgs_84)
        # the bounding box in the as_crs converted coordinates
        zarr_array = geo_tiff.read()
        array = np.array(zarr_array)
    #    print(geo_tiff.)
        with rasterio.open(tiff_file) as src:
                transform = src.transform
        #'coordinates_layout_2021-06-15_.dat'
        file_coord = open('coordinates_' + tiff_file[8:len(tiff_file)-4] + '_.dat', 'r').read().splitlines()
        substrate_orig = tifffile.imread(tiff_file)
        #sub_mednp.median(substrate_orig))
        substrate=substrate_orig
        #if ('layout_2021-06-15.tif' in tiff_file):
        #        substrate=np.where(np.abs(substrate)>2000,0,substrate)
        #else:
        substrate=np.where(np.abs(substrate)>12500,12500,substrate)

        sub_max=12500
        maxcolor=1000
        for i in range(4):
                substrate[:,:,i]=substrate[:,:,i]/sub_max*maxcolor                

        fig.add_subplot(1, 4, ii)
        plt.imshow(substrate[:,:substrate.shape[1]//2,:3])
#        plt.imshow(substrate[::-1,::-1,:3])
        #plt.show()
        from matplotlib.lines import Line2D
        def process_box(coords_strings):
            coords = []
            for coord_str in coords_strings:
                x, y = map(float, coord_str.split())
                if(abs(x)+abs(y)>0):
                    coords.append((x, y))
            #print(coords)
            if not len(coords):
                return
            pixel_coords = [rasterio.transform.rowcol(transform, x, y) for x, y in coords]
            rows, cols = zip(*pixel_coords)
            for i in range(len(pixel_coords)):
                start = (cols[i], rows[i])
                end = (cols[(i + 1) % len(pixel_coords)], rows[(i + 1) % len(pixel_coords)])
                line = Line2D([start[0], end[0]], [start[1], end[1]], color='orange',lw=2)
                plt.gca().add_line(line)
        
            # Optionally, plot the points themselves
            plt.scatter(cols, rows, color='red')
        
            # Display the plot
        for index, row in d.iterrows():
             box=[row['ul'],row['ur'],row['br'],row['bl']]
             box=[b.replace('_',' ') for b in box]
             process_box(box)
             
#        for il in range(len(file_coord)):
#              if 'crop_' in file_coord[il]:
#                    process_box(file_coord[il+1:il+5])

        #plt.gca().invert_yaxis()  # Invert y axis to match image coordinates
plt.show()

        #rasterio.transform.rowcol(transform, spatial_x, spatial_y)

#        print(geo_tiff.tif_bBox_converted)

