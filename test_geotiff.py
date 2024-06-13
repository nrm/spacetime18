from geotiff import GeoTiff
import tifffile
import glob

tiff_files = ['layouts/layout_2021-06-15.tif',
              'layouts/layout_2021-08-16.tif',
              'layouts/layout_2021-10-10.tif',
              'layouts/layout_2022-03-17.tif']

#tiff_files=glob.glob('2_40/*.tif')
#tiff_files=glob.glob('1_20/*.tif')
print(tiff_files)
#exit(0)

for tiff_file in tiff_files:
    try:
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
#        print(geo_tiff.tif_bBox_converted)
    except:
        tf=tifffile.imread(tiff_file)   
        print(tiff_file,' no geotiff data! sizes',tf.shape)