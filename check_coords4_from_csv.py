import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

#coords_layout = open("coordinates_layout_2021-08-16_.dat", 'r')

parser = argparse.ArgumentParser(description="Get name of substrate")
parser.add_argument('substrate_path', type=str, help ='Path to the substrate file')
args = parser.parse_args()
coordinates_newcrop_path = args.substrate_path

#results_csv_path='res/16/combined_result.csv'
results_csv_path='res/23/combined_result.csv'
#results_csv_path='res/17/combined_syntresult.csv'

df=pd.read_csv(results_csv_path,delimiter=';')

print(df)

layouts=df.layout_name.unique()
reference='layout_2021-08-16.tif'

corners=["ul","ur","br","bl"]
for corner in corners:
    x=df[corner].str.split('_',expand=True)
    print(x)
    df[corner+'_y']=x[0].astype(float)
    df[corner+'_x']=x[1].astype(float)
#print(df)
df=df.sort_values(by='crop_name')

lays={}
for layout in layouts:
    lays[layout]=pd.DataFrame(df[df.layout_name==layout])
fig = plt.figure()
ii=1
for layout in layouts:
    print(layout)
    fig.add_subplot(2, 2, ii)
    ii+=1
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    try:
        for corner in corners:
            #print(lays[layout][corner+'_x'].values)
            #print(lays[reference][corner+'_x'].values)
            xx=lays[layout][corner+'_x'].values-lays[reference][corner+'_x'].values
            yy=lays[layout][corner+'_y'].values-lays[reference][corner+'_y'].values
            mask=lays[layout][corner+'_x'].values > 0
            if corner==corners[-1]:
                plt.title(layout+': '+ str(len(xx[mask]))+' crops')
            #print(mask)
            #print(len(xx[mask]))
            print(corner)
            print(np.mean(xx[mask]),np.mean(yy[mask]),np.std(xx[mask]),np.std(yy[mask]))
            print(np.mean(np.abs(xx[mask])),np.mean(np.abs(yy[mask])),np.median(np.abs(xx[mask])),np.median(np.abs(yy[mask])))
            mean_diff_x=np.mean(xx[mask])
            mean_diff_y=np.mean(yy[mask])
            std_diff_x=np.std(xx[mask])
            std_diff_y=np.std(yy[mask])
            plt.scatter(xx[mask],yy[mask],label='{} mx={:.1f} my={:.1f} sx={:.1f} sy={:.1f}'.format(corner,mean_diff_x,mean_diff_y,std_diff_x,std_diff_y))
        plt.legend()            
            
    except:
        continue


plt.show()


if False:
    plt.plot([],[],'o',color='red',label='x1y1')
    plt.plot([],[],'o',color='blue',label='x2y2')
    plt.plot([],[],'o',color='black',label='x3y3')
    plt.plot([],[],'o',color='orange',label='x4y4')
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.legend()
    plt.title(coordinates_layout_path+' ndata='+str(num))
#    plt.show()
    list_o_c = list(o_c.keys())

    print("************")
    print("statistics")
    print("min: ", min(mean_o_c), list_o_c[np.argmin(mean_o_c)])
    print("max: ", max(mean_o_c), list_o_c[np.argmax(mean_o_c)])
    print("median: ", np.median(mean_o_c))
    print("mean: ", np.mean(mean_o_c))
plt.show()
