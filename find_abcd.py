import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

for i in range(0,5):
    for j in range(0,4):
        fn='xy/crop_{}_{}_0000.dat'.format(i,j)
        # f = open(fn, 'r')
        # lines = f.readlines()
        # data = pd.DataFrame(index=range(len(lines)), columns=["x_old", "y_old", "x", "y"])
        data = pd.read_csv(fn, sep='\s+' , names = ["x_old", "y_old", "x", "y"])
        # f.close()
        x_old, y_old = data.x_old.values, data.y_old.values
        x, y = data.x.values, data.y.values
        x_0, y_0 = x[0], y[0]
        x = x - x_0
        y = y - y_0
        model = LinearRegression().fit(np.transpose(np.array([x_old,y_old])), np.transpose(np.array([x,y])))
        print('coef:', model.coef_)
        exit(0)
        
        x0y0_minicrop = crop_coords
            xy_minicrop = substrate_coords
            f = open('xy/'+crop_file_name[5:len(crop_file_name)-3]+'dat', 'w')
            for h in range(0, len(x0y0_minicrop)):
                f.write(str(x0y0_minicrop[h][0]) + "  " + str(x0y0_minicrop[h][1]) + "  " + str(xy_minicrop[h][0]) + "  " + str(xy_minicrop[h][1]) + "\n")
            f.close()