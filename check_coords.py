import numpy as np

coords_layout = open("coordinates_layout_2021-08-16_.dat", 'r')
coords_newcrop = open("coordinates_newcrop.dat", 'r')

x1_o = {}
x1_c = {}
x2_o = {}
x2_c = {}
x3_o = {}
x3_c = {}
x4_o = {}
x4_c = {}
y1_o = {}
y1_c = {}
y2_o = {}
y2_c = {}
y3_o = {}
y3_c = {}
y4_o = {}
y4_c = {}
lines = coords_layout.readlines()
lines_crop = coords_newcrop.readlines()
for i in range(0,len(lines),5):
    _x1_c, _y1_c = lines[i+1].split()
    lines[i] = lines[i][:-1]
    x1_c[lines[i]] = _x1_c
    y1_c[lines[i]] = _y1_c
    _x2_c, _y2_c = lines[i+2].split()
    x2_c[lines[i]] = _x2_c
    y2_c[lines[i]] = _y2_c
    _x3_c, _y3_c = lines[i+3].split()
    x3_c[lines[i]] = _x3_c
    y3_c[lines[i]] = _y3_c
    _x4_c, _y4_c = lines[i+4].split()
    x4_c[lines[i]] = _x4_c
    y4_c[lines[i]] = _y4_c

for i in range(0,len(lines_crop),5):
    _x1_o, _y1_o = lines_crop[i+1].split()
    lines_crop[i] = lines_crop[i][:-1]
    x1_o[lines_crop[i]] = _x1_o
    y1_o[lines_crop[i]] = _y1_o
    _x2_o, _y2_o = lines_crop[i+2].split()
    x2_o[lines_crop[i]] = _x2_o
    y2_o[lines_crop[i]] = _y2_o
    _x3_o, _y3_o = lines_crop[i+3].split()
    x3_o[lines_crop[i]] = _x3_o
    y3_o[lines_crop[i]] = _y3_o
    _x4_o, _y4_o = lines_crop[i+4].split()
    x4_o[lines_crop[i]] = _x4_o
    y4_o[lines_crop[i]] = _y4_o

o_c = {}
for name in x1_c:
    _o_c = []
    _o_c.append([float(x1_o[name]) - float(x1_c[name]), float(y1_o[name]) - float(y1_c[name])])
    _o_c.append([float(x2_o[name]) - float(x2_c[name]), float(y2_o[name]) - float(y2_c[name])])
    _o_c.append([float(x3_o[name]) - float(x3_c[name]), float(y3_o[name]) - float(y3_c[name])])
    _o_c.append([float(x4_o[name]) - float(x4_c[name]), float(y4_o[name]) - float(y4_c[name])]) 
    o_c[name] = _o_c

mean_o_c = []
for name, _o_c in o_c.items():
    print("______________")
    print(name)
    x1y1 = np.sqrt(_o_c[0][0]**2 + _o_c[0][1]**2)
    print("x1, y1", x1y1)
    x2y2 = np.sqrt(_o_c[1][0]**2 + _o_c[1][1]**2)
    print("x2, y2", x2y2)
    x3y3 = np.sqrt(_o_c[2][0]**2 + _o_c[2][1]**2)
    print("x3, y3", x3y3)
    x4y4 = np.sqrt(_o_c[3][0]**2 + _o_c[3][1]**2)
    print("x4, y4", x4y4)
    mean_o_c.append((x1y1 + x2y2 + x3y3 + x4y4)/4)
    print("mean", mean_o_c[-1])

list_o_c = list(o_c.keys())

print("************")
print("statistics")
print("min: ", min(mean_o_c), list_o_c[np.argmin(mean_o_c)])
print("max: ", max(mean_o_c), list_o_c[np.argmax(mean_o_c)])
print("median: ", np.median(mean_o_c))
print("mean: ", np.mean(mean_o_c))