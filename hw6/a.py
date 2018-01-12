import numpy as np
import skimage.io as io
from os import listdir
from os.path import isfile, join
import time
import sys

t0 = time.clock()
def to_display(v):
    a = v - np.min(v)
    a /= np.max(a)
    a = (a*255)
    return a.astype(np.uint8).reshape(600,600,3)

mypath=sys.argv[1]

files = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
files = sorted(files)

for i,pic in enumerate(files):
    if i==0:
        arr = io.imread(pic).reshape(1,-1)
    else:
        arr = np.vstack((arr,io.imread(pic).reshape(1,-1)))
    #print(arr.shape)
arr = arr.astype(np.float64)
avg = np.mean(arr, axis=0)
print(avg.shape)

#io.imsave('avg.jpg',avg.reshape(600,600,3).astype(np.uint8))

for i in range(arr.shape[0]):
    arr[i,:] = arr[i,:] - avg
U, s, VT = np.linalg.svd(arr.transpose(), full_matrices=0) 
print("SVD time : ", time.clock()-t0)

#io.imsave('1.jpg',to_display(U[:,0]))
#io.imsave('2.jpg',to_display(U[:,1]))
#io.imsave('3.jpg',to_display(U[:,2]))
#io.imsave('4.jpg',to_display(U[:,3]))
#io.imsave('10.jpg',to_display(U[:,9]))

eigenface = U[:,0].reshape(1,-1)
#np.save('singularvalues.npy', s)

for i in range(1,4):
    eigenface = np.vstack((eigenface, U[:,i].reshape(1,-1)))
#np.save('eigenface.npy', eigenface)

#s = np.load('singularvalues.npy')
s /= np.sum(s)
print("ratios: ", s[:4])
#eigenface = np.load('eigenface.npy')
eigenface = eigenface[:4,:]
pic=sys.argv[2]
'''
for i,pic in enumerate( [files[32],files[50],files[128],files[65]]):
    if i==0:
        arr_t = io.imread(pic).reshape(1,-1)
    else:
        arr_t = np.vstack((arr_t,io.imread(pic).reshape(1,-1)))
'''
arr_t = io.imread(join(mypath,pic)).reshape(1,-1)
arr_t = arr_t.astype(np.float64) - avg
weights = np.dot(arr_t, eigenface.transpose())

decoded = np.dot(weights, eigenface) + avg
arr_t += avg

io.imsave('reconstruction.jpg',to_display(decoded[0]))

#io.imsave('d1.jpg',to_display(decoded[0]))
#io.imsave('d2.jpg',to_display(decoded[1]))
#io.imsave('d3.jpg',to_display(decoded[2]))
#io.imsave('d4.jpg',to_display(decoded[3]))

#io.imsave('o1.jpg',to_display(arr_t[0]))
#io.imsave('o2.jpg',to_display(arr_t[1]))
#io.imsave('o3.jpg',to_display(arr_t[2]))
#io.imsave('o4.jpg',to_display(arr_t[3]))

