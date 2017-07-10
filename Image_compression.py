'''
The idea is to find the dominant colors and represent with them
instead of the 24-bit colors to a lower-dimensional color space using the cluster assignments.
'''

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import scipy.misc as smp


#Setting initial points
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids

#Finding which point belongs to which cluster and storing it in id
def group_points( x, centroids) :
    m = x.shape[0]
    k = centroids.shape[0]

    id = np.zeros(m)

    for i in range(m) :
        min_dist = 100000000000000 #initialise maximum possible value

        for j in range(k) :
            #find distance of this point from the corresponding centre using distance formula
            dist = np.sum((x[i,:] - centroids[j,:]) ** 2)

            #If calculated distance is lesser than previous, then the point belongs to this cluster
            #Changing the id and distance of this point
            if dist < min_dist :
                min_dist = dist
                id[i] = j

    #Returning the centroid groups of all points
    return id

#Moving Centroids
def move_centroids( x, id, k) :
    m,n = x.shape
    #Centroids will store the position of newly computed centroids
    centroids = np.zeros((k,n))

    for i in range(k) :
        indices = np.where(id == i)
        #Computing mean of centres in the group to reposition the centroids
        centroids[i, :] = (np.sum(x[indices, :], axis=1) / len(indices[0])).ravel()


    return centroids

def run_k_means( x, centroids, num_iterations) :
    m,n = x.shape
    k = centroids.shape[0]
    id = np.zeros(m)


    for i in range(num_iterations) :
        id = group_points( x, centroids)
        centroids = move_centroids( x, id, k)

    return id, centroids

#Loading Data

photo = "bird_small.png"
im = Image.open(photo) #Can be many different formats.
pix = im.load()
height, width = im.size #Get the width and hight of the image for iterating over
#print pix[x,y] #Get the RGBA Value of the a pixel of an image
pixels = []

for i in range(height) :
    list1 = []
    for j in range(width) :
        list = []
        for x in pix[i,j] :
            list.append(x)

        list1.append(pix[i,j])
    pixels.append(list1)

pixels = np.array(pixels)

#print pixels
print  pixels.shape

img = smp.toimage(pixels)       # Create a PIL image
img.show()

A = np.array( pixels, dtype=float)

# normalize value ranges
A = A / 255.0


# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

# randomly initialize the centroids
initial_centroids = init_centroids(X, 10)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 5)

# get the closest centroids one last time
idx = group_points(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

#print X_recovered[0][0]

# Display the image
smp.imshow(X_recovered)

# Save the image
smp.imsave("compressed_"+photo, X_recovered)

# For bigger image
plt.imshow(X_recovered)
plt.show()




