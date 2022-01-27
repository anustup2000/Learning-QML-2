#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


def show_images(data, targets):
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
    for i in range(3):
        for l in range(10):
            axes[i, l].imshow(data[targets==l][i,:,:], cmap=plt.cm.gray_r)
            axes[i, l].set_xticks([])
            axes[i, l].set_yticks([])
            axes[i, l].set_title(f"Label: {targets[targets==l][i]}")



def transform_function(x, y):
    """ Implements f(x,y) = [x, y, z = x^2 + y^2] """
    return np.array([x, y, x**2.0 + y**2.0])

def plot_transform(A,B,X,Y):
   
    # Transform
    A1 = np.array([transform_function(x,y) for 
                x,y in zip(np.ravel(A[:,0]), np.ravel(A[:,1]))])
    B1 = np.array([transform_function(x,y) 
                for x,y in zip(np.ravel(B[:,0]), np.ravel(B[:,1]))])

    # Plot in 3D
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Data in 3D (separable with hyperplane)")
    ax.scatter(A1[:,0], A1[:,1], A1[:,2], marker='o')
    ax.scatter(B1[:,0], B1[:,1], B1[:,2], marker='s',
            c='C3')  # make red
    ax.view_init(5, 60)

    x = np.arange(-1.25, 1.25, 0.25)
    y = np.arange(-1.25, 1.25, 0.26)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    Z[:,:] = 0.5
    ax.plot_surface(X, Y, Z, color='#343A3F')

    # Project data to 2D
    ax2d = fig.add_subplot(122)
    ax2d.set_title("Data in 2D (with hyperplane projection)")
    ax2d.scatter(A1[:,0], A1[:,1], marker='o')
    ax2d.scatter(B1[:,0], B1[:,1], marker='s',
                c='C3')  # make red

    ax2d.add_patch(pl.Circle((0,0), radius=np.sqrt(0.5),
                fill=False, linestyle='solid', linewidth=4.0,
                color='#343A3F'))

