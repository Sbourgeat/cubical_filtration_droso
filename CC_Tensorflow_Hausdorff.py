"""
This script is a modified version of a code originally written by Samuel Bourgeat, a Ph.D student at EPFL in the Jaksic lab.

The script imports several libraries including pyvista, gudhi, numpy, matplotlib, pandas, plotly, and tensorflow.

It also imports the PIL library for image reading and tifffile for working with TIFF image files.

The script then defines an input directory and an output directory for the image files.

It reads a TIFF image file and performs some preprocessing steps such as reshaping the image, increasing contrast, and computing density filtration.

Next, it computes the cubical complex and persistence entropy of the preprocessed image.

It normalizes the image and initializes a tensor cubical layer with tensorflow. It then runs an optimization process to adjust the image based on a persistence loss function and regularization.

After the optimization, the script saves the outputs and plots the results including the optimized image and the persistence diagrams.
"""

# import  libraries

print("Importing packages")

import pyvista as pv
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# import velour

from sklearn.metrics import pairwise_distances

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot



from gtda.images import DensityFiltration


# image reading functions
print("Importing images")

from PIL import Image
import tifffile as TIF

# Tensorflow files

from tqdm import tqdm
from gudhi.tensorflow import LowerStarSimplexTreeLayer, CubicalLayer, RipsLayer
import tensorflow as tf
from tensorflow_graphics.nn.loss.hausdorff_distance import evaluate as hausdorff

# import files

INPUT_DIR = "only_brain/"
OUTPUT_DIR = "Res/"

# Import the tiff files with tifffile

for file in os.listdir(INPUT_DIR):
     
    fly_id = file.split(".")[0]

    img = TIF.imread(file)
    target = TIF.imread(INPUT_DIR + "900_all_female.tif")

# img = Image.open(options.input)
    img = img.max() - img
    image = img / img.max()

    target = target.max() - target
    target = target / target.max()
# plt.imshow(image[120])


# Reshape the image to fit the format needed (only if there is one image to analyse)
    print("Initializing images")

    X = image  # [160]
    X_target = target
# X = X.reshape(1, *X.shape)

# Increase the contrast between voxels in the images
    print("Density filtration")

    DF = DensityFiltration()

    X_df = DF.fit_transform(X)

    X_target_df = DF.fit_transform(X_target)


# Normalize the image to have a max value of 1 and min value 0 for each pixels

    image = X_df / X_df.max()

    target = X_target_df / X_target_df.max()


# Initialize the tensor cubical layer

    print("Initializing tensorflow")

    X = tf.Variable(initial_value=np.array(image, dtype=np.float32), trainable=True)
    X_target = tf.Variable(initial_value=np.array(target, dtype=np.float32), trainable=True) # The target of the registration
    layer = CubicalLayer(homology_dimensions=[1])


# Initialize learning rate

    lr = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=1e-3, decay_steps=10, decay_rate=0.01
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)


# Compute the cubical complex of the target
    dgm_target = layer.call(X_target)[0][0]
    print("Persistence diagram of the target computed")

# Run the optimisation

    ep = 2000


    losses, dgms = [], []
    for epoch in tqdm(range(ep + 1)):
        with tf.GradientTape() as tape:
            dgm = layer.call(X)[0][0]
            # Hausdorff distance between brain A and B 
            print("Computing Hausdorff distance")
            persistence_loss = hausdorff(
                dgm,dgm_target
            )
            # This value defines the max distance we allow to have between the diagonal and the points in the persistence diagrams
            # 0-1 regularization for the pixels
            regularization = 0#tf.math.reduce_sum(tf.math.minimum(tf.abs(X),tf.abs(1-X)))
            loss = persistence_loss + regularization
        gradients = tape.gradient(loss, [X])

        # We also apply a small random noise to the gradient to ensure convergence
        np.random.seed(epoch)
        gradients[0] = gradients[0] + np.random.normal(
            loc=0.0, scale=0.001, size=gradients[0].shape
        )

        optimizer.apply_gradients(zip(gradients, [X]))
        losses.append(loss.numpy())
        dgms.append(dgm)

        #plt.figure()
        #plt.imshow(X.numpy(), cmap='Greys')
        #plt.title('Image at epoch ' + str(epoch))
        #plt.show()


# Save outputs
# losses.to_csv(output_dir + "Losses.csv")
# dgsm.to_csv(output_dir + "dgms.csv")
    # create a folder with fly_id.split(_)[0] as the name and copy the image to that folder.
    #check if the folder exists
    if os.path.exists(OUTPUT_DIR + fly_id.split("_")[0]):
        pass
    else:
        os.mkdir(OUTPUT_DIR + fly_id.split("_")[0])
    
    DIR = OUTPUT_DIR + fly_id.split("_")[0])
    TIF.imwrite(DIR + "registered" + fly_id + ".tiff", X.numpy())

# ploting results
    # save the plots in the new DIR
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # save the plot
    plt.savefig(DIR + "losses" + fly_id + ".png")



    plt.figure()
    plt.scatter(dgms[0][:, 0], dgms[0][:, 1], s=40, marker="D", c="blue")
    for dg in dgms[:-1]:
        plt.scatter(dg[:, 0], dg[:, 1], s=20, marker="D", alpha=0.1)
    plt.scatter(dgms[-1][:, 0], dgms[-1][:, 1], s=40, marker="D", c="red")
    plt.plot([0, 1], [0, 1])
    plt.title("Optimized persistence diagrams")
    #save the plot
    plt.savefig(DIR + "dgms" + fly_id + ".png")
