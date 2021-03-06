# Unsupervised Learning
# MovieLens 100k dataset
# Matrix Factorization with side information
# By Juan Carlos Rojas
# Copyright 2021, Texas Tech University - Costa Rica

"""
ORIGINAL VERSION
Do not modify!!
This is a reference
"""

import pandas as pd
import numpy as np
import pickle
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#
# Load and prepare data
#

# Load the training and test data from the Pickle file
with open("movielens_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Get the side info
with open("movielens_side_info.pickle", "rb") as f:
      users_side_info, movies_side_info = pickle.load(f)

# Drop the columns we're not going to use
# For some reason I was not able to make the model converge with this data in it
movies_side_info = movies_side_info.drop(columns="Date")

# Standardize scale of columns in user side-info table
for col in users_side_info.columns:
    if col == "User_ID":
        continue
    users_side_info[col] = (users_side_info[col] - users_side_info[col].mean())/users_side_info[col].std()

# Standardize scale of columns in movies side-info table
for col in movies_side_info.columns:
    if col == "Movie_ID":
        continue
    movies_side_info[col] = (movies_side_info[col] - movies_side_info[col].mean())/movies_side_info[col].std()

# Get the sizes
n_users = max(train_data["User_ID"])
n_movies = max(train_data["Movie_ID"])

# Create a user vector
train_user = train_data.loc[:,["User_ID"]]
test_user = test_data.loc[:,["User_ID"]]

# Merge the user side info
cols = users_side_info.columns         # Keep all side-info columns
#cols = ["User_ID"]                      # Don't keep any side-info columns
train_user = pd.merge(train_user, users_side_info[cols], on="User_ID", how='left')
test_user = pd.merge(test_user, users_side_info[cols], on="User_ID", how='left')

# Create a movies vector
train_movie = train_data.loc[:,["Movie_ID"]]
test_movie = test_data.loc[:,["Movie_ID"]]

# Merge the movie side info
cols = movies_side_info.columns        # Keep all side-info columns
#cols = ["Movie_ID"];                   # Don't keep any side-info columns
train_movie = pd.merge(train_movie, movies_side_info[cols], on="Movie_ID", how='left')
test_movie = pd.merge(test_movie, movies_side_info[cols], on="Movie_ID", how='left')

# Reset the train label indices, to be consistent with the merged tables
train_labels = train_labels.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)

# One-hot encode User_ID
# To avoid issues with missing users, we will cast the columns to a specific list of categories
user_list = range(1, n_users+1)
train_user["User_ID"] = train_user["User_ID"].astype(pd.api.types.CategoricalDtype(user_list))
test_user["User_ID"] = test_user["User_ID"].astype(pd.api.types.CategoricalDtype(user_list))
train_user = pd.get_dummies(train_user, columns=["User_ID"])
test_user = pd.get_dummies(test_user, columns=["User_ID"])

# One-hot encode Movie_ID
# To avoid issues with missing movies, we will cast the columns to a specific list of categories
movie_list = range(1, n_movies+1)
train_movie["Movie_ID"] = train_movie["Movie_ID"].astype(pd.api.types.CategoricalDtype(movie_list))
test_movie["Movie_ID"] = test_movie["Movie_ID"].astype(pd.api.types.CategoricalDtype(movie_list))
train_movie = pd.get_dummies(train_movie, columns=["Movie_ID"])
test_movie = pd.get_dummies(test_movie, columns=["Movie_ID"])

# Get the size of the expanded tables
n_user_cols = train_user.shape[1]
n_movie_cols = train_movie.shape[1]

# Print the user and movie tables
#print(train_user)
#print(train_movie)

#
# Train Model
#

n_latent_factors = 5
n_hidden = 5
print("Factorizing into {} latent factors".format(n_latent_factors))
print("With {} hidden nodes".format(n_hidden))

regularization_scale = 0
print("L2 regulatization scale: {}".format(regularization_scale))

# Start Keras model
# This is not a sequential model, so we will assemble it manually

# Users vector embedding
users_input = tf.keras.layers.Input(shape=[n_user_cols])
users_hidden = tf.keras.layers.Dense(
        n_hidden,
        activation="elu",
        kernel_initializer='he_normal', bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.L2(regularization_scale),
        )(users_input)
users_embedded = tf.keras.layers.Dense(
        n_latent_factors,
        activation="linear",
        kernel_initializer='glorot_normal', bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.L2(regularization_scale),
        )(users_hidden)

# Movies vector embedding
movies_input = tf.keras.layers.Input(shape=[n_movie_cols])
movies_hidden = tf.keras.layers.Dense(
        n_hidden,
        activation="elu",
        kernel_initializer='he_normal', bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.L2(regularization_scale),
        )(movies_input)
movies_embedded = tf.keras.layers.Dense(
        n_latent_factors,
        activation="linear",
        kernel_initializer='glorot_normal', bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.L2(regularization_scale),
        )(movies_hidden)

# Dot product of users & movies
dot_product = tf.keras.layers.dot([users_embedded, movies_embedded], axes=1)

# Construct the model
model = tf.keras.Model([users_input, movies_input], dot_product)

# Show the model summary
#model.summary()

# Define the optimizer

# ADAM optimizer
optimizer = tf.keras.optimizers.Adam()
print("Optimizer: ADAM.  Default learning rate")

# Define model
model.compile(
        optimizer=optimizer,
        loss='mse'
        )

# Train the neural network
n_epochs = 50
batch_size = 32

start_time = time.time()
history = model.fit(
        [train_user,train_movie],
        train_labels, 
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=([test_user,test_movie], test_labels),
        verbose=2,
        )
elapsed_time = time.time() - start_time
print("Execution time: {:.1f}".format(elapsed_time))

# Plot the history of the loss

# Compute the best test result from the history
epoch_hist = [i for i in range(0, n_epochs)]
test_mse_hist = history.history['val_loss']
test_best_val = min(test_mse_hist)
test_best_idx = test_mse_hist.index(test_best_val)
print("Best Test RMSE:      {:.4f} at epoch: {}".format(test_best_val ** 0.5, epoch_hist[test_best_idx]))

# Plot the history of the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Movie Rating MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


