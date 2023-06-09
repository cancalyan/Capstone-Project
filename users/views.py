from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, ImageUploadForm
from food.models import Post
import requests
import pickle
import os
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import pandas as pd


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(
                request, f'Your account has been created! You are now able to log in, {username}!')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})


# Load the pre-trained VGG16 model without the top classification layer
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input,
              outputs=GlobalAveragePooling2D()(base_model.output))


@login_required
def profile(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = request.FILES.getlist('image')

            # Initialize lists for storing features and predictions
            features = []
            predictions = []

            # Process each uploaded image
            for image in images:
                # Load the image
                img = Image.open(image)
                # Resize the image to the input size expected by VGG16
                img = img.resize((224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                preprocessed_img = tf.keras.applications.vgg16.preprocess_input(
                    img_array)

                # Extract features using the pre-trained VGG16 model
                feature = model.predict(preprocessed_img)
                features.append(feature.flatten())  # Flatten the feature array

            # Convert the list of features to a NumPy array
            features = np.array(features)

            # Perform clustering
            num_clusters = 5  # Specify the number of clusters you want to create
            kmeans = KMeans(n_clusters=num_clusters)
            cluster_labels = kmeans.fit_predict(features)

            # Define the cluster mapping
            cluster_mapping = {
                0: {'class_name': 'unripe', 'numerical_value': 0.2},
                1: {'class_name': 'underripe', 'numerical_value': 0.4},
                2: {'class_name': 'ripe', 'numerical_value': 0.6},
                3: {'class_name': 'overripe', 'numerical_value': 0.8},
                4: {'class_name': 'rotten', 'numerical_value': 1.0}
            }

            # Get the predicted class and ripeness value for each cluster label
            i = 0
            for cluster_label in cluster_labels:
                predicted_class = cluster_mapping[cluster_label]['class_name']
                ripeness_value = cluster_mapping[cluster_label]['numerical_value']
                predictions.append(predicted_class)

                # Save the post
                post = Post(image=images[i], prediction=predicted_class,
                            ripeness=ripeness_value, author=request.user)
                post.save()

                i += 1

            messages.success(
                request, f'Your posts have been created. The fruits are predicted to be: {", ".join(predictions)}.')
            return redirect('profile')
    else:
        form = ImageUploadForm()

    # Get the user's posts for display on the profile page
    posts = Post.objects.filter(author=request.user)

    context = {
        'form': form,
        'posts': posts,
    }

    return render(request, 'users/profile.html', context)


@login_required
def past_results(request):
    # Get the user's posts for display on the past results page
    posts = Post.objects.filter(author=request.user).order_by('-date_posted')

    context = {
        'posts': posts,
    }

    return render(request, 'users/past_results.html', context)
