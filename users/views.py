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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import pandas as pd
from django.http import FileResponse, Http404
import matplotlib.pyplot as plt


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


@login_required
def download_price_data(request):
    file_path = os.path.join(
        settings.BASE_DIR, 'excelFiles', 'price_data.xlsx')
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        raise Http404


model = load_model('savedModels/saved_modelnew.h5')
model = Model(model.inputs, model.layers[-2].output)

weather_df = pd.read_excel('excelFiles/weather_data.xlsx')
temperature_df = pd.read_excel('excelFiles/temperature_data.xlsx')
price_df = pd.read_excel('excelFiles/price_data2.xlsx')


@login_required
def profile(request):
    post = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            images = request.FILES.getlist('image')

            # Initialize lists for storing features and predictions

            preprocessed_images = []
            # Process each uploaded image
            for image in images:
                new_image = Image.open(image).resize((20, 20))
                preprocessed_image = np.array(new_image)

                if preprocessed_image.shape == (20, 20, 3):
                    preprocessed_image = preprocessed_image / 255.  # Rescale pixel values
                    preprocessed_images.append(preprocessed_image)

            # Percentage Df Create
            preprocessed_images = np.array(preprocessed_images)

            features = model.predict(preprocessed_images)

            num_clusters = 5  # Specify the number of clusters you want to create
            kmeans = KMeans(n_clusters=num_clusters)
            class_labels = kmeans.fit_predict(features)

            counts = np.bincount(class_labels)

            # Create a DataFrame to store the class information
            class_data = {'Image': images, 'Class': class_labels}
            df = pd.DataFrame(class_data)

            # Create a dictionary to map class numbers to "RipenessValues"
            ripeness_values = {
                0: 'unripe',
                1: 'underripe',
                2: 'ripe',
                3: 'overripe',
                4: 'rotten'
            }

            df['Class'] = df['Class'].map(ripeness_values)

            class_percentages = df['Class'].value_counts(normalize=True) * 100

            percentages_df = pd.DataFrame(class_percentages).reset_index()
            percentages_df.columns = ['Class', 'Percentage']

            order = ['unripe', 'underripe', 'ripe', 'overripe', 'rotten']

            percentages_df['Class'] = pd.Categorical(
                percentages_df['Class'], categories=order, ordered=True)

            percentages_df.sort_values(by='Class', inplace=True)
            percentages_df.reset_index(drop=True, inplace=True)

            fruit_ptgs = ', '.join(percentages_df['Percentage'].astype(str))

            # Day-to-day Prediction

            # Convert the initial ripeness states to a dictionary
            ripeness_states = percentages_df.set_index(
                'Class')['Percentage'].to_dict()

            # Define the order of ripeness states
            ripeness_order = ['unripe', 'underripe',
                              'ripe', 'overripe', 'rotten']

            if set(ripeness_states.keys()) != set(ripeness_order):
                print(
                    "Please make sure the ripeness states in the Excel file match the order in the code.")
                exit(1)

            # Define the optimal temperature for peach growth
            optimal_temperature = 26

            # Create a list to store the ripeness states over time
            ripeness_list = []

            # Add the initial ripeness states to the list
            initial_states = ripeness_states.copy()
            initial_states['Day'] = 0
            ripeness_list.append(initial_states.copy())

            for i in range(1, 93):
                # Get the current temperature from the temperature DataFrame
                current_temperature = temperature_df.loc[i-1, 'Temperature']

                # Calculate adjustment factor based on the temperature difference from the optimum
                adjustment_factor = 1 + \
                    abs(current_temperature - optimal_temperature) / \
                    optimal_temperature

    # Compute the transfer amount from one state to the next based on the adjustment factor
                next_states = ripeness_list[-1].copy()
                for j in range(len(ripeness_order) - 1):
                    transfer_percentage = adjustment_factor / 11.2
                    transfer_amount = next_states[ripeness_order[j]
                                                  ] * transfer_percentage / 100
                    next_states[ripeness_order[j]] -= transfer_amount
                    next_states[ripeness_order[j + 1]] += transfer_amount

    # Ensure the percentages don't exceed 100 or fall below the initial percentages
                for state in ripeness_order:
                    next_states[state] = max(
                        0, min(initial_states[state], next_states[state]))

                next_states['Day'] = i
                ripeness_list.append(next_states.copy())

            ripeness_df = pd.DataFrame(ripeness_list)
            ripeness_df.set_index('Day', inplace=True)

            # TOPSIS
            price_df.sort_values(by='Day', inplace=True)
            ripeness_df.sort_values(by='Day', inplace=True)
            # Make sure 'Day' column exists in weather_data
            weather_df.sort_values(by='Day', inplace=True)

            price_df.reset_index(drop=True, inplace=True)
            ripeness_df.reset_index(drop=True, inplace=True)
            weather_df.reset_index(drop=True, inplace=True)

            merged_data = pd.concat(
                [price_df, ripeness_df, weather_df], axis=1)

            data = merged_data[['ForecastedPrice', 'ripe', 'Weather']].values

            row_sums = data.sum(axis=1)
            normalized_data = data / row_sums[:, np.newaxis]

            weights = [0.55, 0.35, 0.1]

            weighted_normalized_data = normalized_data * weights

            ideal_best = np.max(weighted_normalized_data, axis=0)
            ideal_worst = np.min(weighted_normalized_data, axis=0)

            D_plus = np.sqrt(
                np.sum((weighted_normalized_data - ideal_best) ** 2, axis=1))
            D_minus = np.sqrt(
                np.sum((weighted_normalized_data - ideal_worst) ** 2, axis=1))

            Si = D_minus / (D_minus + D_plus)

            merged_data['TOPSIS_score'] = Si

            best_periods = merged_data.sort_values(
                by='TOPSIS_score', ascending=False)

            # for i, row in best_periods.iterrows():
            # Create and save the Post for each row in best_periods
            #   post = Post.objects.create(
            #      image=image,  # You'll need to supply the image object
            # prediction=row['ripe'],
            # You might need to map this differently
            #     topsis_score=row['TOPSIS_score'],
            #    author=request.user,
            # )
            plt.figure(figsize=(10, 6))
            plt.plot(merged_data['Day'][:16], merged_data['TOPSIS_score']
                     [:16], label='TOPSIS score')  # Select the first 15 rows
            plt.title('Best optimum days to harvest')
            plt.xlabel('Day')
            plt.ylabel('Score')
            # show every day on x-axis for the first 15 days
            plt.xticks(range(1, 16))
            plt.grid(True)
            plt.legend()

            # Save the plot image
            plot_file = os.path.join(settings.MEDIA_ROOT, 'plot.png')
            plt.savefig(plot_file, transparent=True)
            plt.close()  # Close the plot to free up memory

            day0_ripeness = merged_data.loc[0, [
                'unripe', 'underripe', 'ripe', 'overripe', 'rotten']]

            # Plot pie chart
            plt.figure(figsize=(10, 6))
            plt.pie(day0_ripeness, labels=day0_ripeness.index, autopct='%1.1f%%')
            plt.title('Ripeness percentages on Day 0')

            pie_file = os.path.join(settings.MEDIA_ROOT, 'pie.png')
            plt.savefig(pie_file, transparent=True)
            plt.close()  # Close the plot to free up memory

            messages.success(
                request, 'Your posts have been created. The fruits are predicted to be: ' + best_periods.to_string(index=True))
            return redirect('profile')
    else:
        form = ImageUploadForm()

    posts = Post.objects.filter(author=request.user)

    context = {
        'form': form,
        'posts': posts,
    }

    if post is not None:  # Only add post to context if it's not None
        context['post'] = post

    return render(request, 'users/profile.html', context)


@login_required
def past_results(request):
    # Get the user's posts for display on the past results page
    posts = Post.objects.filter(author=request.user).order_by('-date_posted')

    context = {
        'posts': posts,
    }

    return render(request, 'users/past_results.html', context)
