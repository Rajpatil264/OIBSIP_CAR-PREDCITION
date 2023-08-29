# OIBSIP_CAR-PREDCITION
I've built a Car Price Prediction App that uses machine learning to estimate car prices based on features. The app provides an interactive interface for entering car details and displays the predicted price using a trained model. It's a useful tool for getting quick price estimates.

# Car Price Prediction App

This repository contains a Streamlit app that predicts car prices using a trained machine learning model. The app utilizes the `pandas`, `streamlit`, and `matplotlib.pyplot` libraries for data analysis, user interface, and visualization.

## Table of Contents

1. [Importing Libraries](#importing-libraries)
2. [Analysis Section](#analysis-section)
3. [Streamlit App Section](#streamlit-app-section)
4. [Graph Section](#graph-section)
5. [Prediction Section](#prediction-section)

## Importing Libraries
The script begins by importing essential libraries, including `pandas`, `streamlit`, `matplotlib.pyplot`, and machine learning-related modules.

## Analysis Section
Data is loaded from a CSV file, and features are preprocessed for analysis and prediction. Categorical columns are one-hot encoded, and data is split into training and testing sets. A `RandomForestRegressor` model is trained to predict car prices.

## Streamlit App Section
The Streamlit app is introduced with a title and an image. An introduction to the app's purpose is provided, explaining its functionality and how to use it.

## Graph Section
Scatterplots are displayed to visualize the relationship between numeric features and car prices. The graphs help users understand the impact of features on car prices.

## Prediction Section
The R-squared score is displayed to indicate the model's predictive performance. Additionally, a scatter plot illustrates the comparison between actual and predicted car prices.

## User Input and Prediction
Users can input car details, both numeric and categorical, and the app will predict the car's price based on the provided information. The predicted car price is displayed with a formatted output.

Created with expertise by  Rajvardhan 

For inquiries or feedback, please feel free to reach out to me at raj2003patil@gmail.com.

**Note:** Ensure you have the required libraries and a compatible Python environment to run this app successfully.
