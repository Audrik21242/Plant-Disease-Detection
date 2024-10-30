# Plant Disease Detection
This project builds a Convolutional Neural Network (CNN) model to predict diseases affecting common food crops, including corn, millet, and more. By analyzing images of crop leaves, the model identifies specific diseases, aiding in quick and effective agricultural diagnostics.

# Table of Contents
1. Project Overview
2. Dataset
3. Model Architecture
4. Data Preprocessing
5. Training Process
6. Technologies Used
7. Deployment
7. Results

# Project Overview
The goal of this project is to create a deep learning model capable of classifying diseases in plants based on leaf images. Using Convolutional and Pooling layers, the model learns to distinguish features associated with different diseases. A web application interface was created for user interaction, allowing users to upload leaf images for disease classification.

# Dataset
The dataset consists of images of leaves with various diseases, sourced from [mention dataset source, if available, e.g., Kaggle or a research repository]. Due to the dataset's large size, the data is processed in batches and split into training, validation, and test sets.

# Model Architecture
The model is built using a Convolutional Neural Network (CNN) architecture, optimized for image classification tasks:

Layers: Multiple convolutional and pooling layers capture spatial features from the leaf images.
Augmentation: Images are resized and rotated to increase variety and robustness, simulating different orientations and lighting conditions.
Batch Processing: The model was trained in batches due to the large dataset size, allowing efficient use of computational resources.


# Data Preprocessing
Data preprocessing steps include:
Image Resizing and Augmentation: Each image is resized, rotated, and transformed to ensure the model generalizes well to various orientations.
Caching and Batch Loading: To avoid latency during training, caching was implemented. The data is loaded in batches to optimize memory usage.
Data Splitting: The dataset is divided into training, validation, and test sets for effective model evaluation.


# Training Process
The model was trained with TensorFlow 2.10, utilizing the tensorflow-directml for efficient training on compatible hardware. The training process involved:
Batch Training: Due to the dataset size, training was conducted in batches, with a few classes at a time.
Evaluation Metrics: The model was evaluated based on accuracy, loss, and confidence level in predictions.


# Technologies Used
Programming Languages: Python
Libraries:
Data Manipulation: NumPy, Pandas
Model Building: TensorFlow (v2.10) with DirectML
Visualization: Matplotlib
Other Utilities: os (for model saving)
Deployment: Flask


# Deployment
A Flask-based web application serves as the user interface:

Frontend: Users can upload an image of a leaf through a simple HTML form.
Backend: The image is processed, and the model returns the predicted disease class along with the confidence level.
Output: The app displays the predicted disease and its confidence percentage to the user.

# Results
The CNN model achieves promising results in identifying plant diseases based on leaf images. Detailed results and model evaluation metrics can be found in the [Results Notebook / Results section of this repository].

