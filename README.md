Shoplifting Detection with Deep Learning
Overview
This project aims to develop a machine learning model to detect shoplifting activities in surveillance videos. The model utilizes deep learning techniques to analyze video frames and classify them as either containing shoplifting behavior or not. The goal is to provide a tool that can assist security personnel in monitoring retail environments and identifying potential security threats.

Dataset
The dataset used for this project is the DCSASS (Dataset for Crime Scene Analysis in Suspicious Shoplifting) dataset. It consists of surveillance videos labeled with different activities, including shoplifting. We filtered the dataset to include only videos labeled as "shoplifting" and "normal" for training the model.

Data Preprocessing
Videos were preprocessed by extracting frames and resizing them to a uniform size (e.g., 224x224 pixels).
Sequences of frames were organized to represent the temporal aspect of the videos.
The dataset was split into training, validation, and testing sets to train and evaluate the model's performance.
Feature Engineering
Relevant features were extracted from the preprocessed videos to capture visual cues indicative of shoplifting behavior.
Deep learning models, such as convolutional neural networks (CNNs), were employed to automatically learn discriminative features from the raw pixel data.
Transfer learning techniques were utilized to leverage pre-trained models for feature extraction.
Model Building
Various architectures were explored, including SlowFast, I3D, and custom-built CNNs, to develop the shoplifting detection model.
Hyperparameters were tuned to optimize model performance, and techniques such as data augmentation and dropout regularization were applied to prevent overfitting.
Evaluation
The model's performance was evaluated using metrics such as accuracy, precision, recall, and F1 score.
A confusion matrix was generated to provide insights into the model's classification performance and identify areas for improvement.
Results
The developed model showed promising results in identifying shoplifting activities from video data.
However, there are limitations and challenges that need to be addressed, such as the need for large annotated datasets, computational resources, and potential biases in the data.
Future Work
Future improvements could involve fine-tuning the model architecture, exploring advanced feature extraction techniques, and incorporating additional contextual information for better detection of shoplifting behaviors.
Dependencies
Python 3.x
TensorFlow
Keras
OpenCV
Matplotlib
NumPy
Pandas
scikit-learn
Usage

Preprocess the dataset: Run theft_detection.ipynb to preprocess the DCSASS dataset.
Train the model: Run train.py to train the shoplifting detection model.
Evaluate the model: Run evaluate.py to evaluate the model's performance.
Make predictions: Use the trained model to make predictions on new video data using theft_detection.ipynb.
