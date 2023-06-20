# Prediction of Online Customers' Intentions

This repo contains the Python Notebook for Predicting the Online Intentions of a Shopper visiting a retail website. The goal of the project is to predict whether the buyer will make a purchase or not based on their behavior which include features like "No. of clicks", "Time spent on the website", "Price of the previous purchase" and so on (represented in the dataset online_shoppers_intentions.csv). The goal of the project is to train ML models that would make predictions using Semisupervised Machine Learning. The approach tries to maximize the performance of the ML models with minimal labelled data.

### Technologies Used

* Python
* Sci-kit Learn
* TensorFlow

### Data Preprocessing Techniques Involved
* One Hot Encoding of Categorial Features
* Normalization of Numerical Features
* Cyclic Transformation of Date
* Oversampling/Undersampling

### Algorithms Applied

#### Self-Training
Self-training is a semi-supervised learning method that uses labeled data to train a traditional classifier(Decision Tree/ SVM/ Random Forest), and then applies the classifier to unlabeled data to generate class probabilities. A confidence threshold is set, and instances whose probabilities exceed the threshold are added to the labeled data pool. The model is then retrained on the expanded labeled dataset and the process is repeated recursively until all remaining unlabeled instances fall below the confidence threshold.

#### Semi-Supervised Ensemble
Semi-supervised ensemble is an approach that uses ensemble classifiers to train on unlabelled data. A voting classifier with three base classifiers (Random Forest, Gaussian NB, and XGBoost) is trained on labeled data, and used to predict on unlabelled data. One-third of the unlabelled data with the most confident predictions are added to the labeled pool and the model is trained again. This process is repeated recursively until all unlabelled data is labeled.

#### Unsupervised Pretraining
Unsupervised pretraining uses an autoencoder to learn features of unlabeled data without labels. The entire data (excluding the test set) is passed through an Autoencoder, consisting of an encoder and decoder, to compress the data. The trained encoder is then applied to the labeled data to extract encoded features. Random Forest is used as a supervised algorithm to predict outcomes based on the encoded data. This approach aims to learn essential features of the data and utilize them for prediction.
