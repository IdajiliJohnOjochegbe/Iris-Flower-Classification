# H2O Random Forest and AutoML on Iris Dataset

This project demonstrates how to use H2O's machine learning platform to train a Random Forest model and use AutoML on the Iris dataset. The Iris dataset is a classic dataset in machine learning, and H2O provides an easy-to-use platform for building and evaluating machine learning models.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Running the Project](#running-the-project)
  - [Random Forest](#random-forest)
  - [AutoML](#automl)
- [Conclusion](#conclusion)

## Requirements
- Python 3.6 or above
- H2O.ai

## Installation
To install the necessary packages, use the following command:

```sh
pip install h2o
```
## Dataset
The dataset used in this project is the Iris dataset from the UCI Machine Learning Repository. The dataset can be downloaded using the following code:
```sh
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = requests.get(url)
with open("iris.csv", "w") as file:
    file.write(response.text)
```
## Running the Project

### Random Forest

1. Start the H2O cluster:
```sh
import h2o
h2o.init()
``` 
2. Import the dataset into H2O:
```sh
data = h2o.import_file("iris.csv")
```
3. Split the dataset into training and testing sets:
```sh
train, test = data.split_frame(ratios=[0.8], seed=1234)
```
4. Specify the response column and the features:
```sh
response = "C5"
features = data.columns[:-1]
```
5. Train the Random Forest model:
```sh
from h2o.estimators import H2ORandomForestEstimator
model = H2ORandomForestEstimator(ntrees=50, max_depth=20, seed=1234)
model.train(x=features, y=response, training_frame=train)
```
6. Evaluate the model on the test set:
```sh
performance = model.model_performance(test_data=test)
print(performance)
```
7. Make predictions:
```sh
predictions = model.predict(test)
print(predictions)
```
8.Save the model:
```sh
model_path = h2o.save_model(model=model, path="./", force=True)
print("Model saved to:", model_path)
```
## AutoML
Train using AutoML:
```sh
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=features, y=response, training_frame=train)
```
2. View the AutoML Leaderboard:
```sh
lb = aml.leaderboard
print(lb)
```

3. Evaluate the best model from AutoML on the test set:
```sh
performance = aml.leader.model_performance(test_data=test)
print(performance)
```
4. Make predictions with the best model:
```sh
predictions = aml.leader.predict(test)
print(predictions)
```
5. Save the best model from AutoML:
```sh
model_path = h2o.save_model(model=aml.leader, path="./", force=True)
print("Best model saved to:", model_path)
```
## Conclusion
This project demonstrates how to build and evaluate a Random Forest model and use H2O's AutoML on the Iris dataset. H2O provides a powerful platform for machine learning, making it easy to build, evaluate, and deploy models.
