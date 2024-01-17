# Random Forest Model Deployment

This repository contains a Streamlit web application for deploying and interacting with a Random Forest regression model trained on a sales dataset. The model predicts sales based on various features like day, day_of_week, month, quarter, and year.

## Getting Started

To run this application locally, you'll need to have Python installed. You can install the required dependencies using the following:

```bash
pip install pandas scikit-learn streamlit
```

Clone this repository to your local machine:

```bash
git clone https://github.com/CLINTONPAU/streamlit-predictions.git
cd streamlit-predictions
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Visit the provided URL in your web browser to interact with the application.

## Dataset

The dataset used for training and testing the model is loaded from the following URL:

[Sales Dataset](https://raw.githubusercontent.com/CLINTONPAU/streamlit-predictions/main/sales.csv)

## Feature Engineering

The dataset undergoes feature engineering to extract relevant information such as day, day_of_week, month, quarter, and year. This enhances the model's ability to capture temporal patterns in the data.

## Model Training and Hyperparameter Tuning

A Random Forest regression model is trained on the engineered dataset. Hyperparameter tuning is performed using GridSearchCV to find the optimal set of hyperparameters. The chosen metrics for optimization are mean squared error.

## Streamlit App

The Streamlit application allows users to input values for day, day_of_week, month, quarter, and year through a user-friendly sidebar. The application then utilizes the trained Random Forest model to predict sales based on the provided input.

### Model Input Sidebar

Users can adjust the input features using sliders in the sidebar to observe how changes in these features affect the model's predictions.

### Model Evaluation on Test Set

The model's performance is evaluated on a separate test set, and metrics such as mean squared error, mean absolute error, and R-squared are displayed to assess the model's accuracy.

### Actual vs. Predicted Values

A line chart is presented, visualizing the actual sales values against the predicted values on the test set.

Feel free to explore the application and gain insights into the predictions made by the Random Forest model!
