import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st



# Load the dataset
url = 'https://raw.githubusercontent.com/CLINTONPAU/streamlit-predictions/main/sales.csv'
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
def engineered_features(df):
    # Make a copy to avoid tampering with the original dataset
    df = df.copy()
    
    # Convert the index to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Feature engineering through extraction from the dataset copy
    df['day'] = df.index.day  
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month 
    df['quarter'] = df.index.quarter  
    df['year'] = df.index.year  
    
    return df
# Execute feature engineering on the main dataset
df = engineered_features(df)


# Define features and target variables
features = ['day', 'day_of_week', 'month', 'quarter', 'year']
target = ['Sales']

# Split data into features and targets for training and testing
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Model training and hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train.values.ravel())

# Generate optimal hyperparameters
best_rf_model = grid_search.best_estimator_

# Streamlit App
st.title("Random Forest Model Deployment")

# Sidebar for model input
st.sidebar.header("Model Input")

# Collect user input features
user_input = {}
user_input['day'] = st.sidebar.slider("Select day", float(X_test['day'].min()), float(X_test['day'].max()))
user_input['day_of_week'] = st.sidebar.slider("Select day_of_week", float(X_test['day_of_week'].min()), float(X_test['day_of_week'].max()))
user_input['month'] = st.sidebar.slider("Select month", float(X_test['month'].min()), float(X_test['month'].max()))
user_input['quarter'] = st.sidebar.slider("Select quarter", float(X_test['quarter'].min()), float(X_test['quarter'].max()))
user_input['year'] = st.sidebar.slider("Select year", float(X_test['year'].min()), float(X_test['year'].max()))

# Feature engineering for user input
user_input_df = pd.DataFrame([user_input])
user_input_engineered = engineered_features(user_input_df)

# Make predictions on user input
user_prediction = best_rf_model.predict(user_input_engineered)

# Display user input and prediction
st.write("User Input:")
st.write(user_input_df)
st.write(f"Predicted {target[0]}: {user_prediction[0]}")

# Evaluate the model on the test set
test_predictions = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

# Display evaluation metrics
st.write("\nModel Evaluation on Test Set:")
st.write(f'Mean Squared Error: {mse}')
st.write(f'Mean Absolute Error: {mae}')
st.write(f'R-squared: {r2}')

# Plot the actual vs. predicted values
actual_values = test[target[0]]
predicted_values = test_predictions

# Create a DataFrame for visualization
result_df = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values}, index=test.index)

# Plot the results
st.line_chart(result_df)
