# train_model.py
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    # Load the data
    df = pd.read_csv('calories.csv')
    
    # Preprocess the data
    df = df.drop('User_ID', axis=1)  # Drop User_ID as it's not useful for prediction
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # male=1, female=0
    
    # Separate features and target
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the model and preprocessing objects
    with open('calories_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("\nModel saved successfully as 'calories_model.pkl'")
    print("Scaler saved as 'scaler.pkl'")
    print("Encoder saved as 'encoder.pkl'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    train_and_save_model()