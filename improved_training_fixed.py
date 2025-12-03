# improved_training_fixed_v2.py
import numpy as np
import pandas as pd
import pickle
import warnings
import json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def create_new_features(df):
    """Create engineered features for better prediction"""
    df_eng = df.copy()
    
    # BMI (Body Mass Index)
    df_eng['BMI'] = df_eng['Weight'] / ((df_eng['Height']/100) ** 2)
    
    # Metabolic equivalents (simplified)
    df_eng['MET'] = df_eng['Heart_Rate'] / 100
    
    # Age groups
    df_eng['Age_Group'] = pd.cut(df_eng['Age'], 
                                  bins=[0, 30, 50, 100], 
                                  labels=[0, 1, 2])
    
    # Weight status
    df_eng['Weight_Status'] = pd.cut(df_eng['BMI'],
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=[0, 1, 2, 3])
    
    # Interaction terms
    df_eng['Duration_Weight'] = df_eng['Duration'] * df_eng['Weight']
    df_eng['HeartRate_Duration'] = df_eng['Heart_Rate'] * df_eng['Duration']
    df_eng['Duration_Temp'] = df_eng['Duration'] * df_eng['Body_Temp']
    df_eng['Weight_HeartRate'] = df_eng['Weight'] * df_eng['Heart_Rate']
    df_eng['Intensity_Score'] = (df_eng['Heart_Rate'] * df_eng['Duration']) / 100
    
    return df_eng

def train_improved_model():
    print("="*60)
    print("IMPROVED MODEL TRAINING (V2 - WITH FEATURE SAVING)")
    print("="*60)
    
    # 1. Load data
    df = pd.read_csv('calories.csv')
    print(f"Original data shape: {df.shape}")
    
    # 2. Feature Engineering
    print("\n1. Creating new features...")
    df_engineered = create_new_features(df)
    print(f"New data shape after engineering: {df_engineered.shape}")
    
    # 3. Prepare data
    # Drop irrelevant columns
    cols_to_drop = ['User_ID']
    for col in cols_to_drop:
        if col in df_engineered.columns:
            df_engineered = df_engineered.drop(col, axis=1)
    
    # Encode categorical variables
    categorical_cols = df_engineered.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col != 'Calories':  # Don't encode target
            le = LabelEncoder()
            df_engineered[col] = le.fit_transform(df_engineered[col].astype(str))
            label_encoders[col] = le
    
    # Save feature names BEFORE splitting
    feature_names = [col for col in df_engineered.columns if col != 'Calories']
    print(f"\n2. Features used ({len(feature_names)}):")
    for i, feat in enumerate(feature_names, 1):
        print(f"   {i:2d}. {feat}")
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save feature names as JSON for easy reading
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=4)
    
    # Save encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Separate features and target
    X = df_engineered.drop('Calories', axis=1)
    y = df_engineered['Calories']
    
    # Ensure consistent column order
    X = X[feature_names]
    
    print(f"\n3. Target variable:")
    print(f"   Min: {y.min():.1f}")
    print(f"   Max: {y.max():.1f}")
    print(f"   Mean: {y.mean():.1f}")
    print(f"   Std: {y.std():.1f}")
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\n4. Data split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler with feature information
    with open('scaler.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }, f)
    
    # 6. Try Multiple Models
    print("\n5. Training multiple models...")
    print("-"*50)
    
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2', n_jobs=-1)
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mae': mae,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  MAE:      {mae:.2f}")
        print(f"  RMSE:     {rmse:.2f}")
        print(f"  CV R²:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 7. Select and save best model
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    best_model_name = max(results.items(), 
                         key=lambda x: x[1]['test_r2'])[0]
    best_model = results[best_model_name]['model']
    
    print(f"\n✅ BEST MODEL: {best_model_name}")
    print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"   MAE: {results[best_model_name]['mae']:.2f}")
    print(f"   RMSE: {results[best_model_name]['rmse']:.2f}")
    
    # Save best model with feature info
    with open('best_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'performance': {
                'test_r2': results[best_model_name]['test_r2'],
                'mae': results[best_model_name]['mae'],
                'rmse': results[best_model_name]['rmse']
            }
        }, f)
    
    print(f"\n✅ Model saved with {len(feature_names)} features")
    print(f"✅ Feature names saved to 'feature_names.json'")
    
    # 8. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Save feature importance
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['feature'][:15], 
                feature_importance_df['importance'][:15])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        print("\n✅ Feature importance plot saved")
    
    # 9. Save prediction examples
    print("\n" + "="*60)
    print("SAVING EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Create example predictions for testing
    examples = []
    for i in range(5):
        example_features = X_test.iloc[i].values
        example_scaled = scaler.transform([example_features])
        prediction = best_model.predict(example_scaled)[0]
        actual = y_test.iloc[i]
        
        examples.append({
            'features': example_features.tolist(),
            'predicted': float(prediction),
            'actual': float(actual),
            'error': float(abs(prediction - actual))
        })
    
    with open('example_predictions.json', 'w') as f:
        json.dump(examples, f, indent=4)
    
    print("✅ Example predictions saved for testing")
    
    return best_model, results, feature_names

if __name__ == "__main__":
    best_model, all_results, feature_names = train_improved_model()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nFiles created:")
    print("1. best_model.pkl - Trained model with metadata")
    print("2. scaler.pkl - Feature scaler with feature info")
    print("3. feature_names.json - List of feature names")
    print("4. feature_importance.csv - Feature importance scores")
    print("5. example_predictions.json - Test predictions")
    
    print("\nTo use in Streamlit app:")
    print("1. Run: streamlit run app_fixed.py")
    print("2. Make sure all .pkl files are in the same directory")