# app_fixed.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Calories Burnt Prediction",
    page_icon="🔥",
    layout="wide"
)

# Title
st.title("🔥 Calories Burnt Prediction")
st.markdown("Predict calories burnt during exercise using machine learning")

def load_model_and_features():
    """Load model, scaler, and feature names"""
    try:
        # Load model with metadata
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        model_name = model_data.get('model_name', 'Unknown')
        trained_feature_names = model_data.get('feature_names', [])
        n_features = model_data.get('n_features', 0)
        
        # Load scaler with metadata
        with open('scaler.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
        
        scaler = scaler_data['scaler']
        scaler_feature_names = scaler_data.get('feature_names', [])
        
        # Load feature names
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Verify consistency
        if len(trained_feature_names) != len(scaler_feature_names):
            st.warning(f"⚠️ Feature count mismatch: Model={len(trained_feature_names)}, Scaler={len(scaler_feature_names)}")
        
        return {
            'model': model,
            'model_name': model_name,
            'scaler': scaler,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'performance': model_data.get('performance', {})
        }
        
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def prepare_features_for_prediction(input_features, expected_features):
    """
    Prepare features in the exact same way as training
    """
    # Extract basic features
    gender = 1 if input_features['Gender'] == 'male' else 0
    age = float(input_features['Age'])
    height = float(input_features['Height'])
    weight = float(input_features['Weight'])
    duration = float(input_features['Duration'])
    heart_rate = float(input_features['Heart_Rate'])
    body_temp = float(input_features['Body_Temp'])
    
    # Calculate all engineered features
    features_dict = {}
    
    # Original features
    features_dict['Gender'] = gender
    features_dict['Age'] = age
    features_dict['Height'] = height
    features_dict['Weight'] = weight
    features_dict['Duration'] = duration
    features_dict['Heart_Rate'] = heart_rate
    features_dict['Body_Temp'] = body_temp
    
    # Engineered features (must match training)
    # BMI
    features_dict['BMI'] = weight / ((height/100) ** 2)
    
    # MET
    features_dict['MET'] = heart_rate / 100
    
    # Age Group
    if age <= 30:
        features_dict['Age_Group'] = 0
    elif age <= 50:
        features_dict['Age_Group'] = 1
    else:
        features_dict['Age_Group'] = 2
    
    # Weight Status based on BMI
    bmi = features_dict['BMI']
    if bmi < 18.5:
        features_dict['Weight_Status'] = 0
    elif bmi < 25:
        features_dict['Weight_Status'] = 1
    elif bmi < 30:
        features_dict['Weight_Status'] = 2
    else:
        features_dict['Weight_Status'] = 3
    
    # Interaction terms
    features_dict['Duration_Weight'] = duration * weight
    features_dict['HeartRate_Duration'] = heart_rate * duration
    features_dict['Duration_Temp'] = duration * body_temp
    features_dict['Weight_HeartRate'] = weight * heart_rate
    features_dict['Intensity_Score'] = (heart_rate * duration) / 100
    
    # Create feature array in EXACT same order as training
    feature_array = []
    missing_features = []
    
    for feat in expected_features:
        if feat in features_dict:
            feature_array.append(features_dict[feat])
        else:
            # If feature is missing, use 0 (or appropriate default)
            st.warning(f"Feature '{feat}' not found in input. Using default value 0.")
            feature_array.append(0)
            missing_features.append(feat)
    
    if missing_features:
        st.warning(f"Missing features: {missing_features}")
    
    return np.array(feature_array), features_dict

def predict_calories(input_features):
    """Main prediction function with proper feature alignment"""
    # Load model and features
    model_data = load_model_and_features()
    
    if model_data is None:
        return None, None, None
    
    try:
        # Prepare features in exact same order
        feature_array, all_features = prepare_features_for_prediction(
            input_features, 
            model_data['feature_names']
        )
        
        # Debug: Show feature count
        st.sidebar.info(f"Model expects {len(model_data['feature_names'])} features")
        st.sidebar.info(f"Prepared {len(feature_array)} features")
        
        # Reshape for prediction
        features_reshaped = feature_array.reshape(1, -1)
        
        # Scale features
        features_scaled = model_data['scaler'].transform(features_reshaped)
        
        # Predict
        prediction = model_data['model'].predict(features_scaled)[0]
        
        # Ensure reasonable bounds
        if prediction < 0:
            prediction = 10
        elif prediction > 2000:
            prediction = 2000
        
        return round(prediction, 1), model_data, all_features
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.info("Check if features match the training data")
        return None, None, None

# Sidebar with model info
with st.sidebar:
    st.header("📊 Model Information")
    
    # Try to load model info
    model_data = load_model_and_features()
    
    if model_data:
        st.success(f"✅ Model: {model_data['model_name']}")
        st.success(f"✅ Features: {model_data['n_features']}")
        
        # Show performance
        perf = model_data.get('performance', {})
        if perf:
            st.metric("R² Score", f"{perf.get('test_r2', 0):.3f}")
            st.metric("MAE", f"{perf.get('mae', 0):.1f}")
            st.metric("RMSE", f"{perf.get('rmse', 0):.1f}")
        
        # Show feature count
        with st.expander("View Features"):
            features = model_data.get('feature_names', [])
            for i, feat in enumerate(features, 1):
                st.write(f"{i}. {feat}")
    else:
        st.warning("⚠️ Model not loaded")
        st.write("Run training first:")
        st.code("python improved_training_fixed_v2.py")

# Main input form
st.header("📝 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.slider("Age (years)", 15, 80, 30, 1)
    height = st.slider("Height (cm)", 120, 220, 175, 1)
    weight = st.slider("Weight (kg)", 40, 150, 70, 1)

with col2:
    st.subheader("💪 Exercise Metrics")
    duration = st.slider("Duration (min)", 1, 180, 30, 1)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120, 1)
    body_temp = st.slider("Body Temp (°C)", 36.0, 42.0, 38.5, 0.1)

# Display input summary
st.subheader("📋 Input Summary")
input_df = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Height (cm)": height,
    "Weight (kg)": weight,
    "Duration (min)": duration,
    "Heart Rate (bpm)": heart_rate,
    "Body Temp (°C)": body_temp
}])
st.dataframe(input_df, use_container_width=True)

# Prediction button
if st.button("🔥 Predict Calories Burnt", type="primary", use_container_width=True):
    # Prepare input
    input_features = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp
    }
    
    with st.spinner("Calculating..."):
        # Get prediction
        prediction, model_info, all_features = predict_calories(input_features)
        
        if prediction is not None:
            # Display result
            st.success(f"## 🎯 Predicted Calories: **{prediction} kcal**")
            
            # Show additional info
            with st.expander("📊 Prediction Details"):
                # Show engineered features
                st.write("**Engineered Features:**")
                engineered_df = pd.DataFrame([{
                    'BMI': round(all_features.get('BMI', 0), 2),
                    'MET': round(all_features.get('MET', 0), 2),
                    'Age Group': all_features.get('Age_Group', 0),
                    'Weight Status': all_features.get('Weight_Status', 0),
                    'Duration × Weight': round(all_features.get('Duration_Weight', 0), 1),
                    'Heart Rate × Duration': round(all_features.get('HeartRate_Duration', 0), 1),
                    'Intensity Score': round(all_features.get('Intensity_Score', 0), 1)
                }])
                st.dataframe(engineered_df, use_container_width=True)
                
                # Show calories per minute
                calories_per_min = prediction / duration
                st.metric("Calories per Minute", f"{calories_per_min:.1f}")
            
            # Activity level
            calories_per_min = prediction / duration
            if calories_per_min < 5:
                level = "Light"
                color = "green"
            elif calories_per_min < 10:
                level = "Moderate"
                color = "orange"
            else:
                level = "Intense"
                color = "red"
            
            st.info(f"**Activity Level:** :{color}[{level}]")
            
            # Comparison with common activities
            st.subheader("📈 Comparison with Common Activities")
            
            comparisons = {
                "Walking (3 mph)": 3.5 * weight * duration / 200,
                "Cycling (10 mph)": 7.0 * weight * duration / 200,
                "Running (6 mph)": 10.0 * weight * duration / 200,
                "Swimming": 8.0 * weight * duration / 200
            }
            
            comp_data = []
            for activity, cal in comparisons.items():
                comp_data.append({
                    "Activity": activity,
                    f"Est. Calories ({duration}min)": round(cal, 1),
                    "Your Prediction": prediction,
                    "Difference": round(prediction - cal, 1)
                })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)

# Debug section
with st.expander("🔍 Debug Information"):
    if st.button("Check Feature Alignment"):
        try:
            # Load and display feature info
            with open('feature_names.json', 'r') as f:
                features = json.load(f)
            
            st.write(f"**Trained Features ({len(features)}):**")
            for i, feat in enumerate(features, 1):
                st.write(f"{i}. {feat}")
            
            # Test with sample input
            test_input = {
                "Gender": "male",
                "Age": 30,
                "Height": 175,
                "Weight": 70,
                "Duration": 30,
                "Heart_Rate": 120,
                "Body_Temp": 38.5
            }
            
            feature_array, all_features = prepare_features_for_prediction(test_input, features)
            st.write(f"\n**Prepared {len(feature_array)} features:**")
            st.write(feature_array)
            
        except Exception as e:
            st.error(f"Debug error: {e}")

# Training instructions
st.markdown("---")
st.subheader("🛠️ Setup Instructions")

with st.expander("Click here for setup instructions"):
    st.write("""
    ### If you haven't trained the model:
    
    1. **Install requirements:**
    ```bash
    pip install streamlit numpy pandas scikit-learn xgboost matplotlib seaborn
    ```
    
    2. **Run the new training script:**
    ```bash
    python improved_training_fixed_v2.py
    ```
    
    3. **Check created files:**
    - `best_model.pkl` - Trained model
    - `scaler.pkl` - Feature scaler
    - `feature_names.json` - Feature names
    - `feature_importance.png` - Feature importance plot
    
    4. **Run this app:**
    ```bash
    streamlit run app_fixed.py
    ```
    
    ### Expected Features:
    The model should be trained with 13 features including:
    - 7 original features
    - 6 engineered features (BMI, MET, Age_Group, etc.)
    """)

# Footer
st.markdown("---")
st.caption("Calories Burnt Prediction Model v2.0 | Feature-aligned version")