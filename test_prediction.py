# test_prediction.py
import pickle
import json
import numpy as np

def test_prediction():
    """Test that model and features align"""
    
    print("="*60)
    print("TESTING FEATURE ALIGNMENT")
    print("="*60)
    
    try:
        # Load feature names
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print(f"\n1. Feature names loaded ({len(feature_names)} features):")
        for i, feat in enumerate(feature_names, 1):
            print(f"   {i:2d}. {feat}")
        
        # Load model
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        model_feature_names = model_data.get('feature_names', [])
        
        print(f"\n2. Model expects {len(model_feature_names)} features")
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
        
        scaler = scaler_data['scaler']
        scaler_feature_names = scaler_data.get('feature_names', [])
        
        print(f"3. Scaler expects {len(scaler_feature_names)} features")
        
        # Check consistency
        if (len(feature_names) == len(model_feature_names) == len(scaler_feature_names)):
            print("\n✅ SUCCESS: All feature counts match!")
            
            # Test with sample data
            print("\n4. Testing with sample input...")
            
            # Create sample input matching our features
            sample_input = {}
            for feat in feature_names:
                if feat == 'Gender':
                    sample_input[feat] = 1  # male
                elif feat == 'Age':
                    sample_input[feat] = 30
                elif feat == 'Height':
                    sample_input[feat] = 175
                elif feat == 'Weight':
                    sample_input[feat] = 70
                elif feat == 'Duration':
                    sample_input[feat] = 30
                elif feat == 'Heart_Rate':
                    sample_input[feat] = 120
                elif feat == 'Body_Temp':
                    sample_input[feat] = 38.5
                elif feat == 'BMI':
                    sample_input[feat] = 70 / ((175/100) ** 2)
                elif feat == 'MET':
                    sample_input[feat] = 120 / 100
                elif feat == 'Age_Group':
                    sample_input[feat] = 0
                elif feat == 'Weight_Status':
                    sample_input[feat] = 1
                elif feat == 'Duration_Weight':
                    sample_input[feat] = 30 * 70
                elif feat == 'HeartRate_Duration':
                    sample_input[feat] = 120 * 30
                elif feat == 'Duration_Temp':
                    sample_input[feat] = 30 * 38.5
                elif feat == 'Weight_HeartRate':
                    sample_input[feat] = 70 * 120
                elif feat == 'Intensity_Score':
                    sample_input[feat] = (120 * 30) / 100
                else:
                    sample_input[feat] = 0
            
            # Create feature array in correct order
            feature_array = np.array([sample_input[feat] for feat in feature_names])
            
            print(f"   Created feature array with {len(feature_array)} values")
            print(f"   Shape: {feature_array.shape}")
            
            # Try scaling
            feature_scaled = scaler.transform([feature_array])
            print(f"   Scaled shape: {feature_scaled.shape}")
            
            # Try prediction
            prediction = model.predict(feature_scaled)[0]
            print(f"\n✅ Prediction successful: {prediction:.1f} calories")
            
        else:
            print("\n❌ ERROR: Feature counts don't match!")
            print(f"   feature_names.json: {len(feature_names)}")
            print(f"   model features: {len(model_feature_names)}")
            print(f"   scaler features: {len(scaler_feature_names)}")
            
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Run 'improved_training_fixed_v2.py' first")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_prediction()