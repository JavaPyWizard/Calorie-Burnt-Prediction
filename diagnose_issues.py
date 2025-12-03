# diagnose_issues.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("DIAGNOSING MODEL ACCURACY ISSUES")
print("="*60)

# 1. Load and inspect your data
df = pd.read_csv('calories.csv')
print(f"\n1. Dataset Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# 2. Check target variable distribution
print("\n2. Target Variable (Calories) Analysis:")
print(f"   Min: {df['Calories'].min():.1f}")
print(f"   Max: {df['Calories'].max():.1f}")
print(f"   Mean: {df['Calories'].mean():.1f}")
print(f"   Std: {df['Calories'].std():.1f}")
print(f"   Range: {df['Calories'].max() - df['Calories'].min():.1f}")

# 3. Check for data issues
print("\n3. Data Quality Check:")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Duplicate rows: {df.duplicated().sum()}")

# 4. Check correlations
df_encoded = df.copy()
if 'Gender' in df.columns:
    le = LabelEncoder()
    df_encoded['Gender'] = le.fit_transform(df['Gender'])

print("\n4. Correlation with Calories (target):")
correlations = df_encoded.corr()['Calories'].abs().sort_values(ascending=False)
print(correlations)

# Weak correlation warning
weak_features = correlations[correlations < 0.3].index.tolist()
if len(weak_features) > 0:
    print(f"\n   ⚠️ WARNING: Weak correlation with Calories (<0.3):")
    for feat in weak_features:
        print(f"      - {feat}: {correlations[feat]:.3f}")

# 5. Feature importance analysis (simplified)
print("\n5. Feature Importance Analysis:")
# Duration and Weight should be most important
if 'Duration' in correlations:
    print(f"   Duration correlation: {correlations.get('Duration', 0):.3f}")
if 'Weight' in correlations:
    print(f"   Weight correlation: {correlations.get('Weight', 0):.3f}")
if 'Heart_Rate' in correlations:
    print(f"   Heart_Rate correlation: {correlations.get('Heart_Rate', 0):.3f}")

# 6. Check for outliers
print("\n6. Outlier Detection:")
numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# 7. Data visualization
plt.figure(figsize=(15, 10))

# Distribution of Calories
plt.subplot(2, 2, 1)
sns.histplot(df['Calories'], kde=True)
plt.title('Distribution of Calories (Target)')

# Scatter plots for important features
plt.subplot(2, 2, 2)
if 'Duration' in df.columns:
    plt.scatter(df['Duration'], df['Calories'], alpha=0.5)
    plt.xlabel('Duration')
    plt.ylabel('Calories')
    plt.title('Calories vs Duration')

plt.subplot(2, 2, 3)
if 'Weight' in df.columns:
    plt.scatter(df['Weight'], df['Calories'], alpha=0.5)
    plt.xlabel('Weight')
    plt.ylabel('Calories')
    plt.title('Calories vs Weight')

plt.subplot(2, 2, 4)
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')

plt.tight_layout()
plt.savefig('diagnostic_plots.png', dpi=150, bbox_inches='tight')
print(f"\n7. Diagnostic plots saved as 'diagnostic_plots.png'")

print("\n" + "="*60)
print("RECOMMENDATIONS BASED ON ANALYSIS:")
print("="*60)

# Generate recommendations
if correlations['Calories'] < 0.5:  # If self-correlation is weird
    print("1. Check if 'Calories' column has correct data")
elif len(weak_features) > 3:
    print("1. Too many weak features. Consider:")
    print("   - Feature engineering")
    print("   - Removing weak features")
    print("   - Collecting more relevant data")
else:
    print("1. Data structure looks OK for basic modeling")

print("2. If accuracy is still low, try:")
print("   - Feature engineering (create new features)")
print("   - Try different models (XGBoost, Neural Networks)")
print("   - Hyperparameter tuning")
print("   - Ensemble methods")

print("\n3. Run the improved training script below:")
print("   python improved_training_fixed.py")