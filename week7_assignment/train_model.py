"""
Machine Learning Model Training Script
=====================================
This script trains a Random Forest model on the Iris dataset and saves it for deployment.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("Loading Iris dataset...")
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {iris.feature_names}")
    print(f"Target classes: {iris.target_names}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df, iris

def visualize_data(df):
    """Create visualizations of the dataset"""
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pairplot-style scatter plots
    features = df.columns[:-2]  # Exclude target and species columns
    
    # Sepal length vs width
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                   hue='species', ax=axes[0,0], s=60)
    axes[0,0].set_title('Sepal Length vs Width')
    
    # Petal length vs width
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                   hue='species', ax=axes[0,1], s=60)
    axes[0,1].set_title('Petal Length vs Width')
    
    # Feature distributions
    df[features].hist(bins=20, ax=axes[1,0], alpha=0.7)
    axes[1,0].set_title('Feature Distributions')
    
    # Correlation heatmap
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('iris_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    df['species'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Distribution of Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('species_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_model(df, iris):
    """Train a Random Forest model"""
    print("\nTraining Random Forest model...")
    
    # Prepare features and target
    X = df[iris.feature_names]
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': iris.feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    return rf_model, scaler, X_test, y_test, accuracy

def save_model(model, scaler, iris_data):
    """Save the trained model and related objects"""
    print("\nSaving model and related objects...")
    
    # Save model
    joblib.dump(model, 'iris_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save feature names and target names
    model_info = {
        'feature_names': list(iris_data.feature_names),
        'target_names': list(iris_data.target_names),
        'model_type': 'Random Forest Classifier'
    }
    
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("Model saved as 'iris_model.pkl'")
    print("Scaler saved as 'scaler.pkl'")
    print("Model info saved as 'model_info.pkl'")

def main():
    """Main training pipeline"""
    print("="*60)
    print("IRIS DATASET MACHINE LEARNING TRAINING PIPELINE")
    print("="*60)
    
    # Load and explore data
    df, iris = load_and_explore_data()
    
    # Visualize data
    visualize_data(df)
    
    # Train model
    model, scaler, X_test, y_test, accuracy = train_model(df, iris)
    
    # Save model
    save_model(model, scaler, iris)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("Model files saved and ready for deployment.")
    print("="*60)

if __name__ == "__main__":
    main()
