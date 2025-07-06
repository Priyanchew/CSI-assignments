"""
Iris Species Prediction Web App
==============================
A Streamlit web application for predicting iris species using a trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load the trained model, scaler, and model information"""
    try:
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run the training script first!")
        st.stop()

def predict_species(model, scaler, features):
    """Make prediction using the trained model"""
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction, probabilities

def create_feature_comparison_chart(features, model_info):
    """Create a radar chart comparing input features with average values"""
    
    # Average values for each species (approximate)
    species_averages = {
        'setosa': [5.0, 3.4, 1.5, 0.2],
        'versicolor': [5.9, 2.8, 4.3, 1.3],
        'virginica': [6.6, 3.0, 5.6, 2.0]
    }
    
    feature_names = [name.replace(' (cm)', '').title() for name in model_info['feature_names']]
    
    fig = go.Figure()
    
    # Add input features
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=feature_names,
        fill='toself',
        name='Your Input',
        line_color='red'
    ))
    
    # Add average values for each species
    colors = ['blue', 'green', 'orange']
    for i, (species, avg_values) in enumerate(species_averages.items()):
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=feature_names,
            fill='toself',
            name=f'{species.title()} Average',
            line_color=colors[i],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 8]
            )),
        showlegend=True,
        title="Feature Comparison with Species Averages",
        height=500
    )
    
    return fig

def create_probability_chart(probabilities, target_names):
    """Create a bar chart showing prediction probabilities"""
    fig = px.bar(
        x=target_names,
        y=probabilities,
        title="Prediction Probabilities",
        labels={'x': 'Species', 'y': 'Probability'},
        color=probabilities,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Species",
        yaxis_title="Probability"
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Load model and data
    model, scaler, model_info = load_model_and_data()
    
    # Main header
    st.markdown('<h1 class="main-header">🌸 Iris Species Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This web application uses a **Random Forest machine learning model** to predict the species of iris flowers 
    based on their physical characteristics. Simply adjust the measurements below and get instant predictions!
    """)
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">🔧 Input Features</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("Adjust the sliders below to input flower measurements:")
    
    # Create input widgets
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.5,
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.sidebar.slider(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.0,
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.sidebar.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Width of the petal in centimeters"
    )
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Add example buttons
    st.sidebar.markdown("### 📝 Quick Examples")
    col1, col2, col3 = st.sidebar.columns(3)
    
    if col1.button("Setosa Example"):
        features = [5.1, 3.5, 1.4, 0.2]
    if col2.button("Versicolor Example"):
        features = [6.0, 2.8, 4.5, 1.4]
    if col3.button("Virginica Example"):
        features = [6.5, 3.0, 5.8, 2.2]
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 Input Summary</h2>', unsafe_allow_html=True)
        
        # Display input features
        input_df = pd.DataFrame({
            'Feature': model_info['feature_names'],
            'Value': features,
            'Unit': ['cm'] * 4
        })
        
        st.dataframe(input_df, use_container_width=True)
        
        # Make prediction
        prediction, probabilities = predict_species(model, scaler, features)
        predicted_species = model_info['target_names'][prediction]
        max_probability = np.max(probabilities)
        
        # Display prediction
        st.markdown('<h2 class="sub-header">🎯 Prediction Result</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🌺 Predicted Species: <strong>{predicted_species.title()}</strong></h3>
            <p>Confidence: <strong>{max_probability:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display all probabilities
        st.markdown("### 📈 All Probabilities")
        for i, (species, prob) in enumerate(zip(model_info['target_names'], probabilities)):
            st.write(f"**{species.title()}**: {prob:.2%}")
            st.progress(prob)
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Visualizations</h2>', unsafe_allow_html=True)
        
        # Probability chart
        prob_chart = create_probability_chart(probabilities, model_info['target_names'])
        st.plotly_chart(prob_chart, use_container_width=True)
        
        # Feature comparison radar chart
        radar_chart = create_feature_comparison_chart(features, model_info)
        st.plotly_chart(radar_chart, use_container_width=True)
    
    # Additional information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">ℹ️ About the Model</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Model Type</h4>
            <p>Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Dataset</h4>
            <p>UCI Iris Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Features</h4>
            <p>4 Flower Measurements</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Information about iris species
    st.markdown("### 🌸 About Iris Species")
    
    species_info = {
        "Setosa": "Small flowers with short, wide petals. Usually found in Alaska and Maine.",
        "Versicolor": "Medium-sized flowers with moderate petal dimensions. Common in Eastern North America.",
        "Virginica": "Large flowers with long, wide petals. Native to the Eastern United States."
    }
    
    for species, description in species_info.items():
        with st.expander(f"🌺 {species}"):
            st.write(description)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit and scikit-learn</p>
        <p>Machine Learning Model trained on the famous Iris dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
