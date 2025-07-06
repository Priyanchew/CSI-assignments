# Iris Species Prediction Web App 🌸

A machine learning web application built with Streamlit that predicts iris flower species based on physical measurements.

## 🚀 **Live Demo**
**Try the app now:** [iris-species-predication.streamlit.app](https://iris-species-predication.streamlit.app)

> **Note:** If the deployment is currently failing, please see the troubleshooting section below for Python version compatibility fixes.

## Features

- **Interactive Web Interface**: Easy-to-use sliders for input
- **Real-time Predictions**: Instant predictions as you adjust parameters
- **Beautiful Visualizations**: Radar charts and probability plots
- **Model Transparency**: View prediction confidence and feature importance
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
week7_assignment/
├── train_model.py          # Model training script
├── streamlit_app.py        # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── iris_model.pkl         # Trained model (generated after training)
├── scaler.pkl             # Feature scaler (generated after training)
└── model_info.pkl         # Model metadata (generated after training)
```

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

First, run the training script to create the model files:

```bash
python train_model.py
```

This will:
- Load and explore the Iris dataset
- Train a Random Forest classifier
- Generate visualizations
- Save the trained model and scaler

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add iris prediction app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `streamlit_app.py` as the main file
   - Click "Deploy"

### Option 2: Heroku

1. **Create Procfile**:
   ```
   web: sh setup.sh && streamlit run streamlit_app.py
   ```

2. **Create setup.sh**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Local Network Deployment

Run with external access:
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## Usage

1. **Input Features**: Use the sidebar sliders to adjust flower measurements
2. **Quick Examples**: Click example buttons for pre-set values
3. **View Predictions**: See the predicted species and confidence level
4. **Explore Visualizations**: Check probability charts and feature comparisons
5. **Learn About Species**: Expand the information sections to learn more

## Model Details

- **Algorithm**: Random Forest Classifier
- **Dataset**: UCI Iris Dataset (150 samples, 4 features, 3 classes)
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Accuracy**: ~97% (typical performance)
- **Classes**: Setosa, Versicolor, Virginica

## Technical Features

- **Feature Scaling**: StandardScaler for consistent input ranges
- **Interactive Visualizations**: Plotly charts for better user experience
- **Caching**: Streamlit caching for improved performance
- **Responsive Design**: CSS styling for professional appearance
- **Error Handling**: Graceful handling of missing model files

## Troubleshooting

### Streamlit Cloud Deployment Issues

**Python version compatibility error:**
If you see `ModuleNotFoundError: No module named 'distutils'`, this is due to Python 3.13+ compatibility. The fix is already included:
- `runtime.txt` specifies Python 3.11
- Updated `requirements.txt` with compatible package versions

**To redeploy:**
1. Push the updated files to GitHub
2. In Streamlit Cloud, click "Reboot app" or redeploy

### Model files not found
If you see "Model files not found", run the training script first:
```bash
python train_model.py
```

### Import errors
Install all dependencies:
```bash
pip install -r requirements.txt
```

### Port already in use
Use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub.
