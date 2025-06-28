# Streamlit Web Application for Churn Prediction
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb

# Configure page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ChurnPredictorApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_names = [f'X{i}' for i in range(216)]
        
    def load_model_components(self):
        """Load or create model components"""
        try:
            # In a real deployment, you would load saved models
            # For demo, we'll create mock components
            self.create_demo_model()
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def create_demo_model(self):
        """Create demo model components for demonstration"""
        # Create mock components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
        # Create synthetic training data to fit transformers
        np.random.seed(42)
        X_demo = np.random.randn(1000, 216)
        self.scaler.fit(X_demo)
        X_scaled = self.scaler.transform(X_demo)
        self.pca.fit(X_scaled)
        
        # Create mock XGBoost model
        self.model = xgb.XGBClassifier(random_state=42)
        X_pca = self.pca.transform(X_scaled)
        y_demo = np.random.binomial(1, 0.15, 1000)  # 15% churn rate
        self.model.fit(X_pca, y_demo)
    
    def predict_churn(self, customer_data):
        """Make churn prediction for customer data"""
        try:
            # Preprocess data
            scaled_data = self.scaler.transform(customer_data)
            pca_data = self.pca.transform(scaled_data)
            
            # Make prediction
            churn_prob = self.model.predict_proba(pca_data)[:, 1]
            
            # Determine risk level
            risk_levels = []
            for prob in churn_prob:
                if prob > 0.7:
                    risk_levels.append('High Risk')
                elif prob > 0.3:
                    risk_levels.append('Medium Risk')
                else:
                    risk_levels.append('Low Risk')
            
            return churn_prob, risk_levels
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

def main():
    # Initialize app
    app = ChurnPredictorApp()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    if not app.load_model_components():
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home", 
        "🔮 Single Prediction", 
        "📊 Batch Prediction", 
        "📈 Model Analytics",
        "ℹ️ About"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Single Prediction":
        show_single_prediction_page(app)
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page(app)
    elif page == "📈 Model Analytics":
        show_analytics_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page"""
    st.markdown("## Welcome to the Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This system helps predict customer churn probability using machine learning.
        Identify at-risk customers before they leave.
        """)
