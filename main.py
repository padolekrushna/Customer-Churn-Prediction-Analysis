# Churn Prediction Model - Complete Implementation
# This script provides a comprehensive solution for customer churn prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    """
    Comprehensive churn prediction model with full pipeline
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)  # Reduce to 50 components
        self.best_model = None
        self.feature_importance = None
        
    def load_and_explore_data(self, data_path):
        """
        Load and perform initial exploration of the dataset
        """
        print("Loading and exploring data...")
        
        # Load data
        self.df = pd.read_csv(data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Basic info
        print("\nDataset Info:")
        print(f"Total rows: {len(self.df):,}")
        print(f"Total features: {self.df.shape[1] - 2}")  # Excluding UID and target
        
        # Target distribution
        target_dist = self.df['Target_ChurnFlag'].value_counts()
        print(f"\nTarget Distribution:")
        print(f"Retained (0): {target_dist[0]:,} ({target_dist[0]/len(self.df)*100:.1f}%)")
        print(f"Churned (1): {target_dist[1]:,} ({target_dist[1]/len(self.df)*100:.1f}%)")
        
        # Missing values check
        missing_values = self.df.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        return self.df
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing pipeline
        """
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.df.drop(['UID', 'Target_ChurnFlag'], axis=1)
        y = self.df['Target_ChurnFlag']
        
        # Feature correlation analysis
        print("Analyzing feature correlations...")
        correlation_matrix = X.corr().abs()
        
        # Find highly correlated features (>0.95)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for pair in high_corr_pairs:
            features_to_remove.add(pair[1])  # Remove the second feature
        
        X = X.drop(columns=list(features_to_remove))
        print(f"Removed {len(features_to_remove)} highly correlated features")
        print(f"Remaining features: {X.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_train_pca, X_test_pca, y_train, y_test, X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models and compare performance
        """
        print("\nTraining multiple models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"AUC-ROC: {auc_score:.3f}")
            print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models = results
        return results
    
    def select_best_model(self):
        """
        Select the best performing model based on AUC-ROC
        """
        best_auc = 0
        best_model_name = None
        
        print("\nModel Comparison:")
        print("=" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'AUC-ROC':<10} {'CV AUC':<15}")
        print("=" * 60)
        
        for name, results in self.models.items():
            print(f"{name:<20} {results['accuracy']:<10.3f} {results['auc_score']:<10.3f} {results['cv_mean']:.3f}±{results['cv_std']:.3f}")
            
            if results['auc_score'] > best_auc:
                best_auc = results['auc_score']
                best_model_name = name
        
        print("=" * 60)
        print(f"Best Model: {best_model_name} (AUC-ROC: {best_auc:.3f})")
        
        self.best_model = self.models[best_model_name]['model']
        return best_model_name, self.best_model
    
    def analyze_feature_importance(self, X_train, feature_names=None):
        """
        Analyze feature importance for tree-based models
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'PC{i+1}' for i in range(len(importance))]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def generate_insights(self, y_test):
        """
        Generate business insights from the model
        """
        print("\nGenerating Business Insights...")
        
        best_model_name = None
        for name, results in self.models.items():
            if results['model'] == self.best_model:
                best_model_name = name
                break
        
        y_pred_proba = self.models[best_model_name]['probabilities']
        
        # Risk segmentation
        high_risk = (y_pred_proba > 0.7).sum()
        medium_risk = ((y_pred_proba > 0.3) & (y_pred_proba <= 0.7)).sum()
        low_risk = (y_pred_proba <= 0.3).sum()
        
        total_customers = len(y_pred_proba)
        
        print(f"\nCustomer Risk Segmentation:")
        print(f"High Risk (>70%): {high_risk:,} customers ({high_risk/total_customers*100:.1f}%)")
        print(f"Medium Risk (30-70%): {medium_risk:,} customers ({medium_risk/total_customers*100:.1f}%)")
        print(f"Low Risk (<30%): {low_risk:,} customers ({low_risk/total_customers*100:.1f}%)")
        
        # Calculate potential business impact
        # Assumptions: 
        # - Average customer value: $1000/year
        # - Retention cost: $50 per customer
        # - Successfully retain 60% of identified high-risk customers
        
        avg_customer_value = 1000
        retention_cost = 50
        success_rate = 0.6
        
        actual_churners = y_test.sum()
        identified_churners = ((y_pred_proba > 0.5) & (y_test == 1)).sum()
        
        print(f"\nBusiness Impact Analysis:")
        print(f"Total actual churners: {actual_churners}")
        print(f"Correctly identified churners: {identified_churners}")
        print(f"Identification rate: {identified_churners/actual_churners*100:.1f}%")
        
        potential_savings = identified_churners * success_rate * (avg_customer_value - retention_cost)
        intervention_cost = high_risk * retention_cost
        net_savings = potential_savings - intervention_cost
        
        print(f"Potential annual savings: ${potential_savings:,.0f}")
        print(f"Intervention cost: ${intervention_cost:,.0f}")
        print(f"Net savings: ${net_savings:,.0f}")
    
    def create_prediction_function(self):
        """
        Create a function for making predictions on new data
        """
        def predict_churn(new_data):
            """
            Predict churn probability for new customer data
            
            Args:
                new_data: pandas DataFrame with same features as training data
                
            Returns:
                Dictionary with churn probability and risk level
            """
            # Preprocess new data
            scaled_data = self.scaler.transform(new_data)
            pca_data = self.pca.transform(scaled_data)
            
            # Make prediction
            churn_prob = self.best_model.predict_proba(pca_data)[:, 1]
            
            # Determine risk level
            risk_levels = []
            for prob in churn_prob:
                if prob > 0.7:
                    risk_levels.append('High Risk')
                elif prob > 0.3:
                    risk_levels.append('Medium Risk')
                else:
                    risk_levels.append('Low Risk')
            
            return {
                'churn_probability': churn_prob,
                'risk_level': risk_levels,
                'recommendation': ['Immediate intervention needed' if level == 'High Risk' 
                                else 'Monitor closely' if level == 'Medium Risk' 
                                else 'Standard retention program' for level in risk_levels]
            }
        
        return predict_churn

def main():
    """
    Main execution function
    """
    # Initialize model
    churn_model = ChurnPredictionModel()
    
    # Note: Replace with actual data path
    # data_path = 'path_to_your_data.csv'
    # df = churn_model.load_and_explore_data(data_path)
    
    # For demonstration, create synthetic data
    print("Creating synthetic data for demonstration...")
    np.random.seed(42)
    n_samples = 10000
    n_features = 216
    
    # Generate synthetic data
    X_synthetic = np.random.randn(n_samples, n_features)
    
    # Create target with some pattern (customers with higher values in first few features more likely to churn)
    churn_prob = 1 / (1 + np.exp(-(X_synthetic[:, :5].mean(axis=1) * 0.5 + np.random.randn(n_samples) * 0.2)))
    y_synthetic = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    feature_cols = [f'X{i}' for i in range(n_features)]
    df_synthetic = pd.DataFrame(X_synthetic, columns=feature_cols)
    df_synthetic['UID'] = range(n_samples)
    df_synthetic['Target_ChurnFlag'] = y_synthetic
    
    churn_model.df = df_synthetic
    
    print(f"Synthetic dataset created with shape: {df_synthetic.shape}")
    print(f"Churn rate: {y_synthetic.mean():.3f}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = churn_model.preprocess_data()
    
    # Train models
    results = churn_model.train_models(X_train, X_test, y_train, y_test)
    
    # Select best model
    best_model_name, best_model = churn_model.select_best_model()
    
    # Analyze feature importance
    churn_model.analyze_feature_importance(X_train)
    
    # Generate insights
    churn_model.generate_insights(y_test)
    
    # Create prediction function
    predict_func = churn_model.create_prediction_function()
    
    print("\nModel training completed successfully!")
    print("Ready for deployment...")
    
    return churn_model, predict_func

if __name__ == "__main__":
    model, predict_function = main()
