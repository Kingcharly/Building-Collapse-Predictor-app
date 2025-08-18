# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:16:49 2025

@author: CHARLES
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go  # Remove plotly.express import
import json
import os
from joblib import load
import lime
import lime.lime_tabular
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Building Collapse Prediction",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_artifacts():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load pipeline
        pipeline = load(os.path.join(current_dir, "models", "collapse_pipeline.pkl"))
        
        # Load feature schema
        with open(os.path.join(current_dir, "artifacts", "feature_schema.json"), "r") as f:
            schema = json.load(f)
        
        # Load training sample
        training_sample_path = os.path.join(current_dir, "artifacts", "training_sample.csv")
        X_train_sample = pd.read_csv(training_sample_path)
        
        # Load preprocessed training data for LIME
        X_train_transformed = np.load(os.path.join(current_dir, "artifacts", "X_train_transformed.npy"))
        
        # FIX: Load feature names from JSON instead of numpy array
        with open(os.path.join(current_dir, "artifacts", "feature_names_transformed.json"), "r") as f:
            feature_names_transformed = json.load(f)
        
        # Load categorical mappings
        with open(os.path.join(current_dir, "artifacts", "categorical_mappings.json"), "r") as f:
            categorical_mappings = json.load(f)
        
        return pipeline, schema, X_train_sample, X_train_transformed, feature_names_transformed, categorical_mappings
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Looking for files in: {os.path.dirname(os.path.abspath(__file__))}")
        return None, None, None, None, None, None

@st.cache_resource
def create_lime_explainer(_X_train_transformed, _feature_names_transformed):
    """Create LIME explainer using preprocessed data"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=_X_train_transformed,
        feature_names=_feature_names_transformed,
        class_names=['No Collapse', 'Collapse'],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )
    return explainer

def predict_with_pipeline(pipeline, input_data):
    """Make prediction using the pipeline"""
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    return prediction, probabilities

def create_lime_explanation_simple(explainer, pipeline, input_data, num_features=10):
    """Generate LIME explanation using preprocessed approach"""
    try:
        # Transform input data through preprocessing pipeline
        input_df = pd.DataFrame([input_data])
        input_transformed = pipeline.named_steps['preprocessor'].transform(input_df)
        
        # Define prediction function for LIME
        def predict_fn(X):
            # X is already preprocessed, so we only need the classifier
            return pipeline.named_steps['classifier'].predict_proba(X)
        
        # Generate LIME explanation
        explanation = explainer.explain_instance(
            data_row=input_transformed[0],
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=1
        )
        
        return explanation
    except Exception as e:
        st.error(f"Error generating LIME explanation: {e}")
        return None

def plot_lime_explanation(explanation):
    """Create interactive LIME explanation plot"""
    if explanation is None:
        return None
        
    explanation_list = explanation.as_list()
    features = [item[0] for item in explanation_list]
    weights = [item[1] for item in explanation_list]
    
    # Create color mapping
    colors = ['rgba(255, 99, 71, 0.8)' if w < 0 else 'rgba(60, 179, 113, 0.8)' for w in weights]
    
    fig = go.Figure(go.Bar(
        x=weights,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{w:.3f}' for w in weights],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="LIME Feature Contributions",
        xaxis_title="Contribution to Collapse Prediction",
        yaxis_title="Features",
        height=600,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
        template="plotly_white"
    )
    
    return fig

def main():
    # Load resources
    artifacts = load_model_and_artifacts()
    if artifacts[0] is None:
        st.error("Failed to load model. Please ensure model files are in the correct directories.")
        return
    
    pipeline, schema, X_train_sample, X_train_transformed, feature_names_transformed, categorical_mappings = artifacts
    
    # Create LIME explainer
    explainer = create_lime_explainer(X_train_transformed, feature_names_transformed)
    
    # App title and description
    st.title("🏗️ Building Collapse Prediction System")
    st.markdown("""
    This application predicts building collapse probability using machine learning and provides 
    explanations for each prediction using LIME (Local Interpretable Model-agnostic Explanations).
    """)
    
    # Sidebar for input
    st.sidebar.header("Building Characteristics")
    st.sidebar.markdown("Enter the building parameters below:")
    
    # Create input form
    input_data = {}
    
    # Numeric features
    st.sidebar.subheader("📊 Numeric Features")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        input_data['floor'] = st.number_input("Number of Floors", min_value=1, max_value=20, value=3)
        input_data['collapse_risk_score'] = st.number_input("Collapse Risk Score", min_value=0, max_value=3)
        input_data['column_fck'] = st.number_input("Column Compressive Strength (MPa)", 0.0, 40.0, 20.0, 1.0)
        input_data['beam_fck'] = st.number_input("Beam Compressive Strength (MPa)", 0.0, 40.0, 20.0, 1.0)
        input_data['slab_fck'] = st.number_input("Slab Compressive Strength (MPa)", 0.0, 40.0, 20.0, 1.0)
        input_data['bearing_capacity'] = st.number_input("Soil Bearing Capacity (kN/m²)", 50.0, 500.0, 150.0, 10.0)
    
    with col2:
        input_data['Y6_fyk'] = st.number_input("6mm Bar Yield Strength (MPa)", 0.0, 600.0, 0.0, 5.0)
        input_data['Y8_fyk'] = st.number_input("8mm Bar Yield Strength (MPa)", 0.0, 600.0, 0.0, 5.0)
        input_data['Y10_fyk'] = st.number_input("10mm Bar Yield Strength (MPa)", 0.0, 600.0, 250.0, 5.0)
        input_data['Y12_fyk'] = st.number_input("12mm Bar Yield Strength (MPa)", 0.0, 600.0, 410.0, 5.0)
        input_data['Y16_fyk'] = st.number_input("16mm Bar Yield Strength (MPa)", 0.0, 600.0, 410.0, 5.0)
        input_data['Y20_fyk'] = st.number_input("20mm Bar Yield Strength (MPa)", 0.0, 600.0, 0.0, 5.0)
        input_data['Y25_fyk'] = st.number_input("25mm Bar Yield Strength (MPa)", 0.0, 600.0, 0.0, 5.0)
    
    # Categorical features
    st.sidebar.subheader("🏢 Categorical Features")
    
    # Use categorical mappings for dropdowns
    input_data['location'] = st.sidebar.selectbox("Location", categorical_mappings['location'])
    input_data['building_type'] = st.sidebar.selectbox("Building Type", categorical_mappings['building_type'])
    input_data['foundation_type'] = st.sidebar.selectbox("Foundation Type", categorical_mappings['foundation_type'])
    input_data['supervision'] = st.sidebar.selectbox("Supervision", categorical_mappings['supervision'])
    
    # Prediction button
    if st.sidebar.button("🔍 Analyze Building", type="primary"):
        try:
            # Make prediction
            prediction, probabilities = predict_with_pipeline(pipeline, input_data)
            
            # Main content area
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display prediction results
                st.subheader("📋 Prediction Results")
                
                collapse_prob = probabilities[1]
                no_collapse_prob = probabilities[0]
                
                # Create prediction gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = collapse_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Collapse Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if collapse_prob > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Prediction interpretation
                if prediction == 1:
                    st.error(f"⚠️ **HIGH RISK**: Building likely to collapse (Probability: {collapse_prob:.1%})")
                else:
                    st.success(f"✅ **LOW RISK**: Building unlikely to collapse (Probability: {collapse_prob:.1%})")
                    
                # Probability breakdown
            st.subheader("📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['No Collapse', 'Collapse'],
                'Probability': [no_collapse_prob, collapse_prob]
            })
            
            fig_bar = px.bar(prob_df, x='Outcome', y='Probability', 
                           color='Probability', color_continuous_scale='RdYlGn_r')
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            
            
            with col2:
                # LIME Explanation
                st.subheader("🔍 Decision Explanation (LIME)")
                
                with st.spinner("Generating explanation..."):
                    explanation = create_lime_explanation_simple(explainer, pipeline, input_data)
                    
                    if explanation is not None:
                        # Interactive LIME plot
                        fig_lime = plot_lime_explanation(explanation)
                        if fig_lime is not None:
                            st.plotly_chart(fig_lime, use_container_width=True)
                            
                            # Feature contribution analysis
                            explanation_list = explanation.as_list()
                            
                            st.subheader("📈 Key Contributing Factors")
                            
                            # Positive contributors (increase collapse risk)
                            risk_factors = [(f, w) for f, w in explanation_list if w > 0]
                            risk_factors.sort(key=lambda x: x[1], reverse=True)
                            
                            if risk_factors:
                                st.markdown("**🔴 Factors Increasing Collapse Risk:**")
                                for i, (feature, weight) in enumerate(risk_factors[:5]):
                                    st.write(f"  {i+1}. **{feature}**: +{weight:.3f}")
                            
                            # Negative contributors (decrease collapse risk)
                            safety_factors = [(f, w) for f, w in explanation_list if w < 0]
                            safety_factors.sort(key=lambda x: x[1])
                            
                            if safety_factors:
                                st.markdown("**🟢 Factors Reducing Collapse Risk:**")
                                for i, (feature, weight) in enumerate(safety_factors[:5]):
                                    st.write(f"  {i+1}. **{feature}**: {weight:.3f}")
                
                    else:
                        st.error("Could not generate LIME explanation")
                        
                    
                
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please check your input values and try again.")
            
        

    else:
        # Welcome message when no prediction made
        st.info("👈 Please enter building characteristics in the sidebar and click 'Analyze Building' to get predictions and explanations.")
        # Show some example information
        st.subheader("🎯 About This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🤖 Model Information:**
            - **Algorithm**: Random Forest Classifier
            - **Features**: 17 building characteristics
            - **Target**: Binary collapse prediction
            - **Preprocessing**: Categorical encoding
            """)
        
        with col2:
            st.markdown("""
            **🔍 LIME Explanations:**
            - **Local explanations** for each prediction
            - **Feature importance** for the specific building
            - **Interactive visualizations** of decision factors
            - **Transparency** in AI decision-making
            """)
        
    # Building summary
        st.subheader("🏗️ Building Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Floors", input_data['floor'])
            st.metric("Risk Score", f"{input_data['collapse_risk_score']:.1f}")
            st.metric("Location", input_data['location'])
        
        with summary_col2:
            st.metric("Column Strength", f"{input_data['column_fck']:.0f} MPa")
            st.metric("Beam Strength", f"{input_data['beam_fck']:.0f} MPa")
            st.metric("Building Type", input_data['building_type'])
        
        with summary_col3:
            st.metric("Bearing Capacity", f"{input_data['bearing_capacity']:.0f} kN/m²")
            st.metric("Foundation", input_data['foundation_type'])
            st.metric("Supervision", input_data['supervision'])


    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p>🏗️ Building Collapse Prediction System | Developed by Charles Eze Powered by Machine Learning & LIME</p>
    </div>
    """, unsafe_allow_html=True)
    


if __name__ == "__main__":

    main()
