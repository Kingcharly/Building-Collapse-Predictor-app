# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:16:49 2025

@author: CHARLES
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def create_lime_explanation_simple(explainer, pipeline, input_data, num_features=15):
    """Generate LIME explanation using preprocessed approach"""
    try:
        # Transform input data through preprocessing pipeline
        input_df = pd.DataFrame([input_data])
        input_transformed = pipeline.named_steps['preprocessor'].transform(input_df)

        # Row as a 1D dense vector (works for numpy array or sparse vector)
        if hasattr(input_transformed, "toarray"):
            data_row = input_transformed.toarray()[0]
        else:
            data_row = np.asarray(input_transformed)[0].ravel()
        # Define prediction function for LIME
        def predict_fn(X):
            # X is already preprocessed, so we only need the classifier
            return pipeline.named_steps['classifier'].predict_proba(X)

        # Get the actual prediction to determine which class to explain
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]

        # Determine which class to explain (the predicted class)
        class_to_explain = prediction
        
        # Generate LIME explanation
        explanation = explainer.explain_instance(
            data_row=input_transformed[0],
            predict_fn=predict_fn,
            num_features=num_features,
            labels=[prediction],
            top_labels = 2
        )
        
        return explanation, prediction, probabilities
    except Exception as e:
        st.error(f"Error generating LIME explanation: {e}")
        return None
def translate_feature_to_human(feature_name):
    """Translate technical feature names to human-readable descriptions"""
    
    # Define mappings for technical terms to human language
    feature_translations = {
        # Structural strength features
        'column_fck': 'Concrete Column Strength',
        'beam_fck': 'Concrete Beam Strength', 
        'slab_fck': 'Concrete Slab Strength',
        'bearing_capacity': 'Soil Foundation Strength',
        
        # Steel reinforcement features
        'Y8_fyk': '8mm Steel Bar Strength',
        'Y10_fyk': '10mm Steel Bar Strength',
        'Y12_fyk': '12mm Steel Bar Strength',
        'Y16_fyk': '16mm Steel Bar Strength',
        'Y20_fyk': '20mm Steel Bar Strength',
        
        
        # Building characteristics
        'floor': 'Number of Building Floors',
        'collapse_risk_score': 'Weighted Risk Factor based on Location',
        
        # Categorical features (after one-hot encoding)
        'cat__location_Kwara': 'Building Located in Kwara State',
        'cat__location_Lagos': 'Building Located in Lagos State',
        'cat__location_Abuja': 'Building Located in Abuja State',
        'cat__location_Benue': 'Building Located in Benue State',
        'cat__location_Akwa Ibom': 'Building Located in Akwa Ibom State',
        'cat__location_Plateau': 'Building Located in Plateau State',
        
        'cat__building_type_Residential': 'Residential Building Type',
        'cat__building_type_Commercial': 'Commercial Building Type',
        'cat__building_type_Church': 'Church Building Type',
        'cat__building_type_School': 'School Building Type',
        
        'cat_foundation_type_Pile': 'Pile Foundation System',
        'cat_foundation_type_Strip': 'Strip Foundation System',
        'cat_foundation_type_Wide Strip': 'Wide Foundation System',
        'cat_foundation_type_Pad': 'Pad Foundation System',
        'cat__foundation_type_Raft': 'Raft Foundation System',
        'cat__foundation_type_Strip': 'Strip Foundation System',
        
        'cat__supervision_Yes': 'Good Construction Supervision',
        'cat__supervision_No': 'Poor Construction Supervision',
        'cat__supervision_Unknown': 'Unknown Construction Supervision',
        
        # Remainder features (numeric features that come after categorical encoding)
        'remainder__floor': 'Number of Building Floors',
        'remainder__collapse_risk_score': 'Overall Risk Assessment Score',
        'remainder__column_fck': 'Concrete Column Strength',
        'remainder__beam_fck': 'Concrete Beam Strength',
        'remainder__slab_fck': 'Concrete Slab Strength',
        'remainder__Y6_fyk': '6mm Steel Bar Strength',
        'remainder__Y8_fyk': '8mm Steel Bar Strength',
        'remainder__Y10_fyk': '10mm Steel Bar Strength',
        'remainder__Y12_fyk': '12mm Steel Bar Strength',
        'remainder__Y16_fyk': '16mm Steel Bar Strength',
        'remainder__Y20_fyk': '20mm Steel Bar Strength',
        'remainder__Y25_fyk': '25mm Steel Bar Strength',
        'remainder__bearing_capacity': 'Soil Foundation Strength'
    }
    
    # Return translated name or clean up the original if not found
    if feature_name in feature_translations:
        return feature_translations[feature_name]
    else:
        # Clean up feature names that might have prefixes
        cleaned_name = feature_name.replace('remainder__', '').replace('cat__', '')
        cleaned_name = cleaned_name.replace('_', ' ').title()
        return cleaned_name
        
def parse_feature_condition(feature_string):
    """Parse LIME feature condition and convert to human-readable format"""
    
    # Handle different LIME output formats
    if ' <= ' in feature_string:
        parts = feature_string.split(' <= ')
        feature_name = parts[0].strip()
        value = parts[1].strip()
        return feature_name, f"is {value} or below"
    
    elif ' > ' in feature_string:
        parts = feature_string.split(' > ')
        feature_name = parts[0].strip()
        value = parts[1].strip()
        return feature_name, f"is above {value}"
    
    elif ' < ' in feature_string:
        parts = feature_string.split(' < ')
        feature_name = parts[0].strip()
        value = parts[1].strip()
        return feature_name, f"is below {value}"
    
    elif ' >= ' in feature_string:
        parts = feature_string.split(' >= ')
        feature_name = parts[0].strip()
        value = parts[1].strip()
        return feature_name, f"is {value} or above"
    
    elif ' = ' in feature_string:
        parts = feature_string.split(' = ')
        feature_name = parts[0].strip()
        value = parts[1].strip()
        return feature_name, f"equals {value}"
    
    else:
        # Fallback for other formats
        return feature_string.strip(), "meets certain conditions"
        
def format_contribution_explanation(feature_desc, weight, is_risk_factor=True):
    """Create human-readable explanation for each feature contribution"""
    feature_name, condition = parse_feature_condition(feature_desc)
    human_name = translate_feature_to_human(feature_name)
    abs_weight = abs(weight)
    
    # Determine impact level
    if abs_weight > 0.1:
        impact_level = "Strong"
    elif abs_weight > 0.05:
        impact_level = "Moderate"
    else:
        impact_level = "Weak"
    
    if is_risk_factor:
        if "strength" in human_name.lower():
            explanation = f"**{human_name}** is below optimal levels, significantly increasing collapse risk"
        elif "supervision" in human_name.lower() and "poor" in human_name.lower():
            explanation = f"**{human_name}** increases vulnerability to structural failures"
        elif "floor" in human_name.lower():
            explanation = f"**{human_name}** adds structural load and complexity"
        elif "risk" in human_name.lower():
            explanation = f"**{human_name}** indicates elevated danger level"
        else:
            explanation = f"**{human_name}** contributes to structural vulnerability"
    else:
        if "strength" in human_name.lower():
            explanation = f"**{human_name}** meets safety standards, reducing collapse risk"
        elif "supervision" in human_name.lower() and "good" in human_name.lower():
            explanation = f"**{human_name}** ensures quality construction practices"
        elif "foundation" in human_name.lower():
            explanation = f"**{human_name}** provides stable structural support"
        else:
            explanation = f"**{human_name}** contributes to structural safety"
    
    return f"{explanation} ({impact_level} influence: {weight:+.3f})"

def plot_lime_explanation(explanation, label=None):
    """Create interactive LIME explanation plot"""
    if explanation is None:
        return None
        
    # USe the predicted label if provided and available
    try:
        if label is not None:
            explanation_list = explanation.as_list(label=label)
        else:
            #fallback to the first available label
            labels = explanation.availble_labels()
            exlanation_list = explanation.as_list(label=labels[0] if labels else None)
    except Exception:
        # final fallback
        explanation_list = explanation.as_list()
    features = [item[0] for item in explanation_list]
    weight = [item[1] for item in explanation_list]

    # Filter out removed features
    filtered_features = []
    filtered_weight = []

    for f, w in zip (features, weight):
        if 'Y6_fyk' not in f and 'Y25_fyk' not in f:
            filtered_features.append(f)
            filtered_weight.append(w)

    # Translate to human-readable names
    human_features = []
    for feature_desc in filtered_features:
        feature_name, condition = parse_feature_condition(feature_desc)
        human_name = translate_feature_to_human(feature_name)
        human_features.append(f"{human_name} ({condition})")
    # Create color mapping
    colors = ['rgba(255, 99, 71, 0.8)' if w < 0 else 'rgba(60, 179, 113, 0.8)' for w in weight]
    
    fig = go.Figure(go.Bar(
        x=weight,
        y=human_features,
        orientation='h',
        marker_color=colors,
        text=[f'{w:.3f}' for w in weight],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title="🏗️ Building Safety Factor Analysis",
        xaxis_title="Impact on Collapse Risk (Negative = Safer, Positive = Riskier)",
        yaxis_title="Building Safety Factors",
        height=600,
        showlegend=False,
        xaxis=dict(
            zeroline=True, 
            zerolinecolor='black', 
            zerolinewidth=2,
            title_font=dict(size=12)
        ),
        yaxis=dict(title_font=dict(size=12)),
        template="plotly_white",
        font=dict(size=11)
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
        input_data['column_fck'] = st.number_input("Compressive Strength of Column Element (MPa)", 0.0, 40.0, 20.0, 1.0)
        input_data['beam_fck'] = st.number_input("Compressive Strength of Beam Element (MPa)", 0.0, 40.0, 20.0, 1.0)
        input_data['slab_fck'] = st.number_input("Compressive Strength of Slab Element (MPa)", 0.0, 40.0, 20.0, 1.0)
        input_data['bearing_capacity'] = st.number_input("Bearing Capacity of Soil (kN/m²)", 50.0, 500.0, 150.0, 10.0)
    
    with col2:
        input_data['Y8_fyk'] = st.number_input("Yield Strength of 8mm bar (MPa)", 0.0, 600.0, 0.0, 50.0)
        input_data['Y10_fyk'] = st.number_input("Yield Strength of 10mm bar (MPa)", 0.0, 600.0, 250.0, 50.0)
        input_data['Y12_fyk'] = st.number_input("Yield Strength of 12mm bar (MPa)", 0.0, 600.0, 410.0, 50.0)
        input_data['Y16_fyk'] = st.number_input("Yield Strength of 16mm bar (MPa)", 0.0, 600.0, 410.0, 50.0)
        input_data['Y20_fyk'] = st.number_input("Yield Strength of 20mm bar (MPa)", 0.0, 600.0, 0.0, 50.0)
        
    
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
                st.subheader("🔍 AI Decision Explanation")
                st.markdown("*Understanding why the AI made this prediction*")
                
                with st.spinner("Analyzing building factors..."):
                    try:
                        explanation_obj, pred_class, probs = create_lime_explanation_simple(explainer, pipeline, input_data)
                    except Exception as e:
                        st.error(f"Error generating LIME explanations")
                    
                    
                    if explanation_obj is None:
                        st.error('Could not generate LIME explanation')
                        st.info('This may be due to model complexity or data preprocessing issues')
                    else:
                        # Interactive LIME plot
                        fig_lime = plot_lime_explanation(explanation_obj, label=pred_class)
                        if fig_lime is not None:
                            st.plotly_chart(fig_lime, use_container_width=True)
                        # Get list for the predicted class 
                        try:                       
                            # Feature contribution analysis
                            explanation_list = explanation_obj.as_list(label=pred_class)
                        except Exception:
                            labels = explanation_obj.available_labels()
                            exp_list = explanation_obj.as_list(label=labels[0] if labels else None)
                            
                        st.subheader("📈 Detailed Safety Analysis")
                        # Filter out Y6_fyk and Y25_fyk from explanations
                        filtered_explanation = [(f, w) for f, w in explanation_list 
                                   if 'Y6_fyk' not in f and 'Y25_fyk' not in f]

                        # Determine if this is a collapse prediction or safe prediction
                        is_collapse_prediction = (pred_class == 1)
                        if is_collapse_prediction:
                            # For collapse predictions, positive weights increase risk
                            risk_factors = [(f, w) for f, w in filtered_explanation if w > 0]
                            safety_factors = [(f, w) for f, w in filtered_explanation if w < 0]
                
                        risk_title = "🔴 Factors Increasing Collapse Risk:"
                        safety_title = "🟢 Factors Reducing Collapse Risk:"
                        else:
                            # For safe predictions, negative weights support safety, positive weights work against safety
                            safety_factors = [(f, w) for f, w in filtered_explanation if w < 0]
                            risk_factors = [(f, w) for f, w in filtered_explanation if w > 0]
                                
                        risk_title = "🟡 Factors Working Against Safety:"
                        safety_title = "🟢 Factors Supporting Building Safety:"

                        # Sort factors
                        risk_factors.sort(key=lambda x: abs(x[1]), reverse=True)
                        safety_factors.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                        # Display risk factors (most important for low-risk predictions)
                        if risk_factors:
                            st.markdown(f"**{risk_title}**")
                            if is_collapse_prediction:
                                st.markdown("*These factors make the building more vulnerable:*")
                            else:
                                st.markdown("*These factors could be improved for better safety:*")

                                for i, (feature_desc, weight) in enumerate(risk_factors[:5]):
                                    if 'Y6_fyk' in feature_desc or 'Y25_fyk' in feature_desc:
                                        continue
                                    explanation_text = format_contribution_explanation(feature_desc, weight, is_risk_factor=True)
                                    st.markdown(f"  {i+1}. {explanation_text}")
                            
                          
                        # Display safety factors (most important for low-risk predictions)
                        if safety_factors:
                            st.markdown(f"**{safety_title}**")
                            st.markdown("*These factors contributes to the building's safety:*")
                            for i, (feature_desc, weight) in enumerate(safety_factors[:5]):
                                if 'Y6_fyk' in feature_desc or 'Y25_fyk' in feature_desc:
                                    continue
                                explanation_text = format_contribution_explanation(feature_desc, weight, is_risk_factor=False)
                                st.markdown(f"  {i+1}. {explanation_text}")
                                    
                            # Add summary interpretation
                            st.markdown("---")
                            st.subheader("💡 Summary")
                
                            total_risk = sum([w for feature_desc, w in risk_factors])
                            total_safety = abs(sum([w for feature_desc, w in safety_factors]))

                            if is_collapse_prediction:
                
                                if total_risk > total_safety:
                                    st.warning(f"⚠️ **Overall Assessment**: Risk factors (impact: +{total_risk:.3f}) outweigh safety factors (impact: -{total_safety:.3f}). Consider structural improvements.")
                                else:
                                    st.info(f"⚖️ **Mixed Assessment**: While collapse is predicted, safety factors (impact: -{total_safety:.3f}) are significant. Review critical risk factors.")
                            else:
                                st.success(f"✅ **Overall Assessment**: Safety factors (impact: -{total_safety:.3f}) outweigh risk factors (impact: +{total_risk:.3f}). Building shows good structural integrity.")
                                if total_risk > 0:
                                    st.info(f"💡 **Improvement Opportunity**: Consider addressing factors working against safety (impact: {total_risk:.3f}) for even better performance.")
    
               
                                           
                    
                
        
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
    st.markdown(
    """
    <div style="font-size:13px; color:gray; margin-top:30px;">
    ⚠️ <b>Disclaimer:</b> This prediction tool is for research and educational purposes only. 
    The results are not a substitute for professional engineering judgment or compliance with 
    building codes. Always consult qualified professionals before making construction or safety decisions.
    </div>
    """,
    unsafe_allow_html=True
)
    


if __name__ == "__main__":

    main()



































