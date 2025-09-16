import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="AgroYield AI",
    page_icon="🌾",
    layout="centered", # Use 'centered' for a more mobile-friendly look
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6;
    }
    /* Main title styling */
    .st-emotion-cache-10trblm {
        text-align: center;
        font-weight: bold;
        color: #1a1a1a;
    }
    /* Container styling for cards */
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-r421ms {
        border-radius: 15px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px !important;
        background-color: #ffffff;
    }
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        border: none;
        color: white;
        background-color: #007AFF; /* Apple's classic blue */
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
# Caching the model load for better performance
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ **Model file not found!** Please ensure 'model.pkl' is in the project folder.")
        return None

model = load_model()

# --- Sidebar Content ---
st.sidebar.title("🧠 Model Insights")

if model:
    st.sidebar.header("📊 Model Performance")
    # Dummy metrics (replace with actual test set results)
    y_true_dummy = np.array([20, 30, 40, 50, 60])
    y_pred_dummy = np.array([22, 29, 42, 48, 61])
    rmse = np.sqrt(mean_squared_error(y_true_dummy, y_pred_dummy))
    r2 = r2_score(y_true_dummy, y_pred_dummy)

    st.sidebar.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    st.sidebar.metric("R² Score", f"{r2:.2f}")
    st.sidebar.caption("Lower RMSE and higher R² indicate better model accuracy.")
    
    st.sidebar.divider()

    st.sidebar.header("🔑 Feature Importance")
    # Using a placeholder image for feature importance as plotting can be slow
    # To use the actual plot, uncomment the lines below
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        xgb.plot_importance(model, importance_type='gain', ax=ax, show_values=False, grid=False)
        plt.title('Feature Importance (Gain)')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        st.sidebar.pyplot(fig)
    except Exception as e:
        st.sidebar.warning(f"Could not plot feature importance: {e}")

# --- Main Page Content ---
st.title("🌾 AgroYield AI")
st.markdown("<h3 style='text-align: center; color: grey;'>Predicting Crop Yields in Maharashtra</h3>", unsafe_allow_html=True)

st.write("") # Spacer

# --- Input Form ---
with st.container(border=True):
    st.header("⚙️ Input Parameters")
    
    # Define crop options
    CROP_OPTIONS = ["Wheat", "Rice", "Soybean", "Sugarcane", "Cotton"]
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2024, max_value=2040, value=2025)
        crop = st.selectbox("Select Crop", CROP_OPTIONS)
        area = st.number_input("Cultivated Area (Hectares)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=2500.0, value=700.0, step=25.0)
        fertilizer = st.number_input("Fertilizer Usage (kg/ha)", min_value=0.0, max_value=500.0, value=120.0, step=10.0)

    with col2:
        temperature = st.slider("Average Temperature (°C)", 5.0, 50.0, 25.0)
        irrigation = st.slider("Irrigation Frequency (per week)", 0, 7, 2)
        nitrogen = st.slider("Soil Nitrogen (N)", 0.0, 150.0, 40.0)
        phosphorus = st.slider("Soil Phosphorus (P)", 0.0, 150.0, 20.0)
        potassium = st.slider("Soil Potassium (K)", 0.0, 150.0, 35.0)

# --- Prediction Logic ---
if st.button("🔍 Predict Yield", use_container_width=True):
    if model:
        # Create a dictionary for the input features
        input_features = {
            'Year': year,
            'Area': area,
            'Rainfall': rainfall,
            'Temperature': temperature,
            'N': nitrogen,
            'P': phosphorus,
            'K': potassium,
            'Fertilizer': fertilizer,
            'Irrigation': irrigation
        }
        
        # Add one-hot encoded crop features
        for c in CROP_OPTIONS:
            input_features[f'Crop_{c}'] = 1 if crop == c else 0
            
        # Create a DataFrame in the correct column order expected by the model
        # IMPORTANT: This order must match the training data's column order
        expected_columns = ['Year', 'Area', 'Rainfall', 'Temperature', 'N', 'P', 'K', 
                            'Fertilizer', 'Irrigation'] + [f'Crop_{c}' for c in CROP_OPTIONS]
        
        input_df = pd.DataFrame([input_features])
        input_df = input_df[expected_columns] # Ensure column order
        
        # Run prediction
        prediction = model.predict(input_df)[0]
        
        st.write("") # Spacer
        
        with st.container(border=True):
            st.subheader("📈 Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label=f"Predicted Yield for {crop}",
                    value=f"{prediction:.2f}",
                    delta="Quintals/Hectare"
                )
            with col2:
                st.info("💡 **Recommendation:**", icon="ℹ️")
                st.write("For an optimal yield, consider adjusting fertilizer application based on soil test results and maintaining consistent irrigation, especially during dry spells.")

    else:
        st.error("Model not loaded. Cannot perform prediction.")

st.markdown("---")
st.caption("© 2025 AgroYield AI | Built with Streamlit")
