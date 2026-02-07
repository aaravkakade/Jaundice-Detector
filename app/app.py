"""
Streamlit application for jaundice detection.
"""

import streamlit as st
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import JaundicePredictor


# Page configuration
st.set_page_config(
    page_title="Jaundice Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .jaundice-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    try:
        predictor = JaundicePredictor()
        return predictor
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🩺 Jaundice Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses a deep learning model to detect jaundice in images.
        
        **How to use:**
        1. Upload an image using the file uploader
        2. The model will analyze the image
        3. View the prediction and confidence score
        
        **Note:** This is a demonstration tool and should not be used as a substitute for professional medical diagnosis.
        """)
        
        st.header("Model Information")
        st.info("""
        **Architecture:** ResNet18  
        **Classes:** Normal, Jaundice  
        **Input Size:** 224x224 pixels
        """)
    
    # Load model
    predictor = load_model()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to analyze for jaundice"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Make prediction
            try:
                with st.spinner("Analyzing image..."):
                    result = predictor.predict(image)
                
                # Display results
                predicted_class = result['class']
                confidence = result['confidence']
                probabilities = result['probabilities']
                
                # Color-coded prediction box
                if predicted_class == "normal":
                    box_class = "normal-box"
                    emoji = "✅"
                    status = "Normal - No Jaundice Detected"
                else:
                    box_class = "jaundice-box"
                    emoji = "⚠️"
                    status = "Jaundice Detected"
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>{emoji} {status}</h2>
                    <p class="confidence-text">Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("Probability Breakdown")
                for class_name, prob in probabilities.items():
                    label = "🟢 Normal" if class_name == "normal" else "🟡 Jaundice"
                    st.write(f"{label}: {prob*100:.1f}%")
                    st.progress(prob)
                
                # Detailed probabilities
                with st.expander("View Detailed Probabilities"):
                    st.json(probabilities)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.info("👆 Please upload an image to get started")
            # Placeholder text instead of image for compatibility
            st.markdown("_Upload an image above to see it displayed here_")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "⚠️ This tool is for demonstration purposes only. Always consult a healthcare professional for medical diagnosis."
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
