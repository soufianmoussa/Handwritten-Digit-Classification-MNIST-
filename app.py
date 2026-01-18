import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Import our modules
import preprocess
import model

# Page configuration
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon=":pencil2:")

st.title("Handwritten Digit Classification (MNIST)")
st.markdown("Draw a digit (0-9). The model uses the standard MNIST dataset (28x28).")

# Sidebar for controls
st.sidebar.header("Settings")
stroke_width = st.sidebar.slider("Stroke width", 1, 30, 20)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Load Model
@st.cache_resource
def load_nn_model():
    return model.load_weights('model_weights.npz')

Theta1, Theta2 = load_nn_model()

if Theta1 is None or Theta2 is None:
    st.error("Model weights not found! Please run `python train.py` first.")
    st.stop()

# Check for dimension mismatch (Migration safety)
# Input layer size is 784, so Theta1 columns should be 785 (784 + 1 bias)
if Theta1.shape[1] != 785:
    st.warning(f"⚠️ **Model Mismatch Detected** ⚠️\n\nThe current weights expect {Theta1.shape[1]-1} features (old 20x20 model), but the app is configured for 784 features (new 28x28 MNIST model).\n\n**Solution:**\nThe training script `train.py` is likely still running or hasn't finished generating the new weights.\n\n1. Wait for `python train.py` to finish in your terminal.\n2. Once it says 'Done.', click 'Rerun' in the top right menu.")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Canvas")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=realtime_update,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    
    prediction_placeholder = st.empty()
    probs_placeholder = st.empty()
    debug_image_placeholder = st.empty()
    
    image_data = None
    
    # 1. Check Canvas Input
    if canvas_result.image_data is not None:
        if np.sum(canvas_result.image_data[:, :, 3]) > 0: # Check for content
            image_data = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

    # 2. Check Upload Input
    uploaded_file = st.sidebar.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file)
    
    # Process and Predict
    if image_data is not None:
        col_btn1, col_btn2 = st.columns([1, 1])
        if col_btn1.button("Predict"):
            try:
                # Preprocess
                X_input, processed_img = preprocess.preprocess_image(image_data)
                
                # Predict
                p, prob = model.predict_with_confidence(X_input, Theta1, Theta2)
                
                # Store in session state
                st.session_state['last_prediction'] = p[0]
                st.session_state['last_confidence'] = prob[0]
                st.session_state['last_image'] = processed_img
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.text(traceback.format_exc())

        # Show result if exists
        if 'last_prediction' in st.session_state:
            prediction_placeholder.markdown(f"## Predicted Digit: **{st.session_state['last_prediction']}**")
            
            # Confidence bar
            conf = st.session_state['last_confidence']
            probs_placeholder.progress(float(conf))
            probs_placeholder.write(f"Confidence: {conf*100:.2f}%")
            
            # Debug View
            st.markdown("### Processed Input (28x28)")
            debug_image_placeholder.image(st.session_state['last_image'], width=140, clamp=True)
            
    else:
        # Clear state if processed image input is gone
        if 'last_prediction' in st.session_state:
            del st.session_state['last_prediction']
            del st.session_state['last_confidence']
            del st.session_state['last_image']
        prediction_placeholder.info("Draw a digit and click Predict.")

st.markdown("---")
st.caption("Training based on MNIST Dataset (28x28 resolution).")
