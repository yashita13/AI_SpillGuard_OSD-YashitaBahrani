import streamlit as st
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
import requests

st.set_page_config(layout="wide", page_title="AI SpillGuard")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    
    # The app will look for the model file in the same repository
    # --- THIS IS THE NEW LOGIC ---
    checkpoint_path = "improved_model_checkpoint.pth"
    MODEL_URL = "https://github.com/yashita13/AI_SpillGuard_OSD-YashitaBahrani/releases/download/v1.0.0/improved_model_checkpoint.pth" # <--- IMPORTANT

    # Download the model file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        st.info("Downloading the model file... Please wait.")
        try:
            r = requests.get(MODEL_URL, allow_redirects=True)
            with open(checkpoint_path, 'wb') as f:
                f.write(r.content)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None, None
    
    if not os.path.exists(checkpoint_path):
        st.error("Model checkpoint file could not be downloaded!")
        return None, None
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

st.sidebar.title("🌊 AI SpillGuard")
st.sidebar.info(
    "This application uses a U-Net deep learning model to detect and segment oil spills "
    "from satellite imagery. Upload an image to begin the analysis."
)

st.title("Real-Time Oil Spill Detection System")
uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    with st.spinner('Analyzing image...'):
        # --- 1. Standard Prediction ---
        transform = A.Compose([A.Resize(256, 256), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
        input_tensor = transform(image=original_image)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask_raw = (torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy() > 0.85).astype(np.uint8) # Stricter confidence

        # --- 2. Post-Processing to clean up noise ---
        kernel = np.ones((5, 5), np.uint8)
        pred_mask_processed = cv2.morphologyEx(pred_mask_raw, cv2.MORPH_OPEN, kernel)
        pred_mask_resized = cv2.resize(pred_mask_processed, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- 3. Sanity Check to prevent detecting land as oil ---
        is_spill = False
        spill_pixel_count = np.sum(pred_mask_resized)
        if spill_pixel_count > (pred_mask_resized.size * 0.01): # Area threshold
            # Check the color of the detected region in the original image
            spill_pixels = original_image[pred_mask_resized == 1]
            avg_color = np.mean(spill_pixels, axis=0)
            avg_red, avg_green, avg_blue = avg_color[0], avg_color[1], avg_color[2]
            
            # Heuristic: Land is often greener than it is blue. Oil on water is not.
            if not (avg_green > avg_blue * 1.05): # If not significantly green
                 is_spill = True
            else: # If it is green, override the decision
                 spill_pixel_count = 0 


    # --- 4. NEW and IMPROVED UI Section ---
    st.header("Analysis Results")

    # Display the three key images side-by-side
    col1, col2, col3 = st.columns(3)
    col1.image(original_image, caption='Original Image', width='stretch')
    # Create a black and white mask for the middle column
    bw_mask_display = (pred_mask_resized * 255).astype(np.uint8)
    col2.image(bw_mask_display, caption='Predicted Mask', width='stretch')
    # Create the final overlay
    color_mask = np.zeros_like(original_image); color_mask[pred_mask_resized == 1] = [255, 0, 0]
    overlay_image = cv2.addWeighted(original_image, 1, color_mask, 0.4, 0)
    col3.image(overlay_image, caption='Detection Overlay', width='stretch')
    
    st.markdown("---") # Divider
    
    # Display the detailed analysis results
    if is_spill:
        st.error("### Status: Oil Spill Detected")
    else:
        st.success("### Status: No Spill Detected")

    total_pixels = original_image.shape[0] * original_image.shape[1]
    spill_percentage = (spill_pixel_count / total_pixels) * 100

    st.subheader("Quantitative Analysis")
    metric1, metric2 = st.columns(2)
    metric1.metric(label="Spill Pixels Detected", value=f"{int(spill_pixel_count):,}")
    metric2.metric(label="Image Area Covered by Spill", value=f"{spill_percentage:.2f}%")

elif model is None:

    st.header("Model Not Loaded")


