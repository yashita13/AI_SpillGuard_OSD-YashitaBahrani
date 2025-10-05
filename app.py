import streamlit as st
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os

st.set_page_config(layout="wide", page_title="AI SpillGuard")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    
    # The app will look for the model file in the same repository
    checkpoint_path = "improved_model_checkpoint.pth" 
    
    if not os.path.exists(checkpoint_path):
        st.error("Model checkpoint file not found in the repository!")
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

    st.header("Analysis Results")
    col1, col2 = st.columns(2)
    col1.image(original_image, caption='Uploaded Image', width='stretch')

    with st.spinner('Analyzing image...'):
        transform = A.Compose([A.Resize(256, 256), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
        input_tensor = transform(image=original_image)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = (torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy() > 0.9).astype(np.uint8)

        kernel = np.ones((7, 7), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        
        is_spill = np.sum(pred_mask) > (pred_mask.size * 0.015)

        with col2:
            if is_spill:
                st.error("### Oil Spill Detected!")
            else:
                st.success("### No Spill Detected")
            
            pred_mask_resized = cv2.resize(pred_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros_like(original_image)
            color_mask[pred_mask_resized == 1] = [255, 0, 0]
            overlay_image = cv2.addWeighted(original_image, 1, color_mask, 0.4, 0)
            st.image(overlay_image, caption='Detection Overlay', width='stretch')

elif model is None:
    st.header("Model Not Loaded")