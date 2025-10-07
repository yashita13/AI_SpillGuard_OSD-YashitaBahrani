import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import requests
import os
import cv2
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(page_title="AI SpillGuard - Final", page_icon="🌊", layout="wide")

# --- Model Configuration ---
MODEL_FILENAME = "multiclass_checkpoint.pth"
# GITHUB RELEASE LINK FOR THE 'multiclass_checkpoint.pth' FILE
MODEL_URL = "https://github.com/yashita13/AI_SpillGuard_OSD-YashitaBahrani/releases/download/v1.0.0/multiclass_checkpoint.pth" 
NUM_CLASSES = 4
OIL_CLASS_INDEX = 3 # The index we assigned to the 'pink' oil class

# --- Model Loading & Preprocessing ---
@st.cache_resource
def load_model():
    """Downloads and loads the final multi-class PyTorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading model... Please wait.")
        try:
            r = requests.get(MODEL_URL, allow_redirects=True)
            with open(MODEL_FILENAME, 'wb') as f: f.write(r.content)
            st.success("Model downloaded!")
        except Exception as e:
            st.error(f"Error downloading model: {e}"); return None, None
            
    try:
        checkpoint = torch.load(MODEL_FILENAME, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device); model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model weights: {e}"); return None, None

def preprocess_image(image):
    """Prepares a PIL Image for the multi-class model."""
    image_resized = image.resize((256, 256)); img_array = np.array(image_resized).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5]); std = np.array([0.5, 0.5, 0.5]); img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, image_resized

# --- Main Application UI ---
st.title("SpillGuard - AI Oil Spill Detection System")
st.markdown("Upload a satellite image to detect potential oil spills. The model processes at 256x256 and scales results back to the original size.")

model, device = load_model()

if model:
    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_file = st.file_uploader("Upload SAR Image", type=["jpg", "jpeg", "png", "tif"])
        detection_threshold = st.slider("Detection Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        run_detection = st.button("Detect Oil Spills", use_container_width=True)
        st.markdown("---")

    # --- Main Panel for Display ---
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        # Display the uploaded image at the top
        st.subheader("Uploaded Image")
        st.image(original_image, caption=f"Original Image ({original_image.width}x{original_image.height})", use_container_width=True)

        if run_detection:
            with st.spinner("Running analysis..."):
                # Prediction Logic
                input_tensor, processed_image = preprocess_image(original_image)
                with torch.no_grad():
                    logits = model(input_tensor.to(device))
                    probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                    oil_prob_map = probabilities[OIL_CLASS_INDEX, :, :]
                binary_mask = (oil_prob_map > detection_threshold).astype(np.uint8)
                
                # Generate Plots
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                plt.suptitle('SpillGuard Oil Spill Detection Results', fontsize=16)
                axs[0, 0].imshow(processed_image.convert('L'), cmap='gray'); axs[0, 0].set_title(f'Input Image ({processed_image.width}x{processed_image.height})'); axs[0, 0].axis('off')
                im = axs[0, 1].imshow(oil_prob_map, cmap='hot'); axs[0, 1].set_title('Oil Spill Probability'); axs[0, 1].axis('off'); fig.colorbar(im, ax=axs[0, 1])
                axs[1, 0].imshow(binary_mask, cmap='gray'); axs[1, 0].set_title(f'Detection (Threshold: {detection_threshold})'); axs[1, 0].axis('off')
                overlay_np = np.array(processed_image); color_mask = np.zeros_like(overlay_np); color_mask[binary_mask == 1] = [255, 0, 0]
                overlay_display = cv2.addWeighted(overlay_np, 1, color_mask, 0.4, 0); axs[1, 1].imshow(overlay_display); axs[1, 1].set_title('Oil Spill Overlay'); axs[1, 1].axis('off')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Display plots and detailed analysis side-by-side below the main image 
            st.markdown("---")
            st.header("Analysis Results")
            col1, col2 = st.columns([6, 4]) # Make the plot column wider than the text column

            with col1:
                st.pyplot(fig)

            with col2:
                # Using columns for a clean, aligned layout 
                st.subheader("📊 Detection Analysis")
                st.markdown("---")

                # --- Image Information Section ---
                st.markdown("#### Image Information")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Original Size:**")
                    st.write("**Processing Size:**")
                with c2:
                    st.write(f"{original_image.width} x {original_image.height} px")
                    st.write(f"{processed_image.width} x {processed_image.height} px")
                st.markdown("<br>", unsafe_allow_html=True)

                # --- Detection Statistics Section ---
                total_pixels = original_image.width * original_image.height
                spill_pixel_count = int(np.sum(binary_mask) * (total_pixels / binary_mask.size))
                oil_coverage = (spill_pixel_count / total_pixels) * 100
                max_confidence = np.max(oil_prob_map) if spill_pixel_count > 0 else 0
                avg_probability = np.mean(oil_prob_map[oil_prob_map > 0.1]) if spill_pixel_count > 0 else 0
                
                st.markdown("#### Detection Statistics")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Oil Coverage:**")
                    st.write("**Oil Pixels:**")
                    st.write("**Max Confidence:**")
                    st.write("**Avg. Probability:**")
                with c2:
                    st.write(f"{oil_coverage:.2f}%")
                    st.write(f"{spill_pixel_count:,} / {total_pixels:,}")
                    st.write(f"{max_confidence:.3f}")
                    st.write(f"{avg_probability:.3f}")
                st.markdown("<br>", unsafe_allow_html=True)

                # --- Final Assessment Section ---
                severity = "LOW"
                if oil_coverage > 5: severity = "MEDIUM"
                if oil_coverage > 20: severity = "HIGH"
                if oil_coverage > 50: severity = "CRITICAL"
                
                st.markdown("#### Final Assessment")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Threshold Used:**")
                    st.write("**Calculated Severity:**")
                with c2:
                    st.write(f"{detection_threshold}")
                    st.write(f"{severity}")
                st.markdown("<br>", unsafe_allow_html=True)

                # --- Final Status ---
                if oil_coverage > 0.1:
                    st.error("### Status: 🚨 OIL SPILL DETECTED")
                else:
                    st.success("### Status: ✅ NO SPILL DETECTED")
else:
    st.warning("Model is not loaded. Please ensure the model URL in the script is correct and accessible.")
