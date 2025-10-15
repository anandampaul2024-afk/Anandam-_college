import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import HybridResNetViT
from llmhelper1 import generate_explanation
import json
import os

st.title("Football Image Analysis")

uploaded_file = st.file_uploader("Choose a football image...", type=["jpg", "png", "jpeg"])

json_file_path = r"C:\Users\T8635\Desktop\project3\weights\class_index.json"

# Load class names from class_index.json
with open(json_file_path, 'r') as f:
    data = json.load(f)
categories = sorted(list(data['class_to_idx'].keys()))
label_to_index = data['class_to_idx']
index_to_label = {v: k for k, v in label_to_index.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(categories)
weights_path = r"C:\Users\T8635\Desktop\project3\weights\last_hybrid_resnet_vit.pth"
model = HybridResNetViT(num_classes=num_classes)
model.load_state_dict(torch.load(weights_path, map_location=device))  # load trained weights
model.to(device)
model.eval()

pred_class = None  # default

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width= "stretch")

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # CNN prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_class = index_to_label[pred_idx.item()]

    st.write(f"**model Prediction:** {pred_class}")

manual_input = st.text_input("Or enter a football situation manually:")

if st.button("Generate Explanation"):
    if manual_input.strip():
        # User typed their own input
        prompt = f"Provide a detailed football analysis for the following scenario: {manual_input}"
    elif pred_class:
        # Use CNN prediction
        prompt = f"Provide a detailed football analysis for the following scenario: {pred_class} detected in the image."
    else:
        prompt = None

    if prompt:
        explanation = generate_explanation(prompt)
        st.subheader("LLM Explanation")
        st.write(explanation)
    else:
        st.warning("Please upload an image or enter text to get an explanation.")