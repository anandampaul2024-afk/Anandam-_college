import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import HybridResNetViT
from llmhelper1 import generate_explanation
import json
import cv2
import tempfile

st.title("Football Event Analysis")

uploaded_file = st.file_uploader("Upload a football image or video...", type=["jpg", "png", "jpeg", "mp4"])

json_file_path = r"C:\Users\T8635\Desktop\project3\weights\class_index.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)
categories = sorted(list(data['class_to_idx'].keys()))
label_to_index = data['class_to_idx']
index_to_label = {v: k for k, v in label_to_index.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(categories)
weights_path = r"C:\Users\T8635\Desktop\project3\weights\last_hybrid_resnet_vit.pth"
model = HybridResNetViT(num_classes=num_classes)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

threshold = 0.5  # Confidence threshold
pred_class = None

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            max_prob, pred_idx = torch.max(probs, 1)
            if max_prob.item() < threshold:
                pred_class = "No Event"
            else:
                pred_class = index_to_label[pred_idx.item()]
        st.write(f"**CNN Prediction:** {pred_class}")

    elif file_type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)
        st.write("Processing video frames...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_rate = 1  # frames per second
        count = 0
        frame_preds = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(fps) > 0 and count % int(fps // frame_rate) == 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    max_prob, pred_idx = torch.max(probs, 1)
                    if max_prob.item() < threshold:
                        pred_class = "No Event"
                    else:
                        pred_class = index_to_label[pred_idx.item()]
                    frame_preds.append(pred_class)
            count += 1
        cap.release()
        st.write("Predicted classes for sampled frames:")
        st.write(frame_preds)

manual_input = st.text_input("Or enter a football situation manually:")

if st.button("Generate Explanation"):
    if manual_input.strip():
        prompt = f"Provide a detailed football analysis for the following scenario: {manual_input}"
    elif pred_class:
        prompt = f"Provide a detailed football analysis for the following scenario: {pred_class} detected in the image/video."
    else:
        prompt = None

    if prompt:
        explanation = generate_explanation(prompt)
        st.subheader("LLM Explanation")
        st.write(explanation)
    else:
        st.warning("Please upload an image/video or enter text to get an explanation.")