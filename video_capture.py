import cv2
import torch
from PIL import Image
from torchvision import transforms
import json
from model import HybridResNetViT

# --- Load class index ---
json_file_path = r"C:\Users\T8635\Desktop\project3\weights\class_index.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)
label_to_index = data['class_to_idx']
index_to_label = {v: k for k, v in label_to_index.items()}

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_to_index)
weights_path = r"C:\Users\T8635\Desktop\project3\weights\last_hybrid_resnet_vit.pth"
model = HybridResNetViT(num_classes=num_classes)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# --- Video frame extraction ---
video_path = r"C:\Users\T8635\Desktop\project3\blend\scene_005.mp4"
cap = cv2.VideoCapture(video_path)
frame_rate = 1  # frames per second

frames = []
count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if int(fps) > 0 and count % int(fps // frame_rate) == 0:
        frames.append(frame)
    count += 1
cap.release()

# --- Preprocess frames ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

processed_frames = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))).unsqueeze(0) for f in frames]

# --- Predict for each frame with confidence threshold ---
results = []
threshold = 0.9 # Adjust as needed

with torch.no_grad():
    for img_tensor in processed_frames:
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probs, 1)
        if max_prob.item() < threshold:
            pred_class = "No Event"
        else:
            pred_class = index_to_label[pred_idx.item()]
        results.append(pred_class)

# --- Print results ---
print("Predicted classes for each frame:")
print(results)