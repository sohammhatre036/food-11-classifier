# streamlit_full.py (final improved)
import os, io, json
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Food-11 Classifier", layout="wide")

# ---------------------- config
OUT_DIR = "food_checkpoints"
DATA_ROOT = "C:\\Users\\soham\\Desktop\\Project\\food-11"
IMG_SIZE, BATCH_SIZE = 224, 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- helpers
def load_classes():
    path = os.path.join(OUT_DIR, "classes.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Bread","Dairy product","Dessert","Egg","Fried food","Meat","Noodles-Pasta","Rice","Seafood","Soup","Vegetable-Fruit"]

def build_resnet18(n_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

def load_model(ckpt_path, n_classes):
    model = build_resnet18(n_classes).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.eval()

preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------------- model + classes
classes = load_classes()
n_classes = len(classes)
ckpt_path = os.path.join(OUT_DIR, "best_resnet18.pth")
model = load_model(ckpt_path, n_classes) if os.path.exists(ckpt_path) else None

# ---------------------- layout
tab1, tab2 = st.tabs(["🔮 Predict", "📊 Report"])

# ---------------------- PREDICT
with tab1:
    st.header("Upload image → Predict")
    uploaded = st.file_uploader("Upload food image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded and model:
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with col2:
            inp = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
            top3 = probs.argsort()[::-1][:3]

            st.subheader("Top Predictions")
            for i, idx in enumerate(top3, 1):
                st.write(f"{i}. **{classes[idx]}** — {probs[idx]*100:.2f}%")

# ---------------------- REPORT
with tab2:
    st.header("Training curves & Evaluation report")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training curves")
        if os.path.exists(os.path.join(OUT_DIR, "training_curves.png")):
            st.image(os.path.join(OUT_DIR, "training_curves.png"))
        else:
            st.info("No training_curves.png found")

    with col2:
        st.subheader("Confusion matrix (saved)")
        if os.path.exists(os.path.join(OUT_DIR, "confusion_matrix.png")):
            st.image(os.path.join(OUT_DIR, "confusion_matrix.png"))
        else:
            st.info("No confusion_matrix.png found")

    st.markdown("---")
    st.subheader("Evaluate on real data")

    eval_dir = os.path.join(DATA_ROOT, "evaluation")
    if not os.path.isdir(eval_dir):
        st.warning("No evaluation folder found.")
    elif model is None:
        st.warning("Model not loaded.")
    else:
        if st.button("▶️ Run Evaluation"):
            val_tf = preprocess
            eval_ds = datasets.ImageFolder(eval_dir, transform=val_tf)
            eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

            criterion = nn.CrossEntropyLoss(reduction="sum")
            total_loss, total_samples = 0.0, 0
            all_preds, all_labels = [], []

            prog = st.progress(0)
            with torch.no_grad():
                for i, (xb, yb) in enumerate(eval_loader):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    total_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(yb.cpu().numpy())
                    total_samples += xb.size(0)
                    prog.progress((i+1)/len(eval_loader))

            prog.empty()
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            np.save(os.path.join(OUT_DIR, "preds.npy"), all_preds)
            np.save(os.path.join(OUT_DIR, "labels.npy"), all_labels)

            avg_loss = total_loss / total_samples
            acc = (all_preds == all_labels).mean()

            colm1, colm2 = st.columns(2)
            colm1.metric("Accuracy", f"{acc*100:.2f}%")
            colm2.metric("Avg Loss", f"{avg_loss:.4f}")

            # confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix (Evaluation)")
            ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels(classes)
            for (i,j), val in np.ndenumerate(cm):
                ax.text(j, i, int(val), ha="center", va="center", color="black")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

            # classification report as dataframe
            rep = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
            df = pd.DataFrame(rep).transpose()
            st.dataframe(df.style.format({
                "precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"
            }))
