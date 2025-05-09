import albumentations as A
from albumentations.pytorch import ToTensorV2

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from collections import OrderedDict
from io import BytesIO
import os
import gdown

# Créer un dossier pour stocker les modèles
os.makedirs("Models", exist_ok=True)

# Télécharger les modèles depuis Google Drive (liens publics)
gdown.download("https://drive.google.com/uc?id=1pBSrDk6D5HAIZCn2ujgXm5iTEgpKkoR7", "Models/cnn_simple.h5", quiet=False)
gdown.download("https://drive.google.com/uc?id=1WfPfVSqG77XciEwCz_tswgVoq81aQFWN", "Models/mobil_model.h5", quiet=False)
gdown.download("https://drive.google.com/uc?id=1poQ8l55_oatNQeI-jLzTm0GlJkJoPkzY", "Models/attention_model.h5", quiet=False)
gdown.download("https://drive.google.com/uc?id=1hW540qcpaC4GrDqwqBhW-S2Xp23CAmSI", "Models/deeplab.pth", quiet=False)
gdown.download("https://drive.google.com/uc?id=1paxDAiNLuD7ZU-XiJ8RELms3AzhPyJ0Z", "Models/unetpath.pth", quiet=False)

st.set_page_config(page_title="Analyse Échographique Hépatique", layout="wide")

# Définition de la classe UNet
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.final(dec1)

# Authentification
def check_password():
    def login_form():
        with st.form("login"):
            st.title("Connexion à T2S EchoLiver")
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter")
            if submitted:
                if username == "admin" and password == "admin":
                    st.session_state["logged_in"] = True
                else:
                    st.error("Identifiants incorrects")
    if "logged_in" not in st.session_state:
        login_form()
        return False
    return True

@st.cache_resource
def load_models():
    m1 = load_model("Models/cnn_simple.h5")
    m2 = load_model("Models/mobil_model.h5")
    m3 = load_model("Models/attention_model.h5")
    return {"CNN Simple": m1, "Mobile_modèle": m2, "CNN avec Attention": m3}

@st.cache_resource
def load_segmentation_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = deeplabv3_resnet50(pretrained=False, num_classes=1)
    model1.load_state_dict(torch.load("Models/deeplab.pth", map_location=device))
    model1.to(device).eval()
    model2 = UNet(3, 1)
    model2.load_state_dict(torch.load("Models/unetpath.pth", map_location=device), strict=False)
    model2.to(device).eval()
    return {"DeepLabV3": model1, "UNet": model2}

models = load_models()
segmentation_models = load_segmentation_models()

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = cv2.medianBlur(arr, 3)
    arr = arr / 255.0
    return np.expand_dims(arr, 0)

def enhance_ultrasound_for_visualization(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    img = cv2.createCLAHE(clipLimit=3.0).apply(img)
    img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.convertScaleAbs(img, alpha=1.2, beta=10)

def plot_prediction(prediction, class_names):
    fig, ax = plt.subplots()
    bars = ax.bar(class_names.values(), prediction[0])
    ax.set_ylabel('Probabilité')
    ax.set_title('Probabilités par classe')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2%}', ha='center')
    return fig

def segment_image(model, img_pil):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_np = np.array(img_pil)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict): output = output["out"]
        if isinstance(output, (tuple, list)): output = output[0]
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return (mask > 0.5).astype(np.uint8) * 255

if not check_password():
    st.stop()

# Interface
with st.sidebar:
    page = st.radio("Navigation", ["Accueil", "Classification", "Amélioration d'Image", "Segmentation"])

if page == "Accueil":
    st.title("Analyse Échographique Hépatique")
    st.markdown("Bienvenue dans l'application T2S EchoLiver pour l'analyse des images échographiques du foie.")

elif page == "Classification":
    st.title("Classification des tumeurs ultrasoniques")
    model_choice = st.selectbox("Modèle de classification", list(models.keys()))
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image téléversée", use_column_width=True)
        preds = models[model_choice].predict(preprocess_image(img))
        pred_idx = np.argmax(preds)
        confidence = np.max(preds)
        class_map = {0: "Benign", 1: "Malignant", 2: "Normal"}
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classe prédite", class_map[pred_idx])
            st.metric("Confiance", f"{confidence:.2%}")
        with col2:
            st.pyplot(plot_prediction(preds, class_map))
        if confidence < 0.7:
            st.warning("⚠️ La confiance de prédiction est faible.")

elif page == "Amélioration d'Image":
    st.title("Amélioration des Images Échographiques")
    file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
    if file:
        arr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced = enhance_ultrasound_for_visualization(orig)
        col1, col2 = st.columns(2)
        col1.image(orig, caption="Originale", use_column_width=True)
        col2.image(enhanced, caption="Améliorée", use_column_width=True)
        buf = BytesIO(); Image.fromarray(enhanced).save(buf, format="JPEG")
        st.download_button("Télécharger l'image améliorée", buf.getvalue(), file_name="enhanced.jpg")

elif page == "Segmentation":
    st.title("Segmentation des Images Échographiques")
    model_choice = st.selectbox("Modèle de segmentation", list(segmentation_models.keys()))
    f = st.file_uploader("Image à segmenter", type=["jpg", "png", "jpeg"])
    if f:
        img_pil = Image.open(f).convert("RGB")
        st.image(img_pil, caption="Image Originale", use_column_width=True)
        mask = segment_image(segmentation_models[model_choice], img_pil)
        st.image(mask, caption="Masque Segmenté", use_column_width=True)
        buf = BytesIO(); Image.fromarray(mask).save(buf, format="PNG")
        st.download_button("Télécharger le masque", buf.getvalue(), file_name="mask.png")


