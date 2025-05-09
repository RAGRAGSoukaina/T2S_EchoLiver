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

# Télécharger les modèles de classification
gdown.download(id="1pBqEID637icNoLhfp9WV_wUikf4NelAX", output="Models/mobil_model.h5", quiet=False)      # Mobile model
gdown.download(id="1e0Pna1PNoUfG363uk4EW7HP6boUnOJyj", output="Models/cnn_simple.h5", quiet=False)      # CNN simple
gdown.download(id="1btGAg9S1_h9GhktEEw3OR_xD_s-jgD4a", output="Models/attention_model.h5", quiet=False) # Attention model

# Télécharger les modèles de segmentation
gdown.download(id="1btH2vmjSXWyUA7JB_ZQ8h2803mp7ei4T", output="Models/deeplab.pth", quiet=False)        # DeepLab
gdown.download(id="1isTNWtdsAnNDt06Tnq0QtyGLXezxl6KN", output="Models/unetpath.pth", quiet=False)       # UNet

# Configuration de la page
st.set_page_config(page_title="Analyse Échographique Hépatique", layout="wide")

# Classe UNet
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

# Authentification simple
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

# Chargement des modèles
@st.cache_resource
def load_models():
    model1 = load_model("Models/cnn_simple.h5")
    model2 = load_model("Models/mobil_model.h5")
    model3 = load_model("Models/attention_model.h5")
    return {"CNN Simple": model1, "Mobile_modèle": model2, "CNN avec Attention": model3}

@st.cache_resource
def load_segmentation_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplab = deeplabv3_resnet50(pretrained=False, num_classes=1)
    deeplab.load_state_dict(torch.load("Models/deeplab.pth", map_location=device))
    deeplab.to(device).eval()
    unet = UNet()
    unet.load_state_dict(torch.load("Models/unetpath.pth", map_location=device), strict=False)
    unet.to(device).eval()
    return {"DeepLabV3": deeplab, "UNet": unet}

models = load_models()
segmentation_models = load_segmentation_models()

# Fonctions
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = cv2.medianBlur(img_array, 3)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def enhance_ultrasound_for_visualization(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.convertScaleAbs(img, alpha=1.2, beta=10)

def segment_image(model, img_pil):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_np = np.array(img_pil)
    input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(input_tensor)
    if isinstance(output, dict): output = output["out"]
    if isinstance(output, (tuple, list)): output = output[0]
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return (mask > 0.5).astype(np.uint8) * 255

# Interface utilisateur
if not check_password(): st.stop()

with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Accueil", "Classification", "Amélioration d'Image", "Segmentation"])

if page == "Accueil":
    st.title("Analyse Échographique Hépatique")
    st.markdown("Bienvenue dans l'application T2S EchoLiver.")

elif page == "Classification":
    st.title("Classification des Tumeurs")
    model_name = st.selectbox("Choisissez un modèle", list(models.keys()))
    file = st.file_uploader("Téléversez une image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Image Téléversée", use_column_width=True)
        pred = models[model_name].predict(preprocess_image(img))
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        st.metric("Classe prédite", ["Benign", "Malignant", "Normal"][pred_class])
        st.metric("Confiance", f"{confidence:.2%}")

elif page == "Amélioration d'Image":
    st.title("Amélioration d'Image")
    file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if file:
        img_arr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced = enhance_ultrasound_for_visualization(orig)
        st.image([orig, enhanced], caption=["Originale", "Améliorée"], width=300)

elif page == "Segmentation":
    st.title("Segmentation")
    model_name = st.selectbox("Modèle de segmentation", list(segmentation_models.keys()))
    file = st.file_uploader("Image à segmenter", type=["jpg", "jpeg", "png"])
    if file:
        img_pil = Image.open(file).convert("RGB")
        st.image(img_pil, caption="Image Originale", use_column_width=True)
        mask = segment_image(segmentation_models[model_name], img_pil)
        st.image(mask, caption="Masque segmenté", use_column_width=True)

