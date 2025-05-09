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

# Télécharger les modèles depuis Google Drive
gdown.download("https://drive.google.com/uc?id=1pBqEID637icNoLhfp9WV_wUikf4NelAX", "Models/cnn_simple.h5", quiet=False)
gdown.download("https://drive.google.com/uc?id=1e0Pna1PNoUfG363uk4EW7HP6boUnOJyj", "Models/mobil_model.h5", quiet=False)
gdown.download("https://drive.google.com/uc?id=1btGAg9S1_h9GhktEEw3OR_xD_s-jgD4a", "Models/attention_model.h5", quiet=False)
gdown.download("https://drive.google.com/uc?id=1btH2vmjSXWyUA7JB_ZQ8h2803mp7ei4T", "Models/deeplab.pth", quiet=False)
gdown.download("https://drive.google.com/uc?id=1isTNWtdsAnNDt06Tnq0QtyGLXezxl6KN", "Models/unetpath.pth", quiet=False)

# Configuration de la page: doit être le premier appel Streamlit
st.set_page_config(page_title="Analyse Échographique Hépatique", layout="wide")

# Définition de la classe UNet (architecture standard)
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
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (f"{name}_conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                (f"{name}_norm1", nn.BatchNorm2d(features)),
                (f"{name}_relu1", nn.ReLU(inplace=True)),
                (f"{name}_conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                (f"{name}_norm2", nn.BatchNorm2d(features)),
                (f"{name}_relu2", nn.ReLU(inplace=True)),
            ])
        )

# Authentification simple
def check_password():
    def login_form():
        with st.form("login"):
            st.image("/content/drive/MyDrive/Liver/Logo.jpg", width=480)
            st.markdown("<h3 style='text-align: center;'>Connexion à T2S EchoLiver</h3>", unsafe_allow_html=True)
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

# Charger les modèles de classification
@st.cache_resource
def load_models():
    model1 = load_model("/content/drive/MyDrive/Liver/MODELS/CLASSIFICATION/cnn_simple.h5")
    model2 = load_model("/content/drive/MyDrive/Liver/MODELS/CLASSIFICATION/mobil_model.h5")
    model3 = load_model("/content/drive/MyDrive/Liver/MODELS/CLASSIFICATION/attention_model.h5")
    return {"CNN Simple": model1, "Mobile_modèle": model2, "CNN avec Attention": model3}

# Charger les modèles de segmentation
@st.cache_resource
def load_segmentation_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DeepLabV3
    model_deeplab = deeplabv3_resnet50(pretrained=False, num_classes=1)
    state_dict_deeplab = torch.load(
        "/content/drive/MyDrive/Liver/MODELS/SEGMENTATION/model_deeplabv3.pth",
        map_location=device
    )
    model_deeplab.load_state_dict(state_dict_deeplab)
    model_deeplab.to(device).eval()

    # UNet
    model_unet = UNet(in_channels=3, out_channels=1)
    state_dict_unet = torch.load(
        "/content/drive/MyDrive/Liver/MODELS/SEGMENTATION/model_unet.pth",
        map_location=device
    )
    model_unet.load_state_dict(state_dict_unet, strict=False)  # Charger en mode non-strict pour ignorer les clés incompatibles
    model_unet.to(device).eval()

    return {"DeepLabV3": model_deeplab, "UNet": model_unet}

models = load_models()
segmentation_models = load_segmentation_models()

# Fonctions utilitaires
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = cv2.medianBlur(img_array, 3)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def enhance_ultrasound_for_visualization(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_clahe, -1, kernel)
    img_enhanced = cv2.convertScaleAbs(img_sharpened, alpha=1.2, beta=10)
    return img_enhanced

def plot_prediction(prediction, class_names):
    fig, ax = plt.subplots()
    bars = ax.bar(class_names.values(), prediction[0])
    ax.set_ylabel('Probabilité')
    ax.set_title('Probabilités par classe')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}', ha='center', va='bottom')
    return fig

def segment_image(model, img_pil):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_np = np.array(img_pil)  # Convertir PIL -> NumPy
    augmented = transform(image=image_np)
    input_tensor = augmented["image"].unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, dict) and 'out' in output:
        output = output['out']
    elif isinstance(output, (tuple, list)):
        output = output[0]

    mask_tensor = torch.sigmoid(output)
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


# Vérifier l'authentification
if not check_password():
    st.stop()

# Barre latérale de navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Accueil", "Classification", "Amélioration d'Image", "Segmentation"])
    st.markdown("---")
    st.info("""**À propos**
Cette application permet d'analyser les images échographiques du foie à l'aide d'algorithmes avancés.
""")

# Pages de l'application
if page == "Accueil":
    st.title("Analyse Échographique Hépatique")
    st.image("/content/drive/MyDrive/Liver/Logo.jpg", width=600)
    st.markdown("""## Bienvenue dans l'application d'analyse échographique du foie""")
elif page == "Classification":
    st.title("Classification des tumeurs ultrasoniques")
    st.write("Sélectionnez un modèle et téléversez une image pour la classification.")
    selected_model = st.selectbox("Choisissez le modèle à utiliser", list(models.keys()))
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image téléversée", use_column_width=True)
        class_names = {0: "Benign", 1: "Malignant", 2: "Normal"}
        with st.spinner(f"Prédiction en cours avec {selected_model}..."):
            prediction = models[selected_model].predict(preprocess_image(img))
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classe prédite", class_names[pred_class])
            st.metric("Confiance", f"{confidence:.2%}")
        with col2:
            st.pyplot(plot_prediction(prediction, class_names))
        if confidence < 0.7:
            st.warning("⚠️ La confiance de prédiction est faible.")
elif page == "Amélioration d'Image":
    st.title("Amélioration des Images Échographiques")
    st.write("Téléversez une image pour améliorer sa qualité visuelle.")
    file = st.file_uploader("Choisissez une image à améliorer", type=["jpg", "jpeg", "png"] )
    if file:
        arr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enhanced = enhance_ultrasound_for_visualization(orig)
        c1, c2 = st.columns(2)
        c1.image(orig, caption="Originale", use_column_width=True)
        c2.image(enhanced, caption="Améliorée", use_column_width=True, clamp=True)
        buf = BytesIO(); Image.fromarray(enhanced).save(buf, format="JPEG")
        st.download_button("Télécharger", data=buf.getvalue(), file_name="enhanced.jpg")
else:
    # Segmentation
    st.title("Segmentation des Images Échographiques")
    st.write("Sélectionnez un modèle et téléversez une image.")
    seg_model_name = st.selectbox("Modèle de segmentation", list(segmentation_models.keys()))
    f = st.file_uploader("Image à segmenter", type=["jpg","png","jpeg"], key="seg")
    if f:
        img_pil = Image.open(f).convert("RGB")
        st.image(img_pil, caption="Originale", use_column_width=True)
        mask = segment_image(segmentation_models[seg_model_name], img_pil)
        st.image(mask, caption="Masque", use_column_width=True, clamp=True)
        buf2 = BytesIO(); Image.fromarray(mask).save(buf2, format="PNG")
        st.download_button("Télécharger le masque", data=buf2.getvalue(), file_name="mask.png")
