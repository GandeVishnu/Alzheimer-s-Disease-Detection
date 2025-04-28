import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from fpdf import FPDF
import base64
import time
from datetime import datetime
from pymongo import MongoClient

# -------------------- MongoDB Setup --------------------
MONGO_URL = "mongodb+srv://gandevishnu2002:AllCHcrwT8kP1ocf@alzheimersdiseasedetect.oizmrdg.mongodb.net/"   # or your Atlas URL
client = MongoClient(MONGO_URL)
db = client["AlzheimersDiseaseDetection"]    # Database
users_collection = db["users"]   # Users collection
applications_collection = db["applications"]   # Application form collection

# -------------------- Constants --------------------
page_title = "Alzheimers Disease Detection"
page_icon = "🧠"
MODEL_PATH = r"F:\ADNI_5_FINAL_FOLDER\20_04_2025_ADNI_best_model.keras"
IMG_SIZE = (224, 224)
class_labels = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title=page_title, page_icon=page_icon)

# -------------------- Utility Functions --------------------
@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)

model = load_prediction_model()

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100
    return class_labels[predicted_class], confidence, predictions

def encode_image(image):
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

# -------------------- MongoDB Functions --------------------
def save_user(email, name, password):
    user = {"email": email, "name": name, "password": password}
    users_collection.insert_one(user)

def load_users():
    users = users_collection.find()
    return {user["email"]: {"name": user["name"], "password": user["password"]} for user in users}

def save_application_form(data):
    applications_collection.insert_one(data)

# -------------------- Pages --------------------

def add_responsive_styles():
    st.markdown("""
    <style>
    /* Your CSS styles here, same as before */
    </style>
    """, unsafe_allow_html=True)

def home_page():
    add_responsive_styles()
    st.title("🧠 Alzheimer’s Disease Detection")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            st.session_state.page = "Login"
            st.rerun()
    with col2:
        if st.button("Signup"):
            st.session_state.page = "Signup"
            st.rerun()

def login_page():
    add_responsive_styles()
    st.subheader("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    users = load_users()

    if st.button("Login"):
        if email in users and users[email]["password"] == password:
            st.success("Login Successful!")
            st.session_state["Name"] = users[email]["name"]
            st.session_state["Email"] = email
            st.session_state.page = "Guidelines"
            st.rerun()
        else:
            st.error("Invalid email or password.")

    if st.button("⬅ Back"):
        st.session_state.page = "Home"
        st.rerun()

def signup_page():
    add_responsive_styles()
    st.subheader("📝 Signup")
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Re-enter Password", type="password")

    users = load_users()

    if st.button("Signup"):
        if not name or not email or not password or not confirm_password:
            st.error("All fields are required.")
        elif email in users:
            st.error("User already exists!")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            save_user(email, name, password)
            st.success("Signup Successful! Please login.")
            st.session_state.page = "Home"
            st.rerun()

    if st.button("⬅ Back"):
        st.session_state.page = "Home"
        st.rerun()

def guidelines_page():
    add_responsive_styles()
    st.title(f"Welcome, {st.session_state.get('Name', 'User')}")
    st.markdown("📋 Read about Alzheimer’s stages...")
    if st.button("Proceed to Scan"):
        st.session_state.page = "Scan"
        st.rerun()

def scan_page():
    add_responsive_styles()
    st.title("📊 Alzheimer’s MRI Scan")
    uploaded_file = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_label, confidence, _ = predict(image)
        st.success(f"Prediction: {predicted_label}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.session_state["uploaded_image"] = image
        st.session_state["prediction_label"] = predicted_label
        st.session_state["prediction_confidence"] = confidence

    if st.button("➡ Application Form"):
        if "uploaded_image" in st.session_state:
            st.session_state.page = "ApplicationForm"
            st.rerun()
        else:
            st.error("Please upload an image first.")

def application_form_page():
    add_responsive_styles()
    st.title("📝 Application Form")

    name = st.text_input("Name")
    age = st.text_input("Age")
    place = st.text_input("Place")
    phone_number = st.text_input("Phone Number")

    uploaded_image = st.session_state.get("uploaded_image", None)
    prediction_label = st.session_state.get("prediction_label", "N/A")
    prediction_confidence = st.session_state.get("prediction_confidence", 0.0)

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded MRI", use_column_width=True)

    if st.button("📥 Submit Form & Save to Database"):
        if name and age and place and phone_number:
            form_data = {
                "user_email": st.session_state.get("Email", ""),
                "name": name,
                "age": age,
                "place": place,
                "phone_number": phone_number,
                "prediction": prediction_label,
                "confidence": prediction_confidence,
                "image_base64": encode_image(uploaded_image),
                "submitted_at": datetime.now()
            }
            save_application_form(form_data)
            st.success("Application form and scan successfully saved!")
        else:
            st.error("Please fill all the fields.")

# -------------------- App Navigator --------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Signup":
    signup_page()
elif st.session_state.page == "Guidelines":
    guidelines_page()
elif st.session_state.page == "Scan":
    scan_page()
elif st.session_state.page == "ApplicationForm":
    application_form_page()
