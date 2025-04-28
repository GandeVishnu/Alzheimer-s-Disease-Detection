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
from io import BytesIO

# -------------------- MongoDB Setup --------------------
MONGO_URL = "mongodb+srv://gandevishnu2002:AllCHcrwT8kP1ocf@alzheimersdiseasedetect.oizmrdg.mongodb.net/"
client = MongoClient(MONGO_URL)
db = client["AlzheimersDiseaseDetection"]
users_collection = db["users"]
applications_collection = db["applications"]

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Alzheimer's Disease Detection", page_icon="🧠")
MODEL_PATH = r"F:\ADNI_5_FINAL_FOLDER\20_04_2025_ADNI_best_model.keras"
IMG_SIZE = (224, 224)
class_labels = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

# -------------------- Model Loading --------------------
@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)

model = load_prediction_model()

# -------------------- Image Preprocessing and Prediction --------------------
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
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

def decode_image(encoded_image):
    decoded = base64.b64decode(encoded_image)
    buffer = BytesIO(decoded)
    image = Image.open(buffer)
    return image

# -------------------- MongoDB Functions --------------------
def save_user(email, name, password):
    user = {"email": email, "name": name, "password": password}
    users_collection.insert_one(user)

def load_users():
    users = users_collection.find()
    return {user["email"]: {"name": user["name"], "password": user["password"]} for user in users}

def save_application_form(data):
    applications_collection.insert_one(data)

def get_previous_application(email):
    return applications_collection.find_one(
        {"user_email": email},
        sort=[("submitted_at", -1)]
    )

def get_all_previous_applications(email):
    return list(applications_collection.find({"user_email": email}).sort("submitted_at", -1))

# -------------------- UI Styling --------------------
def add_responsive_styles():
    st.markdown("""
        <style>
            input, textarea {
                caret-color: #000 !important;
                color: #000 !important;
            }
            .stApp {
                background-color: #A8D5E3 !important;
            }
            input[type="text"], input[type="password"], textarea {
                background-color: white !important;
                color: #000000 !important;
                border-radius: 8px !important;
                padding: 10px !important;
                border: 2px solid #0e3c4a !important;
                font-size: 16px !important;
            }
            .title-text {
                font-size: 50px;
                font-weight: bold;
                color: DodgerBlue;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 3px;
                margin-top: 30px;
            }
            .subtitle-text {
                font-size: 25px;
                text-align: center;
                color: Tomato;
                margin-bottom: 20px;
            }
            div.stButton > button {
                width: 100%;
                background-color: #0B5ED7;
                color: white;
                padding: 12px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            div.stButton > button:hover {
                background-color: #084298;
                transition: 0.3s ease;
            }
        </style>
    """, unsafe_allow_html=True)

# -------------------- Pages --------------------
def home_page():
    add_responsive_styles()
    st.markdown('<div class="title-text">🧠 Alzheimer\'s Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Analyze brain MRI scans to predict Alzheimer\'s stages using deep learning.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if col1.button("Login"):
        st.session_state.page = "Login"
        st.toast("✅ Moving to Login Page")
        time.sleep(0.5)
        st.rerun()

    if col2.button("Signup"):
        st.session_state.page = "Signup"
        st.toast("✅ Moving to Signup Page")
        time.sleep(0.5)
        st.rerun()

def login_page():
    add_responsive_styles()
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    users = load_users()

    if st.button("Login"):
        if email in users and users[email]["password"] == password:
            st.toast("✅ Login Successful! Redirecting...")
            time.sleep(0.5)
            st.session_state.Name = users[email]["name"]
            st.session_state.Email = email
            st.session_state.page = "guidelines"
            st.rerun()
        else:
            st.error("Invalid email or password.")

    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.toast("✅ Returning to Home")
        time.sleep(0.5)
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
            st.toast("✅ Signup Successful! Returning to Home...")
            time.sleep(0.5)
            st.session_state.page = "Home"
            st.rerun()

    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.toast("✅ Returning to Home")
        time.sleep(0.5)
        st.rerun()

def guidelines_page():
    add_responsive_styles()
    st.markdown(f"<h1 style='color: DodgerBlue;'>Welcome, {st.session_state.get('Name', 'User')}!</h1>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style="color: Tomato;">📋 Understanding Alzheimer's Disease</h2>
        <ul>
            <li><b>Final CN JPEG:</b> Cognitively Normal – No cognitive impairment.</li>
            <li><b>Final EMCI JPEG:</b> Early Mild Cognitive Impairment – Subtle memory lapses.</li>
            <li><b>Final MCI JPEG:</b> Mild Cognitive Impairment – Includes early and late stages.</li>
            <li><b>Final LMCI JPEG:</b> Late Mild Cognitive Impairment – Severe MCI close to AD onset.</li>
            <li><b>Final AD JPEG:</b> Alzheimer's Disease – Significant memory and behavioral changes.</li>
        </ul>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if col1.button("Proceed to Scan"):
        st.session_state.page = "scan"
        st.toast("✅ Redirecting to Scan Page...")
        time.sleep(0.5)
        st.rerun()
    if col2.button("View Previous Scans"):
        st.session_state.page = "previous_scan"
        st.toast("✅ Redirecting to Previous Scans...")
        time.sleep(0.5)
        st.rerun()

def scan_page():
    add_responsive_styles()
    st.title("📊 Alzheimer’s MRI Scan")
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        predicted_label, confidence, _ = predict(image)
        st.success(f"🧠 Prediction: {predicted_label}")
        st.info(f"📈 Confidence: {confidence:.2f}%")

        st.session_state.uploaded_image = image
        st.session_state.prediction_label = predicted_label
        st.session_state.prediction_confidence = confidence

    col1, col2, col3 = st.columns(3)
    if col1.button("⬅ Back"):
        st.session_state.page = "guidelines"
        st.rerun()
    if col2.button("📄 Application Form"):
        st.session_state.page = "application_form"
        st.rerun()
    if col3.button("🚪 Sign Out"):
        st.session_state.page = "Home"
        st.rerun()

# -------------------- (rest continues for previous_scan_page and application_form_page similarly...) --------------------

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
import pymongo.errors

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Alzheimers Disease Detection", page_icon="🧠")

# -------------------- MongoDB Setup --------------------
def initialize_mongo(debug_mode=False):
    try:
        MONGO_URL = st.secrets["mongo"]["url"]
    except KeyError:
        MONGO_URL = "mongodb+srv://gandevishnu2002:AllCHcrwT8kP1ocf@alzheimersdiseasedetect.oizmrdg.mongodb.net/AlzheimersDiseaseDetection?retryWrites=true&w=majority"
        if debug_mode:
            st.warning("Using fallback MongoDB URL for local development.")

    for attempt in range(3):  # Retry 3 times
        try:
            client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=30000)
            server_info = client.server_info()  # Test connection
            if debug_mode:
                st.write("MongoDB Server Info:", server_info)
            db = client["AlzheimersDiseaseDetection"]
            users_collection = db["users"]
            applications_collection = db["applications"]
            if debug_mode:
                st.success("MongoDB connection established!")
            return client, db, users_collection, applications_collection
        except pymongo.errors.ServerSelectionTimeoutError as e:
            st.warning(f"Connection attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            st.error(f"MongoDB connection error: {str(e)}")
            return None, None, None, None
    st.error("Failed to connect to MongoDB after multiple attempts. Please check your connection settings.")
    return None, None, None, None

# Initialize MongoDB connection (set debug_mode=True for detailed output)
mongo_client, db, users_collection, applications_collection = initialize_mongo(debug_mode=False)

# -------------------- Constants --------------------
MODEL_PATH = "20_04_2025_ADNI_best_model.keras"
IMG_SIZE = (224, 224)
class_labels = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

# -------------------- Utility Functions --------------------
@st.cache_resource
def load_prediction_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_prediction_model()

def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image = image.resize(IMG_SIZE)
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(image):
    try:
        img_array = preprocess_image(image)
        if img_array is None:
            return None, None, None
        predictions = model.predict(img_array)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class] * 100
        return class_labels[predicted_class], confidence, predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def encode_image(image):
    try:
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return encoded
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

# -------------------- MongoDB Functions --------------------
def save_user(email, name, password):
    if users_collection is None:
        st.error("Database connection not available.")
        return
    try:
        user = {"email": email, "name": name, "password": password}
        users_collection.insert_one(user)
        st.success("User registered successfully!")
    except pymongo.errors.ServerSelectionTimeoutError:
        st.error("Failed to connect to the database. Please try again later.")
    except pymongo.errors.DuplicateKeyError:
        st.error("User with this email already exists!")
    except Exception as e:
        st.error(f"An error occurred while saving user: {e}")

def load_users():
    if users_collection is None:
        st.error("Database connection not available.")
        return {}
    try:
        users = users_collection.find()
        return {user["email"]: {"name": user["name"], "password": user["password"]} for user in users}
    except pymongo.errors.ServerSelectionTimeoutError:
        st.error("Failed to connect to the database. Please check your MongoDB configuration.")
        return {}
    except Exception as e:
        st.error(f"An error occurred while loading users: {e}")
        return {}

def save_application_form(data):
    if applications_collection is None:
        st.error("Database connection not available.")
        return
    try:
        applications_collection.insert_one(data)
        st.success("Application form saved successfully!")
    except pymongo.errors.ServerSelectionTimeoutError:
        st.error("Failed to connect to the database. Please try again later.")
    except Exception as e:
        st.error(f"An error occurred while saving application: {e}")

# -------------------- Pages --------------------
def add_responsive_styles():
    st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        @media (max-width: 600px) {
            .stButton>button {
                font-size: 14px;
            }
        }
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

    if st.button("Signup"):
        if not name or not email or not password or not confirm_password:
            st.error("All fields are required.")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            save_user(email, name, password)
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
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            predicted_label, confidence, _ = predict(image)
            if predicted_label and confidence:
                st.success(f"Prediction: {predicted_label}")
                st.info(f"Confidence: {confidence:.2f}%")

                st.session_state["uploaded_image"] = image
                st.session_state["prediction_label"] = predicted_label
                st.session_state["prediction_confidence"] = confidence10
                st.session_state["uploaded_image"] = image
                st.session_state["prediction_label"] = predicted_label
                st.session_state["prediction_confidence"] = confidence
            else:
                st.error("Failed to process the image. Please try again.")
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")

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
            try:
                form_data = {
                    "user_email": st.session_state.get("Email", ""),
                    "name": name,
                    "age": age,
                    "place": place,
                    "phone_number": phone_number,
                    "prediction": prediction_label,
                    "confidence": prediction_confidence,
                    "image_base64": encode_image(uploaded_image) if uploaded_image else None,
                    "submitted_at": datetime.now()
                }
                save_application_form(form_data)
            except Exception as e:
                st.error(f"Error submitting form: {e}")
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
