import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
from PIL import Image
from fpdf import FPDF
import base64
from datetime import datetime
from pymongo import MongoClient
from io import BytesIO

# -------------------- MongoDB Setup --------------------
MONGO_URL = "mongodb+srv://gandevishnu2002:AllCHcrwT8kP1ocf@alzheimersdiseasedetect.oizmrdg.mongodb.net/"
try:
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    client.server_info()  # Test connection
    db = client["AlzheimersDiseaseDetection"]
    users_collection = db["users"]
    applications_collection = db["applications"]
except Exception as e:
    st.error(f"MongoDB Connection Error: {str(e)}")

# -------------------- Page Config --------------------
page_title = "Alzheimers Disease Detection"
page_icon = "🧠"
st.set_page_config(page_title=page_title, page_icon=page_icon)

# -------------------- Model Setup --------------------
MODEL_PATH = "20_04_2025_ADNI_best_model.keras"
IMG_SIZE = (224, 224)
class_labels = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)

model = load_prediction_model()

# -------------------- Image Processing Functions --------------------
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
    application = applications_collection.find_one(
        {"user_email": email},
        sort=[("submitted_at", -1)]
    )
    return application

def get_previous_applications(email):
    applications = applications_collection.find({"user_email": email}).sort("submitted_at", -1)
    return list(applications)

# -------------------- Styling --------------------
def add_responsive_styles():
    bg_color = "#A8D5E3"
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {bg_color} !important;
            }}
            input[type="text"], input[type="password"], textarea {{
                background-color: white !important;
                color: #000000 !important;
                border-radius: 8px !important;
                padding: 10px !important;
                border: 2px solid #0e3c4a !important;
                font-size: 16px !important;
            }}
            .title-text {{
                font-size: 50px;
                font-weight: bold;
                color: DodgerBlue;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 3px;
                margin-top: 30px;
            }}
            .subtitle-text {{
                font-size: 25px;
                text-align: center;
                color: Tomato;
                margin-bottom: 20px;
            }}
            .stButton > button {{
                display: block !important;
                width: 100%;
                background-color: #0B5ED7 !important;
                color: white !important;
                padding: 12px !important;
                font-size: 18px !important;
                font-weight: bold !important;
                border-radius: 8px !important;
                border: none !important;
                margin-top: 10px !important;
            }}
            .stButton > button:hover {{
                background-color: #084298 !important;
                transition: 0.3s ease;
            }}
        </style>
    """, unsafe_allow_html=True)

# -------------------- Page Functions --------------------
def home_page():
    add_responsive_styles()
    st.markdown('<div class="title-text">🧠 Alzheimer\'s Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Analyze brain MRI scans to predict Alzheimer\'s disease stages using advanced deep learning models.</div>', unsafe_allow_html=True)
    st.write("DEBUG: Home page rendered")  # Debugging line
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", key="login_button"):
            st.session_state["page"] = "Login"
            st.rerun()
        st.markdown('<a href="#" onclick="document.getElementById(\'login_button\').click()">Login (Fallback)</a>', unsafe_allow_html=True)
    with col2:
        if st.button("Signup", key="signup_button"):
            st.session_state["page"] = "Signup"
            st.rerun()
        st.markdown('<a href="#" onclick="document.getElementById(\'signup_button\').click()">Signup (Fallback)</a>', unsafe_allow_html=True)

def login_page():
    add_responsive_styles()
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    users = load_users()

    if st.button("Login", key="login_submit"):
        if email in users and users[email]["password"] == password:
            st.session_state["Name"] = users[email]["name"]
            st.session_state["Email"] = email
            st.session_state["page"] = "guidelines"
            st.rerun()
        else:
            st.error("Invalid email or password.")

    if st.button("Back to Home", key="login_back"):
        st.session_state["page"] = "Home"
        st.rerun()

def signup_page():
    add_responsive_styles()
    st.subheader("📝 Signup")
    name = st.text_input("Name", key="signup_name")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Re-enter Password", type="password", key="signup_confirm_password")

    users = load_users()

    if st.button("Signup", key="signup_submit"):
        if not name or not email or not password or not confirm_password:
            st.error("All fields are required.")
        elif email in users:
            st.error("User already exists!")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            save_user(email, name, password)
            st.session_state["page"] = "Home"
            st.rerun()

    if st.button("Back to Home", key="signup_back"):
        st.session_state["page"] = "Home"
        st.rerun()

def guidelines_page():
    add_responsive_styles()
    st.markdown(f"<h1 style='color: DodgerBlue;'>Welcome, {st.session_state.get('Name', 'User')}!</h1>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style="color: Tomato;">📋 What is Alzheimer's Disease?</h2>
        <p style="color: black;">Alzheimer's disease is a progressive brain disorder causing memory loss and cognitive decline.</p>
        <ul>
            <li><span style="color:#0B3D91; font-weight:bold;">Final CN JPEG:</span> 
                <span style="color:#000000;">Cognitively Normal – No cognitive impairment.</span></li>
            <li><span style="color:#0B3D91; font-weight:bold;">Final EMCI JPEG:</span> 
                <span style="color:#000000;">Early Mild Cognitive Impairment – Very mild symptoms, subtle memory lapses.</span></li>
            <li><span style="color:#0B3D91; font-weight:bold;">Final MCI JPEG:</span> 
                <span style="color:#000000;">Mild Cognitive Impairment – General MCI, includes both early and late stages.</span></li>
            <li><span style="color:#0B3D91; font-weight:bold;">Final LMCI JPEG:</span> 
                <span style="color:#000000;">Late Mild Cognitive Impairment – More severe than EMCI, close to AD onset.</span></li>
            <li><span style="color:#0B3D91; font-weight:bold;">Final AD JPEG:</span> 
                <span style="color:#000000;">Alzheimer’s Disease – Advanced cognitive decline, significant memory and behavioral changes.</span></li>        
        </ul>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Proceed to Scan", key="guidelines_scan"):
            st.session_state["page"] = "scan"
            st.rerun()
    with col2:
        if st.button("Previous Scan", key="guidelines_previous"):
            st.session_state["page"] = "previous_scan"
            st.rerun()

def scan_page():
    add_responsive_styles()
    st.title("📊 Alzheimer’s MRI Scan")
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predicted_label, confidence, predictions = predict(image)
        st.markdown(f"### 🟢 Prediction: {predicted_label}")
        st.markdown(f"### 📊 Confidence: {confidence:.2f}%")

        st.session_state["uploaded_image"] = image
        st.session_state["prediction_label"] = predicted_label
        st.session_state["prediction_confidence"] = confidence

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅ Back", key="scan_back"):
            st.session_state["page"] = "guidelines"
            st.rerun()
    with col2:
        if st.button("📄 View Application Form", key="scan_application"):
            st.session_state["page"] = "application_form"
            st.rerun()
    with col3:
        if st.button("🚪 Sign Out", key="scan_signout"):
            st.session_state["page"] = "Home"
            st.rerun()

def previous_scan_page():
    add_responsive_styles()
    st.title("📜 Previous Scan Details")
    email = st.session_state.get("Email", "")
    if not email:
        st.error("Please log in to view previous scans.")
        if st.button("Back to Guidelines", key="previous_back"):
            st.session_state["page"] = "guidelines"
            st.rerun()
        return

    applications = get_previous_applications(email)
    if applications:
        for idx, application in enumerate(applications, 1):
            st.markdown(f"### Scan {idx} - Submitted: {application.get('submitted_at', 'N/A')}")
            st.write(f"**Name:** {application.get('name', 'N/A')}")
            st.write(f"**Age:** {application.get('age', 'N/A')}")
            st.write(f"**Place:** {application.get('place', 'N/A')}")
            st.write(f"**Phone Number:** {application.get('phone_number', 'N/A')}")
            st.write(f"**Prediction:** {application.get('prediction', 'N/A')}")
            st.write(f"**Confidence:** {application.get('confidence', 0.0):.2f}%")

            if "image_base64" in application and application["image_base64"]:
                try:
                    st.subheader(f"MRI Scan {idx}:")
                    image = decode_image(application["image_base64"])
                    st.image(image, caption=f"MRI Image - Scan {idx}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image for scan {idx}: {str(e)}")
            else:
                st.info(f"No MRI image available for scan {idx}.")
            st.markdown("---")
    else:
        st.info("No previous scans found.")

    if st.button("Back to Guidelines", key="previous_guidelines"):
        st.session_state["page"] = "guidelines"
        st.rerun()

def application_form_page():
    add_responsive_styles()
    st.title("📝 Application Form")
    name = st.text_input("Name", key="app_name")
    age = st.text_input("Age", key="app_age")
    place = st.text_input("Place", key="app_place")
    phone_number = st.text_input("Phone Number", key="app_phone")

    uploaded_image = st.session_state.get("uploaded_image", None)
    prediction_label = st.session_state.get("prediction_label", "N/A")
    prediction_confidence = st.session_state.get("prediction_confidence", 0.0)

    if uploaded_image:
        st.subheader("Uploaded MRI Scan:")
        st.image(uploaded_image, caption="MRI Image", use_column_width=True)
        st.subheader("Diagnosis Result:")
        st.write(f"🟢 **Prediction:** {prediction_label}")
        st.write(f"📊 **Confidence:** {prediction_confidence:.2f}%")

    if st.button("📥 Download Report", key="app_download"):
        if name and age and place and phone_number:
            form_data = {
                "user_email": st.session_state.get("Email", ""),
                "name": name,
                "age": age,
                "place": place,
                "phone_number": phone_number,
                "prediction": prediction_label,
                "confidence": float(prediction_confidence),
                "image_base64": encode_image(uploaded_image) if uploaded_image else "",
                "submitted_at": datetime.now()
            }
            save_application_form(form_data)
            st.success("Application form and scan successfully saved!")
        else:
            st.error("Please fill all the fields.")

        if name and age and place and phone_number and uploaded_image:
            temp_image_path = "temp_mri_image.jpg"
            uploaded_image.save(temp_image_path)
            pdf_path = generate_pdf(name, age, place, phone_number, temp_image_path, prediction_label, prediction_confidence)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download",
                    data=pdf_file,
                    file_name="Alzheimer_MRI_Report.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
        else:
            st.warning("⚠ Please fill out all details and upload an image before downloading.")

    if st.button("🔁 Scan Page", key="app_scan"):
        st.session_state["page"] = "scan"
        st.rerun()

def generate_pdf(name, age, place, phone_number, image_path, diagnosis, confidence):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Alzheimer's MRI Scan Report", ln=True, align="C")
    pdf.ln(10)

    current_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Report Generated: {current_datetime}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Patient Details:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Name: {name}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Place: {place}", ln=True)
    pdf.cell(0, 10, f"Phone Number: {phone_number}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Diagnosis Result:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Prediction: {diagnosis}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)

    if image_path:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "MRI Scan:", ln=True)
        pdf.image(image_path, x=60, w=100)
        pdf.ln(10)

    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 10, "This report is generated by the Alzheimer's MRI Analysis System.", ln=True, align="C")
    pdf_filename = "Alzheimer_MRI_Report.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

# -------------------- Main Function --------------------
def main():
    st.write("App Version: 1.0.0")  # Debugging version
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    pages = {
        "Home": home_page,
        "Login": login_page,
        "Signup": signup_page,
        "guidelines": guidelines_page,
        "scan": scan_page,
        "application_form": application_form_page,
        "previous_scan": previous_scan_page
    }
    pages[st.session_state["page"]]()

if __name__ == "__main__":
    main()
