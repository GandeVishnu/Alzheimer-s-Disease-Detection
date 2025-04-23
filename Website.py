import streamlit as st
import sqlite3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input 
from PIL import Image
import time
from fpdf import FPDF
import base64
from datetime import datetime


DB_FILE = "User_Credentials.db"
MODEL_PATH = "20_04_2025_ADNI_best_model.keras"
IMG_SIZE = (224, 224)
class_labels = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

def initialize_db():
    if not os.path.exists(DB_FILE):
        create_db()

def create_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def load_users():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT email, name, password FROM users")
    users = cursor.fetchall()
    conn.close()
    return {user[0]: {"name": user[1], "password": user[2]} for user in users}

def save_user(email, name, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (email, name, password) VALUES (?, ?, ?)", (email, name, password))
    conn.commit()
    conn.close()

initialize_db()

def add_responsive_styles():
    bg_color = "#A8D5E3"  

    st.markdown(f"""
        <style>
            /* Set the background color */
            .stApp {{
                background-color: {bg_color} !important;
            }}
            input[type="text"], input[type="password"], textarea {{
                background-color: white !important;  /* White background */
                color: #000000 !important;  /* Black text */
                border-radius: 8px !important;
                padding: 10px !important;
                border: 2px solid #0e3c4a !important;
                font-size: 16px !important;
            }}

            /* Customize title text */
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

            div.stButton > button {{
                width: 100%;
                background-color: #0B5ED7;
                color: white;
                padding: 12px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }}
            div.stButton > button:hover {{
                background-color: #084298;
                transition: 0.3s ease;
            }}


        </style>
    """, unsafe_allow_html=True)

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

def home_page():
    add_responsive_styles()
    st.markdown('<div class="title-text">🧠 Alzheimer\'s Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Analyze brain MRI scans to predict Alzheimer\'s disease stages using advanced deep learning models.</div>', unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            st.session_state["page"] = "Login"
            st.toast("✅ Moving to Login Page", icon="✅")            
            st.rerun()
    with col2:
        if st.button("Signup"):
            st.session_state["page"] = "Signup"
            st.toast("✅ Moving to Signup Page", icon="✅")            
            st.rerun()

def login_page():
    add_responsive_styles()
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")  
    users = load_users()

    if st.button("Login"):
        if email in users and users[email]["password"] == password:
            st.toast("✅ Login Successful! Redirecting...", icon="✅")
            time.sleep(1)  
            st.session_state["Name"] = users[email]["name"]
            st.session_state["page"] = "guidelines"
            st.rerun()
        else:
            st.error("Invalid email or password.")
    
    if st.button("Back to Home"):
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
    
    if st.button("Signup"):
        if not name or not email or not password or not confirm_password:
            st.error("All fields are required.")
        elif email in users:
            st.error("User already exists!")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            save_user(email, name, password)
            st.toast("✅ Signup Successful! Redirecting to Home...", icon="✅")
            time.sleep(1)  
            st.session_state["page"] = "Home"
            st.rerun()

    if st.button("Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()

def guidelines_page():
    add_responsive_styles()
    st.markdown(f"<h1 style='color: DodgerBlue;'>Welcome, {st.session_state.get('Name', 'User')}!</h1>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style="color: Tomato;">📋 What is Alzheimer's Disease?</h2>
    <p style="color: black;">Alzheimer's disease is a progressive brain disorder causing memory loss and cognitive decline.</p>

    <ul>
        <li><span style="color:#0B3D91; font-weight:bold;">Non-Demented:</span> 
            <span style="color:#000000;">Early signs like slight memory lapses and difficulty in recalling words.</span>
        </li>
        <li><span style="color:#0B3D91; font-weight:bold;">Mild Demented:</span> 
            <span style="color:#000000;">Noticeable memory loss, difficulty in recognizing familiar people, and impaired problem-solving skills.</span>
        </li>
        <li><span style="color:#0B3D91; font-weight:bold;">Moderate Demented:</span> 
            <span style="color:#000000;">Severe cognitive decline, loss of independence, confusion, and difficulty in basic activities like speaking and walking.</span>
        </li>
        <li><span style="color:#0B3D91; font-weight:bold;">Very Mild Demented:</span> 
            <span style="color:#000000;">Increased forgetfulness, trouble with organization, and minor confusion in daily tasks.</span>
        </li>
    </ul>
       
    """, unsafe_allow_html=True)
    
    if st.button("Proceed to Scan"):
        st.session_state["page"] = "scan"
        st.rerun()

def scan_page():
    add_responsive_styles()
    st.title(f"📊 Alzheimer’s MRI Scan")
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        predicted_label, confidence, predictions = predict(image)
        st.markdown(f"### 🟢 Prediction: {predicted_label}")
        st.markdown(f"### 📊 Confidence: {confidence:.2f}%")

        st.session_state["uploaded_image"] = image
        st.session_state["prediction_label"] = predicted_label
        st.session_state["prediction_confidence"] = confidence
        
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        if st.button("⬅ Back"):
            st.session_state["page"] = "guidelines"
            st.rerun()

    with col2:
        if st.button("📄 View Application Form"):
            st.session_state["page"] = "application_form"  
            st.rerun()

    with col3:
        if st.button("🚪 Sign Out"):
            st.session_state["page"] = "Home"
            st.rerun()

def application_form_page():
    add_responsive_styles()
    st.title("📝 Application Form")

    uploaded_image = st.session_state.get("uploaded_image", None)
    prediction_label = st.session_state.get("prediction_label", "N/A")
    prediction_confidence = st.session_state.get("prediction_confidence", 0.0)


    name = st.text_input("Name", key="applicant_name")
    age = st.text_input("Age", key="applicant_age")
    place = st.text_input("Place",key="applicant_place")
    phone_number = st.text_input("Phone Number",key="applicant_phone_number")

    if uploaded_image:
        st.subheader("Uploaded MRI Scan:")
        st.image(uploaded_image, caption="MRI Image", use_container_width=True)

        st.subheader("Diagnosis Result:")
        st.write(f"🟢 **Prediction:** {prediction_label}")
        st.write(f"📊 **Confidence:** {prediction_confidence:.2f}%")

    if st.button("📥 Download Report"):
        if name and age and place and phone_number:
            # Save the uploaded image temporarily
            temp_image_path = "temp_mri_image.jpg"
            uploaded_image.save(temp_image_path)

            # Generate the PDF report
            pdf_path = generate_pdf(name, age, place, phone_number, temp_image_path, prediction_label, prediction_confidence)

            # Provide the PDF as a downloadable file using Streamlit's download_button
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Report",  # Label for the button
                    data=pdf_file,  # File content to be downloaded
                    file_name="Alzheimer_MRI_Report.pdf",  # The name of the file when downloaded
                    mime="application/pdf"  # MIME type for PDF
                )
        else:
            st.warning("⚠ Please fill out all details before downloading.")



    if st.button("🔙 Back"):
        st.session_state["page"] = "scan"
        st.rerun()


def generate_pdf(name, age, place, phone_number, image_path, diagnosis, confidence):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Alzheimer's MRI Scan Report", ln=True, align="C")
    pdf.ln(10)

    # Adding Date and Time
    current_datetime = datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
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
        pdf.image(image_path, x=50, w=100)  
        pdf.ln(10)
        
   
    pdf_filename = "Alzheimer_MRI_Report.pdf"
    pdf.output(pdf_filename)

    return pdf_filename

def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    pages = {"Home": home_page, "Login": login_page, "Signup": signup_page, "guidelines": guidelines_page, "scan": scan_page,   "application_form": application_form_page }
    pages[st.session_state["page"]]()

if __name__ == "__main__":
    main()
