import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from fpdf import FPDF
import base64
import time
from datetime import datetime
import pytz
from pymongo import MongoClient
from io import BytesIO
import re
import os
# -------------------- MongoDB Setup --------------------
MONGO_URL = st.secrets.get("MONGO_URL")  
if not MONGO_URL:
    st.error("MongoDB URL not found. Please set MONGO_URL in .env or Streamlit secrets.")
    st.stop()
client = MongoClient(MONGO_URL)
db = client["AlzheimersDiseaseDetection"]    
users_collection = db["users"]   
applications_collection = db["applications"]  
#----
page_title="Alzheimers Disease Detection"
page_icon="🧠"
st.set_page_config(page_title=page_title,page_icon=page_icon)
MODEL_PATH = "20_04_2025_ADNI_best_model.keras"
IMG_SIZE = (224, 224)
class_labels = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)

model = load_prediction_model()

# -------------------- Image Processing --------------------
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

def get_previous_applications(email):
    applications = applications_collection.find({"user_email": email}).sort("submitted_at", -1)
    return list(applications)

# -------------------- Styling --------------------
def add_responsive_styles():
    try:
        with open("styles.css", "r") as css_file:
            css = css_file.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: styles.css not found. Please ensure it is in the same directory as the Python script.")

# -------------------- Pages --------------------
def home_page():
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown('<div class="title-text">🧠 Alzheimer\'s Disease Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Analyze brain MRI scans to predict Alzheimer\'s disease stages using advanced deep learning models.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                st.session_state["page"] = "Login"
                st.toast("✅ Moving to Login Page", icon="✅")
                time.sleep(0.5)
                st.rerun()
        with col2:
            if st.button("Signup"):
                st.session_state["page"] = "Signup"
                st.toast("✅ Moving to Signup Page", icon="✅")
                time.sleep(0.5)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)

def login_page():
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.subheader("🔐 Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password")
        users = load_users()
        if st.button("Login"):
            if email in users and users[email]["password"] == password:
                st.toast("✅ Login Successful! Redirecting..", icon="✅")
                time.sleep(0.5)
                st.session_state["Name"] = users[email]["name"]
                st.session_state["Email"] = email
                st.session_state["page"] = "guidelines"
                st.rerun()
            else:
                st.error("Invalid email or password.")
        if st.button("Back to Home"):
            st.session_state["page"] = "Home"
            st.toast("✅ Back to Home Page", icon="✅")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)

def signup_page():
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.subheader("📝 Signup")
        name = st.text_input("Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Re-enter Password", type="password", key="signup_confirm_password")
        users = load_users()
        if st.button("Signup"):
            if not name or not name.strip():
                st.error("Name cannot be empty.")
            elif not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+\-=\[\]{};:'\",.<>?]).+$", password):
                st.error("Password must contain at least one uppercase letter, one lowercase letter, and one special character.")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            elif not email:
                st.error("Email is required.")
            elif email in users:
                st.error("User already exists!")
            else:
                save_user(email, name, password)
                st.toast("✅ Signup Successful! Redirecting to Home...", icon="✅")
                time.sleep(0.5)
                st.session_state["page"] = "Home"
                st.rerun()
        if st.button("Back to Home"):
            st.session_state["page"] = "Home"
            st.toast("✅ Back to Home Page", icon="✅")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)

def guidelines_page():
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: DodgerBlue; padding: 10px; font-size:35px'>Welcome, {st.session_state.get('Name', 'User')}!</h1>", unsafe_allow_html=True)
        st.markdown("""
            <h2 class="alzheimers-title">📋 What is Alzheimer's Disease?</h2>
            <p class="alzheimers-description">Alzheimer's disease is a progressive brain disorder causing memory loss and cognitive decline.</p>
            <ul class="alzheimers-list">
                <li><span class="label">Final CN JPEG:</span> <span class="description">Cognitively Normal – No cognitive impairment.</span></li>
                <li><span class="label">Final EMCI JPEG:</span> <span class="description">Early Mild Cognitive Impairment – Very mild symptoms, subtle memory lapses.</span></li>
                <li><span class="label">Final MCI JPEG:</span> <span class="description">Mild Cognitive Impairment – General MCI, includes both early and late stages.</span></li>
                <li><span class="label">Final LMCI JPEG:</span> <span class="description">Late Mild Cognitive Impairment – More severe than EMCI, close to AD onset.</span></li>
                <li><span class="label">Final AD JPEG:</span> <span class="description">Alzheimer’s Disease – Advanced cognitive decline, significant memory and behavioral changes.</span></li>
            </ul>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Proceed to Scan"):
                st.session_state["page"] = "scan"
                st.toast("✅ Redirecting to Scan Page...", icon="✅")
                time.sleep(0.5)
                st.rerun()
        with col2:
            if st.button("Previous Scan"):
                st.session_state["page"] = "previous_scan"
                st.toast("✅ Redirecting to Previous Scan Page...", icon="✅")
                time.sleep(0.5)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)

def scan_page():
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.title(f"📊 Alzheimer’s MRI Scan")
        uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width =True)
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
                st.toast("✅ Back to Guidelines Page...", icon="✅")
                time.sleep(0.5)
                st.rerun()
        with col2:
            if st.button("📄 View Application Form"):
                st.session_state["page"] = "application_form"
                st.toast("✅ Redirecting to Application Page...", icon="✅")
                time.sleep(0.5)
                st.rerun()
        with col3:
            if st.button("🚪 Sign Out"):
                st.session_state["page"] = "Home"
                st.toast("✅ Back to Home Page...", icon="✅")
                time.sleep(0.5)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)

def previous_scan_page():
    st.title("📜 Previous Scan Details")
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        email = st.session_state.get("Email", "")
        if not email:
            st.error("Please log in to view previous scans.")
            if st.button("Back to Guidelines"):
                st.session_state["page"] = "guidelines"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)
            return
        applications = get_previous_applications(email)
        if applications:
            for idx, application in enumerate(applications, 1):
                submitted_at = application.get("submitted_at")
                if submitted_at:
                    try:
                        user_timezone = pytz.timezone("Asia/Kolkata")
                        submitted_at = datetime.strptime(submitted_at, "%d-%m-%Y %H:%M:%S")
                        submitted_at = user_timezone.localize(submitted_at)
                        submitted_str = submitted_at.strftime("%d-%m-%Y %I:%M %p")
                    except Exception as e:
                        st.error(f"Error parsing date for scan {idx}: {str(e)}")
                        submitted_str = str(submitted_at)
                else:
                    submitted_str = "N/A"
                st.markdown(f"### Scan {idx} - Submitted: {submitted_str}")
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
                        st.image(image, caption=f"MRI Image - Scan {idx}", use_container_width =True)
                    except Exception as e:
                        st.error(f"Error displaying image for scan {idx}: {str(e)}")
                else:
                    st.info(f"No MRI image available for scan {idx}.")
                st.markdown("---")
        else:
            st.info("No previous scans found.")
        if st.button("Back to Guidelines"):
            st.session_state["page"] = "guidelines"
            st.toast("✅ Back to Guidelines Page...", icon="✅")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)

def application_form_page():
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.title("📝 Application Form")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, step=1)
        place = st.text_input("Place")
        phone_number = st.text_input("Phone Number")
        uploaded_image = st.session_state.get("uploaded_image", None)
        prediction_label = st.session_state.get("prediction_label", "N/A")
        prediction_confidence = st.session_state.get("prediction_confidence", 0.0)
        if uploaded_image:
            st.subheader("Uploaded MRI Scan:")
            st.image(uploaded_image, caption="MRI Image", use_container_width=True)
            st.subheader("Diagnosis Result:")
            st.write(f"🟢 **Prediction:** {prediction_label}")
            st.write(f"📊 **Confidence:** {prediction_confidence:.2f}%")
        if st.button("📥 Generate Report"):
            # Validate fields
            if not name or not name.strip():
                st.error("Name cannot be empty.")
            elif age < 0:
                st.error("Age must be a non-negative integer.")
            elif not place or not place.strip():
                st.error("Place cannot be empty.")
            elif not phone_number or not re.match(r"^\d{10}$", phone_number):
                st.error("Phone number must be exactly 10 digits.")
            elif not uploaded_image:
                st.error("Please upload an MRI image.")
            else:
                with st.spinner("Generating report..."):
                    india_timezone = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.now(india_timezone)
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    formatted_datetime = current_time.strftime("%d-%m-%Y %H:%M:%S")

                    # Sanitize patient name for filename
                    sanitized_name = re.sub(r'[^a-zA-Z0-9]', '_', name.strip())
                    pdf_filename = f"Alzheimer_MRI_Report_{sanitized_name}_{timestamp}.pdf"
                    temp_image_path = f"temp_mri_image_{sanitized_name}_{timestamp}.jpg"

                    # Save form data
                    form_data = {
                        "user_email": st.session_state.get("Email", ""),
                        "name": name,
                        "age": int(age),
                        "place": place,
                        "phone_number": int(phone_number),
                        "prediction": prediction_label,
                        "confidence": float(prediction_confidence),
                        "image_base64": encode_image(uploaded_image) if uploaded_image else None,
                        "submitted_at": formatted_datetime
                    }
                    save_application_form(form_data)

                    # Save temporary image
                    uploaded_image.save(temp_image_path)

                    # Generate PDF
                    pdf_path = generate_pdf(name, age, place, phone_number, temp_image_path, prediction_label, prediction_confidence, pdf_filename, formatted_datetime)

                    # Store PDF path in session state
                    st.session_state["pdf_path"] = pdf_path
                    st.success("Report generated successfully!")

                    # Clean up temporary image
                    try:
                        os.remove(temp_image_path)
                    except OSError as e:
                        st.warning(f"Warning: Could not delete temporary image: {e}")

        # Show download button if PDF is available
        if "pdf_path" in st.session_state:
            with open(st.session_state["pdf_path"], "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Report",
                    data=pdf_file,
                    file_name=os.path.basename(st.session_state["pdf_path"]),
                    mime="application/pdf"
                )
            # Clean up PDF file after download
            try:
                os.remove(st.session_state["pdf_path"])
                del st.session_state["pdf_path"]
            except OSError as e:
                st.warning(f"Warning: Could not delete PDF file: {e}")

        if st.button("🔁 Guidelines Page"):
            st.session_state["page"] = "guidelines"
            st.toast("✅ Back to Guidelines Page...", icon="✅")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 alzheimers-disease-detection</div>', unsafe_allow_html=True)


def generate_pdf(name, age, place, phone_number, image_path, diagnosis, confidence, pdf_filename, formatted_datetime):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Alzheimer's MRI Scan Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Report Generated: {formatted_datetime}", ln=True)
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
    pdf.output(pdf_filename)
    return pdf_filename

def main():
    add_responsive_styles()
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
