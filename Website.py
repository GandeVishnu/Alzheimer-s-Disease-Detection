import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="Login / Sign Up App",
    page_icon="🔐",
    layout="centered"
)

# Initialize session state if not already
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Function to switch pages
def switch_page(page_name):
    st.session_state.page = page_name

# Home Page
def home_page():
    st.title("👋 Welcome to the Application")

    st.write("Please **Login** or **Sign Up** to continue.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):
            switch_page("login")

    with col2:
        if st.button("Sign Up", use_container_width=True):
            switch_page("signup")

# Login Page
def login_page():
    st.title("🔑 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        # Dummy authentication
        if username and password:
            st.success(f"Logged in as {username}")
        else:
            st.error("Please enter Username and Password.")

    if st.button("⬅️ Back to Home", use_container_width=True):
        switch_page("home")

# Signup Page
def signup_page():
    st.title("📝 Sign Up")

    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up", use_container_width=True):
        if new_username and new_password and confirm_password:
            if new_password == confirm_password:
                st.success(f"Account created for {new_username}!")
            else:
                st.error("Passwords do not match.")
        else:
            st.error("Please fill out all fields.")

    if st.button("⬅️ Back to Home", use_container_width=True):
        switch_page("home")

# Main app controller
def main():
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
    else:
        st.error("Unknown page!")

# Run the app
if __name__ == "__main__":
    main()
