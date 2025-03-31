import streamlit as st
from typing import Optional
from .auth_service import AuthService

class AuthUI:
    """
    Handles the authentication user interface using Streamlit.
    
    Attributes:
        service: An instance of AuthService to handle authentication logic.
    """
    def __init__(self, auth_service: AuthService):
        """
        Initializes the AuthUI with an authentication service.
        
        Args:
            auth_service: An instance of AuthService to manage authentication.
        """
        self.service = auth_service
    
    def show_login_form(self) -> Optional[dict]:
        """
        Displays the login form and processes login attempts.
        
        Returns:
            A dictionary containing user information if login is successful, None otherwise.
        """
        with st.form(key='login_form'):
            st.subheader("Login")
            email = st.text_input('Email')
            password = st.text_input('Password', type='password')
            
            if st.form_submit_button("Login"):
                try:
                    return self.service.login_user(email, password)
                except Exception as e:
                    st.error(str(e))
        return None
    
    def show_register_form(self) -> Optional[dict]:
        """
        Displays the registration form and processes user registration.
        
        Returns:
            A dictionary containing user information if registration is successful, None otherwise.
        """
        with st.form(key='register_form'):
            st.subheader("Register")
            email = st.text_input('Email')
            password = st.text_input('Password', type='password')
            confirm = st.text_input('Confirm Password', type='password')
            
            if st.form_submit_button("Register"):
                if password != confirm:
                    st.error("Passwords don't match!")
                else:
                    try:
                        return self.service.register_user(email, password)
                    except Exception as e:
                        st.error(str(e))
        return None
    
    def show_auth_selector(self):
        """
        Displays a radio button selector for choosing between login and registration.
        
        Returns:
            A string indicating the selected authentication action ("Login" or "Register").
        """
        action = st.radio("Select action", ["Login", "Register"], horizontal=True)
        st.divider()
        return action