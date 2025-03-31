import firebase_admin
from firebase_admin import credentials
import streamlit as st
import os

def initialize_firebase():
    """
    Initializes the Firebase Admin SDK with service account credentials.
    
    This function:
    - Checks if Firebase is already initialized to prevent duplicate initialization
    - Attempts to initialize Firebase with credentials from a service account JSON file
    - Handles and displays any initialization errors
    - Returns initialization status
    
    Returns:
        bool: True if Firebase was successfully initialized or already initialized,
              False if initialization failed
    """
    if not firebase_admin._apps:
        try:
            firebase_config = {k: str(v) for k,v in st.secrets["FIREBASE_CREDENTIALS"].items()}
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {e}")
            return False
    return True