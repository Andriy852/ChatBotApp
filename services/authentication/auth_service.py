from firebase_admin import auth, exceptions
from datetime import datetime
from typing import Optional, Tuple, Dict
import re
from config import constants
import requests
import json

class AuthError(Exception):
    """Custom exception for authentication errors."""
    pass

class AuthService:
    """
    Service for handling user authentication, including registration and login.
    
    Attributes:
        db: The database instance used for storing and retrieving user data.
    """
    def __init__(self, db):
        """
        Initializes the AuthService with a database instance.
        
        Args:
            db: The database client used to store user information.
        """
        self.db = db
    
    def validate_email(self, email: str) -> bool:
        """
        Validates an email address using a regular expression.
        
        Args:
            email: The email address to validate.
        
        Returns:
            True if the email is valid, False otherwise.
        """
        return re.match(constants.EMAIL_REGEX, email) is not None
    
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """
        Validates a password based on predefined security rules.
        
        Args:
            password: The password to validate.
        
        Returns:
            A tuple where the first element is a boolean indicating validity,
            and the second element is an error message if invalid.
        """
        if len(password) < constants.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {constants.MIN_PASSWORD_LENGTH} characters"
        return True, ""
    
    def register_user(self, email: str, password: str) -> Dict[str, str]:
        """
        Registers a new user with an email and password.
        
        Args:
            email: The user's email address.
            password: The user's password.
        
        Returns:
            A dictionary containing the user details.
        
        Raises:
            AuthError: If email format is invalid, password is too short, or registration fails.
        """
        if not self.validate_email(email):
            raise AuthError("Invalid email format")
            
        valid, msg = self.validate_password(password)
        if not valid:
            raise AuthError(msg)
            
        try:
            user = auth.create_user(email=email, password=password, email_verified=False)
            self._save_user_data(user.uid, email)
            return user
        except Exception as EMAIL_EXISTS:
            raise AuthError("Email already exists")
        except Exception as e:
            raise AuthError(f"Registration failed: {str(e)}")
    
    def login_user(self, email: str, password: str) -> Dict[str, str]:
        """
        Logs in a user using email and password.
        
        Args:
            email: The user's email address.
            password: The user's password.
        
        Returns:
            A dictionary containing the user details.
        
        Raises:
            AuthError: If login fails due to invalid credentials or server issues.
        """
        try:
            auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={constants.FIREBASE_API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            response = requests.post(auth_url, data=json.dumps(payload))
            data = response.json()
            if data.get("error"):
                error_message = data['error']['message']
                if error_message == 'INVALID_EMAIL':
                    raise AuthError(f"Login failed: Invalid Email")
                else:
                    raise AuthError(f"Login failed: Wrong Password")
            
            user_id = data['localId']
            user = auth.get_user(user_id)

            self.db.collection('users').document(user.uid).update({
                'last_login': datetime.now()
            })
            return user
        except exceptions.FirebaseError as e:
            raise AuthError(f"Login failed: {str(e)}")
    
    def _save_user_data(self, uid: str, email: str) -> None:
        """
        Saves user data to the database.
        
        Args:
            uid: The unique user ID.
            email: The user's email address.
        """
        user_data = {
            'email': email,
            'created_at': datetime.now(),
            'last_login': datetime.now()
        }
        self.db.collection('users').document(uid).set(user_data)