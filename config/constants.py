import os
MIN_PASSWORD_LENGTH = 8
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
FIREBASE_API_KEY = os.environ["FIREBASE_API_KEY"]
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
PERSIST_DIRECTORY = "./memory"
COLLECTION_NAME = "user_facts"