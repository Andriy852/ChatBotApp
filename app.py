import streamlit as st
from firebase_admin import firestore
from config.firebase_config import initialize_firebase
from services.authentication.auth_ui import AuthUI
from services.authentication.auth_service import AuthService
from services.chat.conversation_service import ConversationService
from services.chat.model_service import ModelService
from services.chat.memory_service import MemoryService
from services.chat.retrieval_service import RetrievalService
from ui.chat_ui import ChatUI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import constants

# Initialize services
if not initialize_firebase():
    st.stop()

db = firestore.client()
auth_service = AuthService(db)
auth_ui = AuthUI(auth_service)

# Initialize vector store
embeddings = OpenAIEmbeddings(model=constants.EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=constants.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=constants.PERSIST_DIRECTORY
)

if not st.session_state.get('logged_in'):
    action = auth_ui.show_auth_selector()
    if action == "Login":
        user = auth_ui.show_login_form()
    else:
        user = auth_ui.show_register_form()
    
    if user:
        st.session_state.logged_in = True
        st.session_state.user = user
        st.rerun()
else:
    conv_service = ConversationService(db)
    model_service = ModelService()
    memory_service = MemoryService(vector_store)
    retrieval_service = RetrievalService(model_service.llm, vector_store, st.session_state.user.uid)
    chat_ui = ChatUI(conv_service, model_service, memory_service, retrieval_service)
    # Start chat interface
    # vector_store.delete(where={"user_id": "BduNQR9VTrZTfusKURQXcEeO0Bx1"})
    chat_ui.render()