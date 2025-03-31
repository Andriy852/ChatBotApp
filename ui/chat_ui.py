import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Optional, Dict
from services.chat.conversation_service import ConversationService
from services.chat.model_service import ModelService
from services.chat.memory_service import MemoryService
from services.chat.retrieval_service import RetrievalService

class ChatUI:
    """
    Streamlit-based chat user interface that handles conversation management,
    message display, and user interactions.
    
    Attributes:
        conv_service: Service for conversation persistence and management
        model_service: Service for model interactions and response generation
        memory_service: Service for long-term memory operations
        retrieval_service: Service for document retrieval when needed
    """
    def __init__(
        self, 
        conv_service: ConversationService,
        model_service: ModelService,
        memory_service: MemoryService,
        retrieval_service: RetrievalService
    ):
        """
        Initializes the chat UI with required services.
        
        Args:
            conv_service: Conversation management service
            model_service: Model interaction service
            memory_service: Long-term memory service
            retrieval_service: Document retrieval service
        """
        self.conv_service = conv_service
        self.model_service = model_service
        self.memory_service = memory_service
        self.retrieval_service = retrieval_service
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = None
        if 'model_params' not in st.session_state:
            self._init_model_params()

    def _init_model_params(self):
        """Initializes default model parameters in session state."""
        st.session_state.model_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 512,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }

    def render(self):
        """
        Main rendering method that orchestrates the entire UI.
        
        This method calls all component renderers in sequence:
            1. Model parameter controls
            2. Sidebar components
            3. Chat message history
            4. User input handling
            5. User facts management
        """
        self._render_model_controls()
        self._render_sidebar()
        self._render_chat_messages()
        self._handle_user_input()
        self._render_user_facts()

    def _render_user_facts(self):
        """
        Renders the button and display for user facts.
        
        When clicked, shows all stored facts about the user from memory.
        """
        if st.button("🔍 Show My Facts", help="Display stored information about you"):
            all_docs = self.memory_service.get_facts(st.session_state.user.uid)
            if not all_docs:
                st.toast("No facts found", icon="ℹ️")
            else:
                with st.expander("My Facts", expanded=True):
                    for doc in all_docs:
                        st.write(doc)

    def _render_model_controls(self):
        """
        Renders the model parameter controls in the main interface.
        
            Includes sliders for:
            - Temperature
            - Top-p sampling
            - Max tokens
            - Frequency penalty
            - Presence penalty
        """
        with st.container():
            cols = st.columns(5)
            with cols[0]:
                st.session_state.model_params['temperature'] = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=0.7, 
                    step=0.1,
                    help="Creativity (0=factual, 2=creative)"
                )
            with cols[1]:
                st.session_state.model_params['top_p'] = st.slider(
                    "Top-p", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.9, 
                    step=0.05,
                    help="Response diversity"
                )
            with cols[2]:
                st.session_state.model_params['max_tokens'] = st.slider(
                    "Max Tokens", 
                    min_value=0, 
                    max_value=16384, 
                    value=2028, 
                    step=64,
                    help="Response length limit"
                )
            with cols[3]:
                st.session_state.model_params['frequency_penalty'] = st.slider(
                    "Freq Penalty", 
                    min_value=-2.0, 
                    max_value=2.0, 
                    value=0.0, 
                    step=0.1,
                    help="Reduce repetition (+)"
                )
            with cols[4]:
                st.session_state.model_params['presence_penalty'] = st.slider(
                    "Presence Penalty", 
                    min_value=-2.0, 
                    max_value=2.0, 
                    value=0.0, 
                    step=0.1,
                    help="Encourage new topics (+)"
                )

    def _render_sidebar(self):
        """
        Renders all sidebar components including:
            - Logout button
            - Memory actions
            - Conversation history
        """
        with st.sidebar:
            self._render_logout_button()
            self._render_memory_actions()
            self._render_conversation_history()

    def _render_logout_button(self):
        """Renders the logout button and user email display."""
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{st.session_state.user.email}**")
        with col2:
            if st.button("Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    def _render_memory_actions(self):
        """Renders the fact extraction button in the sidebar."""
        if st.button("💾 Extract Facts", 
                   use_container_width=True,
                   help="Save key information to long-term memory"):
            self._handle_fact_extraction()

    def _handle_fact_extraction(self):
        """
        Handles the fact extraction process from current conversation.
        
        Extracts facts from user messages and stores new information to memory.
        """
        if not st.session_state.messages:
            st.toast("No messages to extract", icon="ℹ️")
            return
            
        with st.spinner("Identifying important facts..."):
            user_messages = [
                msg.content for msg in st.session_state.messages 
                if isinstance(msg, HumanMessage)
            ]
            facts = self.memory_service.extract_facts(user_messages)
            if facts == "No facts to extract.":
                st.toast("No facts found", icon="ℹ️")
            else:
                facts_to_add = [
                    fact for fact in facts 
                    if self.memory_service.is_new_info(
                        st.session_state.user.uid, 
                        fact
                    )
                ]
                if facts_to_add:
                    self.memory_service.store_facts(
                        st.session_state.user.uid, 
                        facts_to_add
                    )
                    st.toast("Saved new information!", icon="✅")
                else:
                    st.toast("Information already saved", icon="ℹ️")

    def _render_conversation_history(self):
        """
        Renders the conversation history panel in the sidebar.
        
        Shows list of past conversations and handles conversation switching.
        """
        if st.button("➕ New Chat", key="new_chat", use_container_width=True):
            self._reset_conversation()
        st.subheader("History")
            
        conversations = self.conv_service.fetch_conversations(
            st.session_state.user.uid
        )
        
        for conv_id, conv_data in conversations.items():
            title = conv_data["title"].split("-")[0].strip()
            if st.button(
                title,
                key=conv_id,
                use_container_width=True,
                type="primary" if st.session_state.conversation_id == conv_id else "secondary"
            ):
                self._load_conversation(conv_id, conv_data)

    def _reset_conversation(self):
        """Resets the current conversation state."""
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

    def _load_conversation(self, conv_id: str, conv_data: Dict) -> None:
        """
        Loads a conversation from history into the current session.
        
        Args:
            conv_id: The conversation ID to load
            conv_data: Dictionary containing conversation data
        """
        st.session_state.conversation_id = conv_id
        st.session_state.messages = [
            self.conv_service._deserialize_message(msg) 
            for msg in conv_data.get("messages", [])
        ]
        st.rerun()

    def _render_chat_messages(self):
        """Renders the chat message history in the main panel."""
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

    def _handle_user_input(self):
        """
        Handles user input from the chat interface.
        
        Processes the message, generates a response, and updates the conversation.
        """
        user_input = st.chat_input("Type your message...")
        if user_input:
            system_msg = self._process_user_message(user_input)
            self._generate_and_display_response(system_msg)
            self._save_conversation()
            st.rerun()

    def _process_user_message(self, user_input: str):
        """
        Processes a user message and determines if retrieval is needed.
        
        Args:
            user_input: The message text from the user
            
        Returns:
            The system message to use for response generation
        """
        retrieval_check = self.retrieval_service.execute({
            "messages": st.session_state.messages,
            "query": user_input,
            "llm": self.model_service.llm,
            "retriever": self.retrieval_service.retriever})
        if retrieval_check["needs_retrieval"]:
            system_msg = f"""You are a helpful assistant.
                            Please answer the question based on the context provided.
                            If no context is available, answer based on your knowledge.
                            If the context is not relevant, ignore it.
                            Context: {retrieval_check["context"]}"""
        else:
            system_msg = "You are a helpful assistant."
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.write(user_input)
        return system_msg

    def _generate_and_display_response(self, system_msg):
        """
        Generates an AI response and displays it in the chat.
        
        Args:
            system_msg: The system message to use for context
        """
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = self.model_service.generate_response(
                    st.session_state.messages,
                    system_msg,
                    **st.session_state.model_params
                )
                st.session_state.messages.append(response)
                st.write(response.content)

    def _save_conversation(self):
        """Save conversation to database"""
        if not st.session_state.conversation_id:
            title = self.model_service.name_conversation(
                st.session_state.messages
            )
            st.session_state.conversation_id = (
                self.conv_service.create_conversation(
                    st.session_state.user.uid,
                    st.session_state.messages,
                    title
                )
            )
            
        self.conv_service.save_messages(
            st.session_state.user.uid,
            st.session_state.conversation_id,
            st.session_state.messages
        )