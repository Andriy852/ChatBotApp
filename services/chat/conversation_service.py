from firebase_admin import firestore
from datetime import datetime
import uuid
from typing import Dict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ConversationService:
    """
    Handles storing and retrieving user conversations from Firestore db.
    
    Attributes:
        db: Firestore database instance.
    """
    def __init__(self, db: firestore.Client):
        """
        Initializes the ConversationService with a Firestore database.
        
        Args:
            db: Firestore database client instance.
        """
        self.db = db
    
    def fetch_conversations(self, user_id: str) -> Dict[str, Dict]:
        """
        Fetches all conversations of a given user sorted by the last update time.
        
        Args:
            user_id: The ID of the user whose conversations are being fetched.
        
        Returns:
            A dictionary where keys are conversation IDs and values are conversation data.
        """
        convs = self.db.collection("users").document(user_id) \
                .collection("conversations") \
                .order_by("updated_at", direction=firestore.Query.DESCENDING) \
                .stream()
        return {conv.id: conv.to_dict() for conv in convs}
    
    def save_messages(self, user_id: str, conv_id: str, 
                      messages: List[Union[HumanMessage, AIMessage, SystemMessage]]) -> None:
        """
        Saves messages to a conversation in Firestore.
        
        Args:
            user_id: The ID of the user.
            conv_id: The ID of the conversation.
            messages: A list of messages to save.
        """
        conv_ref = self.db.collection("users").document(user_id) \
                    .collection("conversations").document(conv_id)
        
        serialized = [self._serialize_message(msg) for msg in messages]
        conv_ref.set({
            "messages": serialized,
            "updated_at": datetime.now()
        }, merge=True)
    
    def create_conversation(self, user_id: str, 
                            messages: List[Union[HumanMessage, AIMessage, SystemMessage]], 
                            title: str) -> str:
        """
        Creates a new conversation for a user and stores it in Firestore.
        
        Args:
            user_id: The ID of the user.
            messages: A list of messages to initialize the conversation (can be empty).
            title: The title of the conversation.
        
        Returns:
            The ID of the newly created conversation.
        """
        conv_id = f"{user_id}_{uuid.uuid4()}"
        self.db.collection("users").document(user_id) \
            .collection("conversations").document(conv_id) \
            .set({
                "title": f"{title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "messages": [],
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
        return conv_id
    
    def _serialize_message(self, 
                           message: Union[HumanMessage, AIMessage, SystemMessage]) -> Dict[str, Union[str, Dict]]:
        """
        Serializes a message object into a dictionary format for storage.
        
        Args:
            message: A message instance (HumanMessage, AIMessage, or SystemMessage).
        
        Returns:
            A dictionary representation of the message.
        """
        return {
            "type": type(message).__name__,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs
        }
    
    def _deserialize_message(self, 
                             message_dict: Dict[str, str]) -> Union[HumanMessage, AIMessage, SystemMessage]:
        """
        Deserializes a dictionary representation of a message back into an object.
        
        Args:
            message_dict: A dictionary representing a stored message.
        
        Returns:
            A message instance (HumanMessage, AIMessage, or SystemMessage).
        """
        msg_type = message_dict["type"]
        content = message_dict["content"]
        
        if msg_type == "HumanMessage":
            return HumanMessage(content=content)
        elif msg_type == "AIMessage":
            return AIMessage(content=content)
        elif msg_type == "SystemMessage":
            return SystemMessage(content=content)
        return HumanMessage(content=content)