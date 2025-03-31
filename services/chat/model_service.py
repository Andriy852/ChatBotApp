from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from config import constants
from typing import List
from langchain_core.prompts import PromptTemplate

class ModelService:
    """
    A service for interacting with a language model, generating responses, and naming conversations.
    """
    def __init__(self):
        """
        Initializes the ModelService with the default language model.
        """
        self.llm = self._init_model()
    
    def _init_model(self, model: str = constants.DEFAULT_MODEL) -> ChatOpenAI:
        """
        Initializes and returns a language model instance.
        
        Args:
            - model: The model name to use.

        Returns: An instance of ChatOpenAI.
        """
        return ChatOpenAI(model=model)
    
    def generate_response(self, messages: List, system_msg: str, **params) -> str:
        """
        Generates a response from the language model based on provided messages and a system message.
        
        Args:
            - messages: A list of user messages.
            - system_msg: A system message to set context.
            - params: Additional parameters for model invocation.
        Return: The generated response as a string.
        """
        system_msg = SystemMessage(content=system_msg)
        messages = [system_msg] + messages
        return self.llm.invoke(messages, **params)
    
    def name_conversation(self, messages: List) -> str:
        """
        Generates a short 3-4 word title summarizing the conversation.
        
        Args:
            - messages: A list of messages from the conversation.
        Return: A generated conversation title.
        """
        prompt_temp = PromptTemplate(input_variables=["message"],
                                 template="Your task it to give a title to the messages below."
                                            "Make the title 3-4 words long. Messages: {messages}")
        chain = prompt_temp | self.llm
        return chain.invoke({"messages": messages}).content