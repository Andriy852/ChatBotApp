from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from typing import List, Union
import streamlit as st

class MemoryService:
    """
    A service for extracting, storing, and retrieving user-specific factual information from conversations.
    """
    def __init__(self, vector_store):
        """
        Initializes the MemoryService with a given vector store.
        
        Attributes:
            - vector_store: The vector database used to store and retrieve facts.
        """
        self.vector_store = vector_store
    
    def extract_facts(self, conversation: List[str]) -> Union[List[str], str]:
        """
        Extracts factual information from a given conversation using an AI model.
        
        Attributes:
            - conversation: A list of conversation messages.
        
        Return: A list of extracted facts or a string indicating no facts were found.
        """
        fact_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        **Role**: You are an information extraction specialist. Analyze the conversation and identify ONLY permanent, verifiable facts about the user's:
         
         **Core Categories**:
            1. Personal Attributes:
            - Name, age, birth date
            - Location 

        2. Preferences & Lifestyle:
            - Hobbies
            - Likes/dislikes (food, activities, brands)
            - Daily habits and routines
            - Hobbies and leisure activities
            - Entertainment preferences (books, movies, music)
            - Shopping preferences

        3. Professional Life:
            - Current job title and employer
            - Industry and specialization
            - Work history and education
            - Skills and certifications
            - Career goals and aspirations

        4. Health & Wellness:
            - Allergies and dietary restrictions
            - Medical conditions and medications
            - Exercise and fitness routines
            - Sleep patterns
            - Health goals

        5. Relationships & Social:
            - Family members (spouse, children, parents)
            - Relationship status
            - Close friends and colleagues
            - Pets and their details
            - Important dates (anniversaries, birthdays)

        6. Technology & Digital:
            - Preferred devices and platforms
            - Frequently used apps/software
            - Tech skill level
            - Privacy preferences
            - Social media usage

        7. Financial Context:
            - Budgeting habits
            - Financial goals
            - Investment interests
            - Spending patterns

        8. Travel & Geography:
            - Frequent destinations
            - Travel preferences (hotels, airlines)
            - Languages spoken
            - Cultural interests
            - Future travel plans
        
        **Rules**:
        1. Extract ONLY direct statements about the user ("I prefer X") not general knowledge
        2. Do not extract facts about others ("My friend likes X") or general observations
        3. Ignore vague or ambiguous statements ("I like it") unless specific
        4. Ignore non-factual statements ("I feel happy") unless they indicate a fact
        5. DON'T extract facts that the user has not explicitly stated. Don't infer them or assume. 
        6. Extract only facts that the user specifically mentioned. Don't infer or assume anything.
        7. Ignore temporary states ("I'm tired today") unless health-related
        8. Convert implied facts to explicit form:
        - "I always order oat milk" → "Preference: oat milk in coffee"
        9. Use this exact format for each fact:
        ```
        - [Category]: [Fact detail] (Confidence: High/Medium/Low)
        ```
        
        **Examples**:
        Good:
        ```
        - Name: Alice (Confidence: High)
        - Dietary preference: Vegetarian (Confidence: Medium)
        - Occupation: Software engineer (Confidence: High)
        ```
        
        Bad (rejected):
        ```
        - The weather is nice today (Not about user)
        - They talked about machine learning (Not a personal fact)
        - User seems tired (Temporary state)
        ```
        
        **Output**:
        Return only extracted facts in the specified format, one per line.
        If you haven't found any facts, return exactly: `No facts to extract.`
        Don't include any other text or explanations. Don't use bullet points or lists.
        Don't use quotation marks.
        """),
        ("human", "Conversation:\n{conversation}")
        ])
        chain = fact_extraction_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        extracted = chain.invoke({"conversation": conversation}).content
        return extracted.split("\n") if extracted != "No facts to extract." else extracted
    
    def store_facts(self, user_id: str, facts: List[str]):
        """
        Stores extracted facts in the vector store for a given user.
        
        Attributes
            - user_id: The unique identifier of the user.
            - facts: A list of factual statements about the user.
        """
        docs = [Document(
            page_content=fact,
            metadata={"user_id": user_id}
        ) for fact in facts]
        self.vector_store.add_documents(docs)
    
    def is_new_info(self, user_id: str, info: str) -> bool:
        """
        Checks if a given fact is new for the user by comparing it with stored information.
        
        Attributes
            - user_id: The unique identifier of the user.
            - info: The factual statement to check.
        Return:
            - True if the fact is new, False otherwise.
        """
        retrieved = self.vector_store.similarity_search_with_score(
            query=info,
            k=1,
            filter={"user_id": user_id}
        )
        if not retrieved:
            return True
        _, score = retrieved[0]
        return score > 0.1
    
    def get_facts(self, user_id: str) -> List[str]:
        """
        Retrieves stored factual information for a given user.
        
        Attritubes
            - user_id: The unique identifier of the user.
        Return: 
            - A list of stored facts about the user.
        """
        retrieved = self.vector_store.similarity_search(
            query="",
            k=100,
            filter={"user_id": user_id}
        )
        return [doc.page_content for doc in retrieved]