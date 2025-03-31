from typing_extensions import Annotated, Literal
from typing_extensions import List, TypedDict, Any, Dict
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_pinecone import PineconeVectorStore

class RetrievalState(TypedDict):
    """
    A typed dictionary representing the state of the retrieval process.
    
    Attributes:
        messages: List of conversation messages (HumanMessage, AIMessage, etc.)
        query: The latest user query that needs processing
        needs_retrieval: Boolean flag indicating whether retrieval is needed
        context: List of retrieved documents (facts) if any were fetched
        llm: Instance of the language model to be used for processing
        retriever: Instance of the retriever to be used for document fetching
    """
    messages: List[Annotated[str, "Conversation history (HumanMessage, AIMessage, etc.)"]]
    query: Annotated[str, "Latest user query"]
    needs_retrieval: Annotated[bool, "Whether retrieval is needed"]
    context: Annotated[List[Document], "Retrieved facts (if any)"]
    llm: Annotated[Any, "LLM instance"]
    retriever: Annotated[Any, "Retriever instance"]

class RetrievalService:
    """
    A service that handles document retrieval based on conversation context.
    
    This service uses a LangGraph workflow to determine when retrieval is needed
    and executes the retrieval process when necessary.
    
    Args:
        model: The ChatOpenAI instance to use for decision making
        vector_store: The vector store to use for document retrieval
        user_id: The ID of the user whose documents should be retrieved
    """
    def __init__(self, model: ChatOpenAI, vector_store: PineconeVectorStore, user_id: str) -> None:
        """
        Initializes the RetrievalService with the given components.
        
        Args:
            model: The ChatOpenAI instance to use for decision making
            vector_store: The vector store to use for document retrieval
            user_id: The ID of the user whose documents should be retrieved
        """
        self.retriever = vector_store.as_retriever(
                    search_kwargs={"k": 1000, "filter": {"user_id": user_id}})
        self.llm = model
        self._initialize_workflow()

    def _initialize_workflow(self) -> None:
        """
        Sets up the LangGraph workflow with nodes and edges.
        
            The workflow consists of:
            - A decision node to determine if retrieval is needed
            - A retrieval node that executes if needed
        """
        self.workflow = StateGraph(RetrievalState)
        
        self.workflow.add_node("should_retrieve", self._should_retrieve)
        self.workflow.add_node("retrieve_if_needed", self._retrieve_if_needed)
        
        self.workflow.add_edge("should_retrieve", "retrieve_if_needed")
        self.workflow.add_edge("retrieve_if_needed", END)
        
        self.workflow.set_entry_point("should_retrieve")
        self.app = self.workflow.compile()

    def execute(self, state: RetrievalState) -> RetrievalState:
        """
        Executes the retrieval workflow with the given state.
        
        Args:
            state: The current retrieval state containing messages, query, etc.
            
        Returns:
            The updated retrieval state after processing
        """
        return self.app.invoke(state)

    def _should_retrieve(self, state: RetrievalState) -> Dict[str, bool]:
        """
        Determines if retrieval is needed based on conversation history and query.
        
        Uses an LLM to make the decision following specific guidelines about
        when retrieval should be performed.
        
        Args:
            state: The current retrieval state
            
        Returns:
            Dictionary with 'needs_retrieval' boolean indicating if retrieval is needed
        """
        retrieval_decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            **Task**: Analyze the conversation and determine if retrieval of user-specific facts is needed to answer the query.

            **Decision Guidelines**:

            1. **RETRIEVE** when:
                   **Answering the query requires knowing addional information about the user**:
                    - Demographics: Name, age, location, birthday
                    - Preferences: Likes/dislikes, habits, routines
                    - Professional: Job, education, skills, career goals
                    - Health: Allergies, conditions, medications, fitness
                    - Relationships: Family, friends, pets, status
                    - Tech: Devices, apps, privacy preferences
                    - Financial: Budgets, goals, spending habits
                    - Travel: Frequent destinations, preferences
                    
                    **Verification Needed**:
                    - References to past user statements ("As I mentioned before...")
                    - Requests involving personal history

                    **User explicitly asks about their own information**
                
            2. **SKIP** when:
                - Query can be answered with general knowledge
                - Answer is present in the conversation
                - Follow-up to your previous response
                - Contains all needed information in the message
                - Is a clarification or rephrasing

            **Special Cases**:
                - If uncertain, default to SKIP
                - Never retrieve for sensitive information requests

            **Output Format**:
            Respond with EXACTLY ONE of these:
                - "RETRIEVE"
                - "SKIP"
             """),
             ("human", "Current conversation: {messages}\n\nQuery to analyze: {query}")])
        
        chain = retrieval_decision_prompt | state["llm"]
        decision = chain.invoke({"messages": state["messages"], "query": state["query"]}).content.strip()
        
        return {"needs_retrieval": decision == "RETRIEVE"}

    def _retrieve_if_needed(self, state: RetrievalState) -> RetrievalState:
        """
        Executes document retrieval if needed based on the current state.
        
        Args:
            state: The current retrieval state
            
        Returns:
            Dictionary with 'context' containing retrieved documents (empty if no retrieval)
        """
        if not state["needs_retrieval"]:
            return {"context": []} 
        
        docs = state["retriever"].get_relevant_documents(query="")
        return {"context": docs}
    