from autogen_ext.models.openai import OpenAIChatCompletionClient
from settings import GEMINI_MODEL, GEMINI_API_KEY, GEMINI_ENDPOINT
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import UserProxyAgent
from autogen_core.models import ModelInfo
from chroma_util import get_chroma_client, get_or_create_financial_collection, query_collection
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_client = OpenAIChatCompletionClient(
    model=GEMINI_MODEL,
    name="Google",
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_ENDPOINT,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="gemini",
        structured_output=True
    ),
)

# Initialize ChromaDB connection with error handling
try:
    chroma_client = get_chroma_client()
    financial_collection = get_or_create_financial_collection(chroma_client)
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    chroma_client = None
    financial_collection = None

def get_relevant_financial_info(query: str, n_results: int = 3) -> str:
    """
    Retrieves relevant financial information from ChromaDB based on the user query.
    """
    # Initialize the return variable
    relevant_info = "No relevant financial information found in the knowledge base."
    
    try:
        # Check if ChromaDB is properly initialized
        if financial_collection is None:
            return "ChromaDB is not properly initialized. Please check your setup."
        
        # Query the collection
        results = query_collection(financial_collection, [query], n_results=n_results)
        
        # Check if we got any results
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant financial information found in the knowledge base."
        
        # Format the retrieved documents
        relevant_info = "Here's relevant information from the financial knowledge base:\n\n"
        for i, doc in enumerate(results['documents'][0]):
            if doc and doc.strip():  # Make sure the document isn't empty
                relevant_info += f"{i+1}. {doc}\n\n"
        
        # If no valid documents were found after filtering
        if relevant_info == "\n\n":
            relevant_info = "No relevant financial information found in the knowledge base."
        
        logger.info(f"Successfully retrieved {len(results['documents'][0])} relevant documents")
        
    except Exception as e:
        logger.error(f"Error retrieving financial information: {e}")
        relevant_info = f"Error retrieving financial information: {str(e)}"
    
    return relevant_info

user_proxy = UserProxyAgent(
    "User_Proxy",
)

financial_analyst = AssistantAgent(
    name="Financial_Analyst",
    system_message=(
        "You are a skilled financial analyst AI with access to a financial knowledge base. "
        "Your goal is to provide accurate and concise financial insights. "
        "When responding to user queries about financial concepts, investments, or advice: "
        "1. Use the provided relevant information from the knowledge base when available "
        "2. Combine that information with your expertise to provide comprehensive answers "
        "3. For specific data like current stock prices, mention that real-time data tools would be needed "
        "4. Provide practical, actionable advice when appropriate "
        "5. If no relevant knowledge base information is available, rely on your general financial knowledge "
        "Always end your response with 'TERMINATE' when you believe the user's query is resolved."
    ),
    model_client=model_client,
)