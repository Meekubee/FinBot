import chromadb
from chromadb.utils import embedding_functions
import logging

CHROMA_DB_PATH = "./chroma_db_data"
COLLECTION_NAME = "financial_advice"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_chroma_client():
    """Returns a persistent ChromaDB client."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info(f"ChromaDB client initialized at {CHROMA_DB_PATH}")
        return client
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {e}")
        raise

def get_or_create_financial_collection(client: chromadb.PersistentClient):
    """
    Gets an existing collection or creates a new one for financial advice.
    Using get_or_create_collection is safer than create_collection directly
    to prevent errors if the script is run multiple times.
    """
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        logger.info(f"Collection '{COLLECTION_NAME}' ready with {collection.count()} documents")
        return collection
    except Exception as e:
        logger.error(f"Error creating/accessing collection: {e}")
        raise

def add_documents_to_collection(collection: chromadb.Collection, documents: list[str], metadatas: list[dict], ids: list[str]):
    """Adds documents to the specified ChromaDB collection."""
    try:
        if collection.count() == 0:
            logger.info(f"Adding {len(documents)} documents to ChromaDB collection '{collection.name}'...")
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Documents added. Total items in collection: {collection.count()}")
        else:
            logger.info(f"Collection '{collection.name}' already contains {collection.count()} items. Skipping addition.")
    except Exception as e:
        logger.error(f"Error adding documents to collection: {e}")
        raise

def query_collection(collection: chromadb.Collection, query_texts: list[str], n_results: int = 3):
    """Queries the collection for relevant documents."""
    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=min(n_results, collection.count()) 
        )
        logger.info(f"Query executed successfully, returned {len(results['documents'][0])} results")
        return results
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        raise

def add_new_financial_document(collection: chromadb.Collection, document: str, metadata: dict, doc_id: str):
    """
    Adds a single new document to the collection.
    Useful for dynamically adding new financial knowledge.
    """
    try:
        # Check if document with this ID already exists
        existing = collection.get(ids=[doc_id])
        if existing['ids']:
            logger.warning(f"Document with ID '{doc_id}' already exists. Skipping.")
            return False
        
        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[doc_id]
        )
        logger.info(f"Added new document with ID: {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding new document: {e}")
        raise

def update_financial_document(collection: chromadb.Collection, doc_id: str, new_document: str, new_metadata: dict):
    """
    Updates an existing document in the collection.
    """
    try:
        collection.update(
            ids=[doc_id],
            documents=[new_document],
            metadatas=[new_metadata]
        )
        logger.info(f"Updated document with ID: {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise

def delete_financial_document(collection: chromadb.Collection, doc_id: str):
    """
    Deletes a document from the collection.
    """
    try:
        collection.delete(ids=[doc_id])
        logger.info(f"Deleted document with ID: {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise

def initialize_knowledge_base():
    """
    Initializes the knowledge base with sample financial data.
    Call this function once to set up your ChromaDB with initial data.
    """
    client = get_chroma_client()
    collection = get_or_create_financial_collection(client)

    sample_knowledge = [
        "Diversification is a strategy employed to minimize risk by investing in a variety of assets. It aims to reduce the impact of any single asset's poor performance on the overall portfolio.",
        "Compounding is the process where the earnings from an investment are reinvested to generate additional earnings over time. It's often referred to as 'interest on interest' and is a powerful concept for long-term wealth growth.",
        "An emergency fund is a stash of money set aside to cover unexpected expenses, such as job loss, medical emergencies, or major car repairs. Financial experts often recommend having 3-6 months' worth of living expenses saved.",
        "Inflation is the rate at which the general level of prices for goods and services is rising, and subsequently, purchasing power is falling. Central banks aim to keep inflation stable to maintain economic health.",
        "A budget is a financial plan that helps you track your income and expenses, allowing you to see where your money is going and make informed decisions about spending and saving.",
        "ETFs (Exchange Traded Funds) are a type of investment fund that holds multiple underlying assets and trades on stock exchanges like individual stocks. They offer diversification and liquidity.",
        "Bonds are debt instruments issued by governments or corporations to raise capital. When you buy a bond, you're lending money in exchange for periodic interest payments and the return of your principal at maturity.",
        "Retirement planning involves setting financial goals for your post-working life and developing strategies to achieve them, often through savings accounts like 401(k)s or IRAs.",
        "Cryptocurrency is a digital or virtual currency that is secured by cryptography, making it nearly impossible to counterfeit or double-spend. Many cryptocurrencies are decentralized networks based on blockchain technology.",
        "Risk tolerance is the degree of variability in investment returns that an investor is willing to withstand. It's a crucial factor in determining an appropriate asset allocation for a portfolio.",
        "Dollar-cost averaging is an investment strategy where you invest a fixed amount of money at regular intervals, regardless of market conditions. This helps reduce the impact of market volatility.",
        "A Roth IRA is a retirement account where contributions are made with after-tax dollars, but qualified withdrawals in retirement are tax-free. This is particularly beneficial for younger investors.",
        "Credit score is a numerical representation of your creditworthiness, typically ranging from 300 to 850. It affects your ability to get loans and the interest rates you'll pay.",
        "Asset allocation is the strategy of dividing your investment portfolio among different asset categories, such as stocks, bonds, and cash, based on your goals, risk tolerance, and time horizon."
    ]

    sample_metadatas = [
        {"source": "financial_glossary", "category": "risk_management"},
        {"source": "financial_glossary", "category": "investing"},
        {"source": "financial_glossary", "category": "personal_finance"},
        {"source": "financial_glossary", "category": "economics"},
        {"source": "financial_glossary", "category": "personal_finance"},
        {"source": "financial_glossary", "category": "investing"},
        {"source": "financial_glossary", "category": "investing"},
        {"source": "financial_glossary", "category": "retirement"},
        {"source": "financial_glossary", "category": "cryptocurrency"},
        {"source": "financial_glossary", "category": "risk_management"},
        {"source": "financial_glossary", "category": "investing"},
        {"source": "financial_glossary", "category": "retirement"},
        {"source": "financial_glossary", "category": "credit"},
        {"source": "financial_glossary", "category": "investing"}
    ]
    
    sample_ids = [f"doc{i+1}" for i in range(len(sample_knowledge))]

    add_documents_to_collection(collection, sample_knowledge, sample_metadatas, sample_ids)
    return collection

if __name__ == "__main__":
    collection = initialize_knowledge_base()
    