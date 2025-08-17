# FinBot: AI Financial Analyst 🤖💰

FinBot is an **AI-powered financial assistant** that provides intelligent, context-aware advice through a REST API.  
It uses a **Retrieval-Augmented Generation (RAG)** architecture to deliver accurate and relevant answers.

When a user asks a question, the application first retrieves relevant documents from a **ChromaDB** vector knowledge base.  
This context, combined with the user's query, is then processed by a **multi-agent system** built with **Microsoft AutoGen** and powered by **Google's Gemini LLM**.  

The backend is built with **Python (FastAPI)** and uses **PostgreSQL** for user data management.  

---

## 🚀 Setup & Installation

### Prerequisites
- Python **3.9+**
- A running **PostgreSQL** instance

---

### Instructions

#### 1. Clone & Enter the Repository
```bash
git clone https://github.com/Meekubee/FinBot.git
cd FinBot
```

#### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

#### 4. Configure Environment

**Note**:
Instead of a .env file, settings.py acts as the configuration file for all secret variables.
Make sure to create or edit settings.py with your:

- LLM API Keys

- Endpoint URLs

- PostgreSQL username, password, host (e.g., localhost), and database name

#### 5. Initialize Databases

- Create the required PostgreSQL tables.

- Populate the ChromaDB knowledge base.
