from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Annotated, List, Optional
import asyncio

import models, database
from database import engine, get_db

from agents import user_proxy, financial_analyst, model_client, get_relevant_financial_info
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat 

from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True



# New Pydantic model for chat requests
class ChatRequest(BaseModel):
    user_id: int # To identify the user interacting
    message: str

# Updated ChatResponse with clean relevant_knowledge
class ChatResponse(BaseModel):
    agent_response: str
    relevant_knowledge: Optional[str] = None  # Only included when there's actual knowledge

# Updated function to filter out non-helpful knowledge
def get_clean_relevant_knowledge(query: str, n_results: int = 3) -> Optional[str]:
    """
    Returns relevant knowledge only if it's actually helpful, otherwise returns None
    """
    raw_knowledge = get_relevant_financial_info(query, n_results)
    
    # Don't return knowledge if it's an error message or "no information found"
    if (raw_knowledge.startswith("Error") or 
        "No relevant" in raw_knowledge or 
        "not properly initialized" in raw_knowledge):
        return None
    
    return raw_knowledge

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Create a new user", tags=["User Management"])
async def create_user(user: UserCreate, db: Annotated[Session, Depends(get_db)]):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    new_user = models.User(username=user.username)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/users/{user_id}", response_model=UserResponse, summary="Get a user by ID", tags=["User Management"])
async def get_user(user_id: int, db: Annotated[Session, Depends(get_db)]):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

@app.get("/users/", response_model=List[UserResponse], summary="Get all users", tags=["User Management"])
async def get_all_users(db: Annotated[Session, Depends(get_db)]):
    users = db.query(models.User).all()
    return users



@app.post("/chat/", response_model=ChatResponse, summary="Send a message to the AI Financial Assistant", tags=["AI Chat"])
async def chat_with_assistant(chat_request: ChatRequest, db: Annotated[Session, Depends(get_db)]):

    user_exists = db.query(models.User).filter(models.User.id == chat_request.user_id).first()
    if not user_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with ID {chat_request.user_id} not found.")

    # Get clean relevant financial information from ChromaDB
    relevant_knowledge = get_clean_relevant_knowledge(chat_request.message)
    
    # Enhanced task with retrieved knowledge
    enhanced_task = f"""
    User Query: {chat_request.message}
    
    Relevant Financial Knowledge:
    {relevant_knowledge if relevant_knowledge else "No specific knowledge base information found for this query."}
    
    Please provide comprehensive financial advice using the above knowledge and your expertise.
    """

    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[user_proxy, financial_analyst],
        termination_condition=termination
    )

    print(f"User {chat_request.user_id} asking: {chat_request.message}")
    try:
        chat_result = await team.run(task=enhanced_task)

        final_agent_response = "No response from agents."
        if chat_result.messages:
            for msg in reversed(chat_result.messages):
                if msg.content and msg.content.strip() and msg.source != "User_Proxy":
                    final_agent_response = msg.content.replace("TERMINATE", "").strip()
                    break
            if not final_agent_response: 
                final_agent_response = chat_result.messages[-1].content.replace("TERMINATE", "").strip()

        return ChatResponse(
            agent_response=final_agent_response,
            relevant_knowledge=relevant_knowledge
        )
    except Exception as e:
        print(f"Error during agent chat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing chat: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if model_client:
        await model_client.close()