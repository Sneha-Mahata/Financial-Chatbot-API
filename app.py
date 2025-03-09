# app.py - Main FastAPI application
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import os
import json
import dill
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langgraph.graph import END, StateGraph

# Initialize FastAPI app
app = FastAPI(title="Financial Advisor Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your Next.js app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]

# Global variables
chatbot = None
AgentState = None
functions = None

# Helper functions to convert between API message format and internal format
def convert_to_internal_messages(api_messages):
    internal_messages = []
    for msg in api_messages:
        if msg["role"] == "user":
            internal_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            internal_messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            internal_messages.append(SystemMessage(content=msg["content"]))
    return internal_messages

def convert_to_api_messages(internal_messages):
    api_messages = []
    for msg in internal_messages:
        if isinstance(msg, HumanMessage):
            api_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            api_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            api_messages.append({"role": "system", "content": msg.content})
    return api_messages

# Load the chatbot components
def load_chatbot():
    global chatbot, AgentState, functions
    
    # Set your OpenAI API key from environment variable
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    try:
        # Load the configuration
        with open("financial_chatbot_config.json", 'r') as f:
            config = json.load(f)
        
        # Load the functions
        with open("financial_chatbot_functions.dill", 'rb') as f:
            functions = dill.load(f)
        
        # Load the AgentState class
        with open("agent_state_class.dill", 'rb') as f:
            AgentState = dill.load(f)
        
        # Recreate the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes from the configuration
        for node_name, func_name in config["nodes"].items():
            workflow.add_node(node_name, functions[func_name])
        
        # Add conditional edges
        workflow.add_conditional_edges(
            config["entry_point"],
            functions["router"]
        )
        
        # Add other edges
        workflow.add_edge("retrieve_knowledge", "generate_response")
        workflow.add_edge("use_stock_price_tool", "generate_response")
        workflow.add_edge("use_calculator_tool", "generate_response")
        workflow.add_edge("use_data_analysis_tool", "generate_response")
        
        # Final edge
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point(config["entry_point"])
        
        # Compile the graph
        chatbot = workflow.compile()
        
        print("Financial chatbot loaded successfully!")
        return True
    
    except Exception as e:
        print(f"Error loading financial chatbot: {str(e)}")
        return False

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global chatbot, AgentState
    
    if chatbot is None:
        success = load_chatbot()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load chatbot")
    
    # Convert API message format to internal format
    message_history = [] if not request.history else convert_to_internal_messages(request.history)
    
    # Add user message to history
    message_history.append(HumanMessage(content=request.message))
    
    # Initialize state
    state = {
        "messages": message_history,
        "query": request.message,
        "retrieval_context": None,
        "tools_output": None,
        "step_count": 0,
        "active_tool": None,
        "final_response": None
    }
    
    try:
        # Run the workflow
        result = chatbot.invoke(state)
        response = result["final_response"]
        updated_history = convert_to_api_messages(result["messages"])
        
        return ChatResponse(response=response, history=updated_history)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat processing: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialize the chatbot on startup
@app.on_event("startup")
async def startup_event():
    success = load_chatbot()
    if not success:
        print("Warning: Failed to load chatbot on startup. Will attempt to load on first request.")

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)