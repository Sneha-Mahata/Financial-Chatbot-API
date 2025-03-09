# app.py - Main FastAPI application
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import os
import json
import dill
import uvicorn
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Import all required LangChain and LangGraph components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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
retriever = None  # Initialize retriever as None

# Define RAG setup function (similar to Kaggle version but adapted for deployment)
def setup_rag_system():
    print("Setting up RAG system with existing knowledge base...")
    
    # Paths adjusted for deployment environment
    embeddings_path = "embeddings"
    chroma_path = "knowledge_base/chromadb"
    
    # Create model instance
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create embedding function
    try:
        embeddings = OpenAIEmbeddings()
        print("Using OpenAI embeddings")
    except Exception as e:
        # Fall back to local embeddings if OpenAI fails
        try:
            print("OpenAI embeddings failed, trying SentenceTransformers")
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e2:
            print(f"Could not initialize embeddings: {str(e2)}")
            return None
    
    # Check if the knowledge base directory exists
    if not os.path.exists(chroma_path):
        print(f"Warning: Knowledge base path {chroma_path} does not exist.")
        return None
    
    # Attempt to load the existing vector store
    try:
        # Specify the embedding function when loading
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        print(f"Successfully loaded Chroma from {chroma_path}")
        
        # Create base retriever with similarity search
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create compressor for more focused retrieval
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compressed retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return retriever
    
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None

# FINANCIAL TOOLS

# Stock Price Tool
def get_stock_price(ticker: str) -> Dict[str, Any]:
    """Get the current price and information for a stock."""
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        
        if data.empty:
            return {
                "status": "error",
                "message": f"Could not find data for ticker symbol {ticker}."
            }
        
        # Get company info
        info = stock.info
        company_name = info.get('shortName', ticker)
        
        # Format the response
        latest_price = data['Close'].iloc[-1]
        open_price = data['Open'].iloc[0]
        
        # Calculate change
        change = latest_price - open_price
        percent_change = (change / open_price) * 100
        
        return {
            "status": "success",
            "data": {
                "company": company_name,
                "ticker": ticker,
                "current_price": latest_price,
                "open_price": open_price,
                "change": change,
                "percent_change": percent_change,
                "date": datetime.now().strftime('%Y-%m-%d %H:%M')
            },
            "message": f"Retrieved stock data for {company_name} ({ticker})"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving stock data for {ticker}: {str(e)}"
        }

# Financial Calculator Tool
def financial_calculator(calculation_type: str, parameters: Dict[str, float]) -> Dict[str, Any]:
    """Perform financial calculations."""
    try:
        if calculation_type.lower() == "compound_interest":
            # Get parameters
            principal = parameters.get("principal")
            rate = parameters.get("rate") / 100  # Convert percentage to decimal
            time = parameters.get("time")
            
            if principal is None or rate is None or time is None:
                return {
                    "status": "error",
                    "message": "Missing required parameters. Need principal, rate, and time."
                }
            
            # Compound interest formula: A = P(1 + r)^t
            amount = principal * (1 + rate) ** time
            interest = amount - principal
            
            return {
                "status": "success",
                "data": {
                    "calculation_type": "compound_interest",
                    "principal": principal,
                    "rate": rate * 100,
                    "time": time,
                    "final_amount": amount,
                    "interest_earned": interest
                },
                "message": "Compound interest calculation completed."
            }
                
        elif calculation_type.lower() == "loan_payment":
            # Get parameters and calculate monthly payment
            principal = parameters.get("principal")
            rate = parameters.get("rate") / 100
            time = parameters.get("time")
            
            if principal is None or rate is None or time is None:
                return {
                    "status": "error",
                    "message": "Missing required parameters. Need principal, rate, and time."
                }
            
            # Monthly rate and number of payments
            monthly_rate = rate / 12
            num_payments = time * 12
            
            # Monthly payment formula: P = (r*PV)/(1-(1+r)^-n)
            if monthly_rate == 0:
                monthly_payment = principal / num_payments
            else:
                monthly_payment = (monthly_rate * principal) / (1 - (1 + monthly_rate) ** -num_payments)
            
            total_paid = monthly_payment * num_payments
            total_interest = total_paid - principal
            
            return {
                "status": "success",
                "data": {
                    "calculation_type": "loan_payment",
                    "loan_amount": principal,
                    "interest_rate": rate * 100,
                    "loan_term_years": time,
                    "monthly_payment": monthly_payment,
                    "total_paid": total_paid,
                    "total_interest": total_interest
                },
                "message": "Loan payment calculation completed."
            }
        else:
            return {
                "status": "error",
                "message": f"Unsupported calculation type: {calculation_type}. Supported types are: compound_interest, loan_payment"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing financial calculation: {str(e)}"
        }

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
    global chatbot, AgentState, functions, retriever
    
    # Set your OpenAI API key from environment variable
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Initialize the retriever if it hasn't been already
    if retriever is None:
        retriever = setup_rag_system()
    
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
    global chatbot, AgentState, retriever
    
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

# Simplified endpoint for basic responses (backup if the main chatbot fails)
@app.post("/simple-chat", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    # Initialize a basic OpenAI chat model
    try:
        # Set your OpenAI API key
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
        # Create a simple OpenAI chat model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        
        # Convert API message format to internal format
        message_history = [] if not request.history else convert_to_internal_messages(request.history)
        
        # Create system message
        system_msg = SystemMessage(content="""You are a financial advisor bot that helps users with financial questions. 
        You can provide information about investments, stocks, financial planning, and calculations. 
        Be professional but conversational in your responses.""")
        
        # Prepare messages for the model
        messages = [system_msg] + message_history + [HumanMessage(content=request.message)]
        
        # Get response
        response = llm.invoke(messages)
        
        # Update message history
        updated_messages = message_history + [
            HumanMessage(content=request.message),
            AIMessage(content=response.content)
        ]
        
        # Convert back to API format
        updated_history = convert_to_api_messages(updated_messages)
        
        return ChatResponse(response=response.content, history=updated_history)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in simple chat: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialize the chatbot on startup
@app.on_event("startup")
async def startup_event():
    global retriever
    # Initialize the retriever
    retriever = setup_rag_system()
    
    # Load the chatbot
    success = load_chatbot()
    if not success:
        print("Warning: Failed to load chatbot on startup. Will attempt to load on first request.")

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)