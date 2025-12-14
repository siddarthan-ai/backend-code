# main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

# --- New Import for CORS ---
from fastapi.middleware.cors import CORSMiddleware
# ---------------------------

# --- Google AI SDK & UTILITIES ---
from google import genai
from google.genai import types
# Removed: import requests (no longer needed)

# --- 1. SETUP & INITIALIZATION ---

# >>>>> CRITICAL TEMPORARY FIX: HARDCODE YOUR NEW, FRESH KEY HERE <<<<<
import os
# The variable name is 'GEMINI_API_KEY'
GEMINI_API_KEY = os.getenv("AIzaSyDPYHKdTZzp8C5OhzJ-iOSmtlRzAjyL1D0")

# Initialize the Gemini Client using the hardcoded key
client = genai.Client(api_key=GEMINI_API_KEY)

# Global Memory Storage: Using a simple dictionary for zero-cost memory
SESSION_HISTORY: Dict[str, List[types.Content]] = {}

# --- TOOL DEFINITIONS ---

# 🛑 NO duckduckgo_search function needed! We use the managed Google Search Tool.

# Define the Tool Declaration for Google Search (built-in)
# This uses the special GoogleSearch tool type.
google_search_tool_declaration = types.Tool(
    google_search=types.GoogleSearch()
)

# --- Define Lily's Personality Separately ---
LILY_SYSTEM_INSTRUCTION = (
    "You are Lily, a helpful, chat-only AI assistant. Your tone is warm, professional, and witty. "
    "You now have access to the powerful Google Search tool for all real-time information. "
    "Use Google Search when you need current facts, news, or weather."
)
# ---------------------------------------------


# --- 2. FASTAPI SERVER SETUP AND CORS CONFIGURATION ---

app = FastAPI(title="Project Lily Backend Core (Google Search Live)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,      
    allow_methods=["*"],         
    allow_headers=["*"],         
)
# ----------------------------------------------------

# --- 3. API LOGIC ---

def get_session_history(session_id: str) -> List[types.Content]:
    """Retrieves or initializes session history."""
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []
    return SESSION_HISTORY[session_id]


class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default_user" 

@app.post("/chat")
async def chat_with_lily(request: ChatRequest):
    session_id = request.session_id
    user_input = request.user_input
    
    history = get_session_history(session_id)
    user_content = types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    history.append(user_content)

    try:
        # Call the Gemini API with history and the Google Search Tool
        # The entire search process is managed by Google's servers in ONE call.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                tools=[google_search_tool_declaration],
                system_instruction=LILY_SYSTEM_INSTRUCTION 
            )
        )
        
        # NOTE: Since Google Search is built-in, there is no need to check
        # for function_calls or make a second API call—Gemini handles it internally.
        
        # Save the direct and final response
        ai_response_text = response.text
        history.append(response.candidates[0].content)
        return {"response": ai_response_text}

    except Exception as e:
        if history and history[-1].role == "user":
            history.pop()
        print(f"An error occurred: {e}")
        # Check for specific API error messages
        if "API Key not found" in str(e):
             return {"response": "Critical Error: The API Key is invalid. Please generate a new key and update main.py."}
        if "RESOURCE_EXHAUSTED" in str(e):
             return {"response": "Error: You have exceeded the free rate limit (too many requests per minute). Please wait 60 seconds."}
        

        return {"response": "I'm sorry, Lily encountered a major technical issue. Please check the server console."}


