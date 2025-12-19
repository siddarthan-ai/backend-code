import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# --- 1. SETUP & INITIALIZATION ---

# Your New API Key integrated directly for CMD execution
GEMINI_API_KEY = "AIzaSyA4vDHqJqC1lV9h_28LEawM7gS7naCM--4"

# Initialize the Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Global Memory Storage
SESSION_HISTORY: Dict[str, List[types.Content]] = {}

# Define the Tool Declaration for Google Search
google_search_tool_declaration = types.Tool(
    google_search=types.GoogleSearch()
)

# -------------------------------------------------------------------------
# LILY'S FULL SYLLABUS KNOWLEDGE BASE (University of Madras)
# -------------------------------------------------------------------------
LILY_SYSTEM_INSTRUCTION = """
You are LILY, a helpful academic AI tutor for a student at the University of Madras.
The student is in the B.Sc. Computer Science with AI (Semester II) program.

Your core knowledge includes:
1. JAVA: JVM architecture, OOP, Threads, and Swing GUI.
2. MATH: Fourier Series, Laplace Transforms, and Vector Differentiation.
3. APTITUDE: HCF/LCM, Profit/Loss, and Probability.
4. ENGLISH: 'Don't Quit', 'Wings of Fire', and workplace E-mails.
5. TAMIL: Sitrilakkiyams and modern literature.

Always provide step-by-step logic for academic problems.
"""

# --- 2. FASTAPI SERVER SETUP ---

app = FastAPI(title="LILY AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default_user"

@app.post("/chat")
async def chat_with_lily(request: ChatRequest):
    session_id = request.session_id
    user_input = request.user_input
    
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []
    
    history = SESSION_HISTORY[session_id]
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(
                tools=[google_search_tool_declaration],
                system_instruction=LILY_SYSTEM_INSTRUCTION 
            )
        )
        
        ai_response_text = response.text
        history.append(response.candidates[0].content)
        return {"response": ai_response_text}

    except Exception as e:
        print(f"Error: {e}")
        return {"response": "Lily encountered a technical issue. Check your CMD console."}
