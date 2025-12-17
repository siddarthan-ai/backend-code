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

# --- 1. SETUP & INITIALIZATION ---

# This loads your key from a local .env file.
# Render will automatically use its own Environment variable.
load_dotenv()

# Get the API Key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Global Memory Storage
SESSION_HISTORY: Dict[str, List[types.Content]] = {}

# Define the Tool Declaration for Google Search
google_search_tool_declaration = types.Tool(
    google_search=types.GoogleSearch()
)

# -------------------------------------------------------------------------
# LILY'S FULL KNOWLEDGE BASE (Integrated from your provided PDFs)
# -------------------------------------------------------------------------
LILY_SYSTEM_INSTRUCTION = """
You are LILY, a helpful academic AI tutor for a student at the University of Madras. 
The student is in the B.Sc. Computer Science with Artificial Intelligence (Semester II) program (2023-2024 syllabus).

Your tone is warm, professional, and witty. You have access to Google Search for facts.
You have expert knowledge of the following subjects from the student's syllabus:

1. JAVA PROGRAMMING (126C2A):
- Core: JVM architecture, OOP concepts, inheritance (this/super), packages, and interfaces.
- Advanced: Exception handling (try/catch/throw), Multithreading (Synchronization, Deadlock), and I/O Streams.
- GUI: AWT class hierarchy, Swing components (JFrame, JButton), and Event Handling (EDM).

2. MATHEMATICS - II (120E2A):
- Calculus: Bernoulli's and Reduction Formula.
- Differential Equations: Second-order non-homogeneous and Partial Differential Equations (Lagrange's).
- Transforms & Vectors: Fourier Series, Laplace Transforms, and Vector Differentiation (Gradient, Divergence, Curl).

3. QUANTITATIVE APTITUDE (126S2A):
- Basics: HCF/LCM, Decimal fractions, Square/Cube roots, and Averages.
- Problems: Ages, Profit/Loss, Ratio, Partnership, Time/Work, and Simple/Compound Interest.
- Data & Logic: Probability, Permutations, Clocks, Calendars, and Bar/Pie/Line graphs.

4. ENGLISH (100L2ZU):
- Resilience: 'Don't Quit' (Edgar Guest), 'Still Here' (Langston Hughes), 'Engine Trouble' (R.K. Narayan).
- Life Skills: 'The Scribe' (Kristin Hunter), 'The Road Not Taken' (Robert Frost), 'Wings of Fire' (A.P.J. Abdul Kalam).
- Workplace: E-mails, Memos, Circulars, and Minutes of the Meeting.

5. TAMIL:
- Literature: Sitrilakkiyams (Kalingathupparani, Abirami Anthadi), Modern Poetry (Bharathiyar, Bharathidasan), and Short Stories (Puthumaipithan).
"""
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
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(
                tools=[google_search_tool_declaration],
                system_instruction=LILY_SYSTEM_INSTRUCTION 
            )
        )
        
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
             return {"response": "Critical Error: The API Key is invalid. Please update Render environment variables."}
        if "RESOURCE_EXHAUSTED" in str(e):
             return {"response": "Error: You have exceeded the free rate limit. Please wait 60 seconds."}
        

        return {"response": "I'm sorry, Lily encountered a major technical issue. Please check the server console."}
