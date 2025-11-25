"""Settings and configuration for the application."""
import os
from dotenv import load_dotenv

load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://cpuhivcyhvqayzgdvdaw.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNwdWhpdmN5aHZxYXl6Z2R2ZGF3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMzNDc4NDgsImV4cCI6MjA2ODkyMzg0OH0.dO22JLQjE7UeQHvQn6mojILNuWi_02MiZ9quz5v8pNk")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Application Settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Chatbot Configuration
CHATBOT_PROMPT_REQUIREMENTS = """
Requirements:
- Reply using small, mobile-friendly messages.
- Start with a warm, gentle greeting directly to the user (no names).
- Maintain a warm, friendly, supportive tone.
- Avoid medical or technical jargon.
- Give simple, practical, everyday suggestions.
- Keep responses short and concise for mobile screens.
- Maximum length per reply: 45–60 words.
- End every message with a hopeful, varied closing line.
- If providing helplines, include only Sri Lankan helplines.
Note: Do not mention being an AI or use any AI-related terms.
"""
