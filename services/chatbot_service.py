# app/chat_api.py
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import json
import random
import re
import logging
from fastapi.middleware.cors import CORSMiddleware

# Try import Groq; if unavailable, we'll gracefully fallback
try:
    from groq import Groq
except Exception:
    Groq = None  # type: ignore

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat_api")

# --- Environment & config ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODELS = {
    "default": "llama-3.1-8b-instant",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "llama3.1-70b": "llama3.1-70b-versatile",
    "llama2-70b-4096": "groq/compound"
}

# --- System prompt (strong, concise, mobile-focused) ---
SYSTEM_PROMPT = """
You are a compassionate, mobile-friendly mental health support companion.
Rules:
- Produce short, supportive, human-like replies suitable for 1-3 mobile lines.
- Start with a warm acknowledgement (e.g., "I hear you.", "Thanks for sharing.").
- Use plain language. Avoid medical jargon, self-labeling, or offering diagnoses.
- Give practical, everyday suggestions (breathing, grounding, short tasks).
- Never say you are an AI or mention system/dev notes.
- If the user expresses suicidal intent or self-harm, switch to crisis mode:
  - Be calm, non-judgmental, encourage contacting local help, include crisis resources.
- Do not output hotline numbers unless in crisis mode.
- Always end with a brief, varied hopeful note (e.g., "You’re not alone.", "Small steps help.").
- Keep replies concise and mobile-friendly.
"""

# --- Crisis detection ---
CRISIS_KEYWORDS = [
    r"\bsuicide\b", r"\bkill myself\b", r"\bend my life\b", r"\bwant to die\b",
    r"\bcan't go on\b", r"\bcan\'t go on\b", r"\bcut myself\b", r"\bself[- ]harm\b",
    r"\bhurt myself\b", r"\bi want to die\b", r"\bno reason to live\b",
    r"\blife is pointless\b", r"\bi want everything to end\b"
]

# Crisis response (numbers only shown when crisis detected)
CRISIS_RESPONSE_TEMPLATE = (
    "I'm really concerned about you and I'm glad you reached out. "
    "If you're thinking about suicide or hurting yourself, please contact local support right now.\n\n"
    "Sumithrayo Hotline (Sri Lanka): 011 2 682 682\n"
    "Sri Lanka College of Psychiatrists Helpline: 071 722 5222\n\n"
    "You don't have to face this alone. Please consider reaching out to someone you trust or a helpline."
)

# --- Fallback & dataset load ---
DATASET = {}
try:
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "models", "chatbot_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        # map tag -> first response (safe short snippet)
        DATASET = {
            intent["tag"].lower(): intent["responses"][0]
            for intent in raw.get("intents", [])
            if "tag" in intent and "responses" in intent and intent["responses"]
        }
    logger.info("Loaded local dataset for contextual hints")
except Exception as e:
    logger.warning(f"Could not load dataset: {e}")
    DATASET = {}

# Simple canned responses (retain your previous style)
SIMPLE_RESPONSES = {
    "hi": ["**HELLO!** How can I support you today?", "**HI THERE!** Hope your day is going okay."],
    "hello": ["**HI THERE!** I'm here to listen.", "**HELLO!** Glad you reached out today."],
    "hey": ["**HEY!** How are you feeling?", "**HI!** I'm here for you."],
    "bye": ["**TAKE CARE!** Remember you're not alone.", "**GOODBYE!** Wishing you peace and comfort."],
    "thanks": ["**YOU'RE WELCOME!** I'm here if you need more support.", "**ANYTIME!** I'm glad to be here for you."]
}

# --- Helper functions ---
def contains_crisis(message: str) -> bool:
    text = (message or "").lower()
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, text):
            return True
    return False

def sanitize_ai_text(text: str) -> str:
    """Strip markdown, odd characters, and collapse whitespace."""
    if not text:
        return ""
    # Remove common markdown bold/italic/headers/quotes
    text = re.sub(r"(\*\*|__|\*|_|`|#+|>\s?)", "", text)
    # Replace multiple whitespace/newlines with single newline where appropriate
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # Collapse long whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text

def enforce_mobile_format(text: str, max_lines: int = 3, max_words: int = 70) -> str:
    """
    Ensure response is mobile friendly:
    - Limit number of newline-separated lines.
    - Limit total words roughly.
    - Add a short hopeful ending if missing.
    """
    text = sanitize_ai_text(text)
    # Split into sentences and build up to max_lines
    sentences = re.split(r'(?<=[.!?])\s+', text)
    lines = []
    for s in sentences:
        if len(lines) >= max_lines:
            break
        s = s.strip()
        if s:
            lines.append(s)
    if not lines:
        lines = [text.strip()]

    # Ensure total word limit
    all_words = " ".join(lines).split()
    if len(all_words) > max_words:
        # truncate words and add ellipsis
        all_words = all_words[:max_words]
        final = " ".join(all_words).rstrip(" ,.;:!") + "..."
    else:
        final = " ".join(lines)

    # hopeful endings pool & ensure one is present
    hopeful_phrases = [
        "You're not alone.",
        "Small steps can help.",
        "You deserve support.",
        "One small step at a time.",
        "It can get a bit lighter."
    ]
    if not any(p.lower() in final.lower() for p in hopeful_phrases):
        final = final + " " + random.choice(hopeful_phrases)

    # Break into short lines for mobile readability (approx 2 sentences per line)
    # Keep at most max_lines lines
    final_sentences = re.split(r'(?<=[.!?])\s+', final)
    out_lines = []
    cur = ""
    for sent in final_sentences:
        if not cur:
            cur = sent
        elif len((cur + " " + sent).split()) <= 25 and len(out_lines) < max_lines:
            cur = cur + " " + sent
        else:
            out_lines.append(cur.strip())
            cur = sent
        if len(out_lines) >= max_lines:
            break
    if cur and len(out_lines) < max_lines:
        out_lines.append(cur.strip())

    # Trim to required lines
    out_lines = out_lines[:max_lines]
    return "\n".join(out_lines).strip()

def get_fallback_response(user_message: str) -> str:
    um = (user_message or "").lower()
    for keyword, responses in {
        k: v for k, v in {
            "sad": [
                "It's okay to feel sad sometimes. A short walk or a small comforting activity might help.",
                "I hear your sadness. Consider reaching out to someone you trust."
            ],
            "anxious": [
                "When anxiety hits, try slow breathing — 4 seconds in, 4 seconds out. Small steps help.",
                "Anxiety is heavy. Can you name 3 things you can see right now? Grounding can help."
            ],
            "stressed": [
                "Stress can feel overwhelming. Try one tiny break — a glass of water or a stretch.",
                "You're carrying a lot. Could one small task be moved or shared?"
            ]
        }.items()
    }.items():
        if keyword in um:
            return random.choice(responses)
    # Try dataset hints
    for tag, advice in DATASET.items():
        if tag in um:
            return advice
    generic = [
        "Thanks for sharing. I'm here to listen — can you tell me a bit more about how you're feeling?",
        "I appreciate you opening up. What's feeling most heavy right now?"
    ]
    return random.choice(generic)

# --- Initialize Groq client gracefully ---
groq_client = None
if Groq is not None and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        # optional quick smoke test (kept minimal)
        try:
            _ = groq_client.chat.completions.create(
                model=MODELS["default"],
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
                max_tokens=10,
                temperature=0.4
            )
            logger.info("Connected to Groq API")
        except Exception as e:
            logger.warning(f"Groq client initialized but smoke test failed: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None
else:
    logger.info("Groq client not configured or import failed; falling back to local responses")

# --- FastAPI setup ---
app = FastAPI(title="SafeSpace Chat API")
router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response models ---
class ChatHistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: str = "default"
    user_id: Optional[str] = None
    history: Optional[List[ChatHistoryItem]] = None
    max_words: Optional[int] = Field(70, description="Approximate max words to keep reply mobile-friendly")

class ChatResponse(BaseModel):
    response: str

# --- Core function to query Groq API ---
def query_groq(model: str, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.4) -> str:
    if groq_client is None:
        raise RuntimeError("Groq client not available")
    # ensure system prompt at top
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Groq responses may vary in property names; adapt defensively
        content = ""
        try:
            content = resp.choices[0].message.content
        except Exception:
            # fallback naming
            content = getattr(resp.choices[0], "text", "") or str(resp)
        return content or ""
    except Exception as e:
        logger.error(f"Error from Groq API: {e}")
        raise

# --- Chat endpoint ---
@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    try:
        model_key = (request.model or "default").lower()
        if model_key not in (k.lower() for k in MODELS.keys()):
            logger.warning(f"Unknown model requested: {request.model}; falling back to default")
            model_key = "default"
        # map to canonical model name
        selected_model = MODELS.get(model_key, MODELS["default"])
        logger.info(f"Selected model: {selected_model}")

        user_message = (request.message or "").strip()
        user_message_lower = user_message.lower()

        # Direct simple responses (exact match on short greetings)
        if user_message_lower in SIMPLE_RESPONSES:
            reply = random.choice(SIMPLE_RESPONSES[user_message_lower])
            reply = enforce_mobile_format(reply, max_lines=3, max_words=request.max_words or 70)
            return ChatResponse(response=reply)

        # Crisis detection
        crisis_mode = contains_crisis(user_message)
        if crisis_mode:
            # Immediately return crisis response with helplines
            reply = enforce_mobile_format(CRISIS_RESPONSE_TEMPLATE, max_lines=4, max_words=request.max_words or 70)
            return ChatResponse(response=reply)

        # Build messages for the model if available
        messages: List[Dict[str, str]] = []
        if request.history:
            for h in request.history:
                messages.append({"role": h.role, "content": h.content})

        # Add current user message
        # Provide contextual hint from local dataset if relevant
        context_hint = ""
        for keyword, advice in DATASET.items():
            if keyword in user_message_lower:
                context_hint = f"\nHelpful background: {advice}"
                break

        user_content = user_message + (context_hint or "")
        # Append a short instruction (system prompt already has the main rules)
        user_content += "\nRespond supportively, briefly, and with empathy."

        messages.append({"role": "user", "content": user_content})

        # Try using Groq first
        if groq_client is not None:
            try:
                ai_raw = query_groq(selected_model, messages, max_tokens=200, temperature=0.4)
                cleaned = sanitize_ai_text(ai_raw)
                formatted = enforce_mobile_format(cleaned, max_lines=3, max_words=request.max_words or 70)
                return ChatResponse(response=formatted)
            except Exception as e:
                logger.warning(f"Groq failed: {e}. Falling back to local responses.")

        # Fallback to local dataset or heuristic responses
        fallback = get_fallback_response(user_message)
        formatted = enforce_mobile_format(fallback, max_lines=3, max_words=request.max_words or 70)
        return ChatResponse(response=formatted)

    except Exception as exc:
        logger.exception(f"Unhandled error in chat endpoint: {exc}")
        # Return a compassionate generic reply
        generic = "I care about what you're sharing. Could you tell me a bit more?"
        return ChatResponse(response=enforce_mobile_format(generic, max_lines=2, max_words=40))

# Mount router
app.include_router(router)

# Simple root for health check
@app.get("/")
def root():
    return {"service": "SafeSpace Chat API", "status": "ok"}
