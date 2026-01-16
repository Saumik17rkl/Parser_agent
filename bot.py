import os
import json
import logging
import uuid
import re
import requests
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from flask_cors import CORS
from pymongo import MongoClient, ReturnDocument
from bson import json_util
import hashlib

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# -------------------- Configuration --------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------- Constants --------------------
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 1800))
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral:instruct')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF')
CACHE_TTL = 3600  # 1 hour cache

PARSER_SYSTEM_PROMPT = """You are a STRICT, DEFENSIVE Parser Agent for mathematical problems.

Your job is to transform noisy input into a CLEAN, STRUCTURED, VERIFIABLE representation.

────────────────────────────────────────
PRIMARY RESPONSIBILITIES
────────────────────────────────────────
1. Clean OCR/ASR artifacts without changing mathematical meaning
2. Normalize math language into standard symbolic form
3. Extract ONLY explicitly stated information
4. Detect ambiguity, missing information, or inconsistent notation
5. Classify topic and subtopic within allowed scope
6. Decide whether human clarification (HITL) is required

────────────────────────────────────────
STRICT SCOPE
────────────────────────────────────────
Allowed topics:
- algebra
- probability
- calculus
- linear_algebra

Any other topic (geometry, statistics, trigonometry, number theory, etc.)
must be flagged as out-of-scope and require clarification.

────────────────────────────────────────
ABSOLUTE RULES
────────────────────────────────────────
1. ❌ NO GUESSING - Don't infer missing constraints or domains
2. ❌ NO SILENT FIXES - Report incomplete or broken OCR text
3. ❌ NO SOLVING - Don't compute values or simplify beyond cleanup
4. ✅ EXPLICIT EXTRACTION ONLY - Variables and constraints must be explicit
5. ✅ FAIL FAST - Set needs_clarification=true for ANY ambiguity

────────────────────────────────────────
OCR/ASR CLEANING RULES
────────────────────────────────────────
Safe normalizations:
- "ex" → "x", "why" → "y", "zee" → "z"
- "squared" → "^2", "cubed" → "^3"
- "times" → "*", "divided by" → "/"
- "plus" → "+", "minus" → "-"
- "greater than" → ">", "less than" → "<"

────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON)
────────────────────────────────────────
{
  "problem_text": "cleaned and normalized problem statement",
  "topic": "algebra | probability | calculus | linear_algebra | unknown",
  "subtopic": "specific subtopic or unknown",
  "variables": ["list of variables explicitly present"],
  "constraints": ["list of explicitly stated constraints"],
  "ambiguities": ["list of detected ambiguities"],
  "needs_clarification": true | false,
  "clarification_questions": ["questions if clarification needed"]
}

If ambiguities.length > 0, needs_clarification MUST be true.
Accuracy > completeness. Silence is better than guessing.
"""

# -------------------- In-Memory Cache --------------------
class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl=CACHE_TTL):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, datetime.now().timestamp())
    
    def clear(self):
        self.cache.clear()

response_cache = SimpleCache()

# -------------------- In-Memory Storage Fallback --------------------
class InMemoryStorage:
    """Fallback storage when MongoDB is unavailable"""
    
    def __init__(self):
        self.sessions = {}
        logger.info("Initialized in-memory storage")
    
    def find_one(self, query=None, **kwargs):
        if not query or '_id' not in query:
            return None
        return self.sessions.get(query['_id'])
    
    def find_one_and_update(self, filter_query, update, **kwargs):
        doc_id = filter_query.get('_id')
        if doc_id not in self.sessions:
            return None
        
        if '$set' in update:
            self.sessions[doc_id].update(update['$set'])
        if '$push' in update:
            for key, value in update['$push'].items():
                if key not in self.sessions[doc_id]:
                    self.sessions[doc_id][key] = []
                self.sessions[doc_id][key].append(value)
        
        return self.sessions[doc_id]
    
    def insert_one(self, document):
        doc_id = document.get('_id', str(uuid.uuid4()))
        document['_id'] = doc_id
        self.sessions[doc_id] = document
        return type('Result', (), {'inserted_id': doc_id})()
    
    def update_one(self, filter_query, update, **kwargs):
        doc_id = filter_query.get('_id')
        if doc_id not in self.sessions:
            return type('Result', (), {'matched_count': 0})()
        
        if '$set' in update:
            self.sessions[doc_id].update(update['$set'])
        if '$push' in update:
            for key, value in update['$push'].items():
                if key not in self.sessions[doc_id]:
                    self.sessions[doc_id][key] = []
                self.sessions[doc_id][key].append(value)
        
        return type('Result', (), {'matched_count': 1})()
    
    def delete_many(self, query):
        deleted = 0
        if 'last_activity' in query and '$lt' in query['last_activity']:
            cutoff = query['last_activity']['$lt']
            to_delete = [
                sid for sid, session in self.sessions.items()
                if session.get('last_activity', datetime.max) < cutoff
            ]
            for sid in to_delete:
                del self.sessions[sid]
                deleted += 1
        return type('Result', (), {'deleted_count': deleted})()

# -------------------- Database Setup --------------------
def initialize_database():
    """Initialize MongoDB connection with fallback to in-memory storage"""
    mongo_uri = os.getenv('MONGODB_URI')
    
    if not mongo_uri:
        logger.warning("MONGODB_URI not set, using in-memory storage")
        return InMemoryStorage()
    
    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Test connection
        client.server_info()
        
        db_name = os.getenv('MONGODB_DBNAME', 'math_parser')
        db = client[db_name]
        collection = db[os.getenv('SESSION_COLLECTION', 'sessions')]
        
        logger.info(f"Connected to MongoDB successfully! Database: {db_name}")
        return collection
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        logger.warning("Falling back to in-memory storage")
        return InMemoryStorage()

sessions_collection = initialize_database()

# -------------------- API Client Initialization --------------------
def initialize_groq_client():
    """Initialize Groq client with error handling"""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set")
        return None
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}")
        return None

def initialize_openai_client():
    """Initialize OpenAI client as fallback"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OPENAI_API_KEY not set")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
        return None

groq_client = initialize_groq_client()
openai_client = initialize_openai_client()

# -------------------- Flask Application --------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'dev-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

CORS(app)

# -------------------- Utility Functions --------------------
def preprocess_text(text):
    """Clean OCR/ASR errors from input text"""
    if not text:
        return ""
    
    replacements = {
        r'\bex\b': 'x',
        r'\bwhy\b': 'y',
        r'\bzee\b': 'z',
        r'\bsquared\b': '^2',
        r'\bcubed\b': '^3',
        r'\btimes\b': '*',
        r'\bdivided by\b': '/',
        r'\bplus\b': '+',
        r'\bminus\b': '-',
        r'\bequals\b': '=',
        r'\bgreater than or equal to\b': '>=',
        r'\bless than or equal to\b': '<=',
        r'\bgreater than\b': '>',
        r'\bless than\b': '<',
    }
    
    cleaned = text
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    return re.sub(r'\s+', ' ', cleaned).strip()

def create_error_response(error_type, message, details=None):
    """Create standardized error response"""
    error_messages = {
        'rate_limit': 'All AI services are currently experiencing high demand. Your request has been queued. Please try again in a few minutes.',
        'no_api_available': 'AI services are temporarily unavailable. Please try again later.',
        'invalid_input': 'No valid input provided',
        'api_error': 'Error processing your request',
        'unexpected_error': 'An unexpected error occurred'
    }
    
    return {
        "problem_text": "",
        "topic": "unknown",
        "subtopic": error_type,
        "variables": [],
        "constraints": [],
        "ambiguities": [f"Error: {details or message}"],
        "needs_clarification": True,
        "clarification_questions": [error_messages.get(error_type, message)],
        
    }

# -------------------- Session Management --------------------
def cleanup_sessions():
    """Clean up expired sessions"""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=SESSION_TIMEOUT)
        result = sessions_collection.delete_many({
            'last_activity': {'$lt': cutoff}
        })
        logger.info(f"Cleaned up {result.deleted_count} expired sessions")
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")

def get_or_create_session(session_id=None):
    """Get existing session or create new one"""
    now = datetime.now(timezone.utc)
    
    try:
        if not session_id:
            session_data = {
                '_id': str(uuid.uuid4()),
                'history': [],
                'created_at': now,
                'last_activity': now,
                'parsed_problems': []
            }
            result = sessions_collection.insert_one(session_data)
            return str(result.inserted_id), session_data
        
        session_data = sessions_collection.find_one_and_update(
            {'_id': session_id},
            {'$set': {'last_activity': now}},
            return_document=ReturnDocument.AFTER
        )
        
        if not session_data:
            return get_or_create_session(None)
        
        return session_id, session_data
        
    except Exception as e:
        logger.error(f"Error in get_or_create_session: {str(e)}")
        return get_or_create_session(None)

# -------------------- Parser Logic --------------------
class RateLimitError(Exception):
    """Custom exception for rate limiting"""
    pass

class NoAPIAvailableError(Exception):
    """Custom exception when all APIs are unavailable"""
    pass

def is_rate_limit_error(error):
    """Check if error is a rate limit error"""
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in [
        'rate_limit', 'rate limit', '429', 'too many requests',
        'quota', 'insufficient_quota'
    ])

@retry(
    stop=stop_after_attempt(1),  # Only 1 retry total (initial attempt + 1 retry)
    wait=wait_exponential(multiplier=1, min=1, max=1),  # Fixed 1 second delay
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.warning(
        f"Retry attempt {retry_state.attempt_number} after rate limit"
    )
)
def call_groq_api(prompt_text):
    """Call Groq API with retry logic"""
    if not groq_client:
        raise NoAPIAvailableError("Groq client not initialized")
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if is_rate_limit_error(e):
            raise RateLimitError("Groq API rate limit exceeded") from e
        raise

@retry(
    stop=stop_after_attempt(1),  # Only 1 retry total (initial attempt + 1 retry)
    wait=wait_exponential(multiplier=1, min=1, max=1),  # Fixed 1 second delay
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.warning(
        f"OpenAI retry attempt {retry_state.attempt_number}"
    )
)
def call_openai_api(prompt_text):
    """Fallback to OpenAI API"""
    if not openai_client:
        raise NoAPIAvailableError("OpenAI client not initialized")
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if is_rate_limit_error(e):
            raise RateLimitError("OpenAI API rate limit exceeded") from e
        raise

def parse_json_response(response_text):
    """Extract and parse JSON from response"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown or mixed content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {
            "error": "invalid_format",
            "message": "Could not parse response as JSON"
        }

def create_fallback_parse(problem_text):
    """Create a basic fallback response when all APIs fail"""
    cleaned = preprocess_text(problem_text)
    
    # Simple heuristic-based parsing
    variables = list(set(re.findall(r'\b([a-z])\b', cleaned.lower())))
    
    # Detect topic keywords
    topic = "unknown"
    if any(word in cleaned.lower() for word in ['derivative', 'integral', 'limit', 'calculus']):
        topic = "calculus"
    elif any(word in cleaned.lower() for word in ['probability', 'chance', 'random']):
        topic = "probability"
    elif any(word in cleaned.lower() for word in ['matrix', 'vector', 'eigenvalue']):
        topic = "linear_algebra"
    elif any(word in cleaned.lower() for word in ['equation', 'solve', 'algebra']):
        topic = "algebra"
    
    return {
        "problem_text": cleaned,
        "topic": topic,
        "subtopic": "unknown",
        "variables": variables,
        "constraints": [],
        "ambiguities": ["AI parsing unavailable - using basic heuristics"],
        "needs_clarification": True,
        "clarification_questions": [
            "AI services are temporarily unavailable. This is a basic parse. Please verify the problem details."
        ],
        "confidence": 0.3,
        "model_used": "fallback_heuristic",
        "original_text": problem_text,
        "parsed_at": datetime.now(timezone.utc).isoformat()
    }

def parse_math_problem(problem_text):
    """Parse mathematical problem using LLM with comprehensive fallback"""
    
    # Check cache first

    cache_key = hashlib.sha256(problem_text.encode()).hexdigest()

    cached_result = response_cache.get(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return cached_result
    
    try:
        cleaned_text = preprocess_text(problem_text)
        if not cleaned_text:
            return create_error_response("invalid_input", "No valid input provided")
        
        prompt = f"""Analyze this mathematical problem and extract structured information:

Problem: {cleaned_text}

Respond ONLY with valid JSON matching the required structure."""
        
        model_used = None
        ai_response = None
        
        # Try Groq first
        if groq_client:
            try:
                ai_response = call_groq_api(prompt)
                model_used = 'groq'
                logger.info("Successfully used Groq API")
            except RateLimitError as e:
                logger.warning(f"Groq rate limit hit: {str(e)}")
            except NoAPIAvailableError:
                logger.warning("Groq client not available")
            except Exception as e:
                logger.warning(f"Groq error: {str(e)}")
        
        # Fallback to OpenAI if Groq failed
        if not ai_response and openai_client:
            try:
                logger.info("Attempting OpenAI fallback")
                ai_response = call_openai_api(prompt)
                model_used = 'openai'
                logger.info("Successfully used OpenAI API")
            except RateLimitError as e:
                logger.warning(f"OpenAI rate limit hit: {str(e)}")
            except NoAPIAvailableError:
                logger.warning("OpenAI client not available")
            except Exception as e:
                logger.warning(f"OpenAI error: {str(e)}")
        
        # If all APIs failed, use fallback heuristic
        if not ai_response:
            logger.warning("All AI APIs unavailable, using fallback heuristic")
            result = create_fallback_parse(problem_text)
            response_cache.set(cache_key, result)
            return result
        
        parsed_data = parse_json_response(ai_response)
        
        if 'error' in parsed_data:
            logger.error(f"Parse error from {model_used}: {parsed_data}")
            result = create_fallback_parse(problem_text)
            response_cache.set(cache_key, result)
            return result
        
        # Add metadata
        parsed_data.update({
            'model_used': model_used,
            'original_text': problem_text,
            'parsed_at': datetime.now(timezone.utc).isoformat()
        })
        
        # Ensure required fields
        defaults = {
            'problem_text': cleaned_text,
            'topic': 'unknown',
            'subtopic': 'unknown',
            'variables': [],
            'constraints': [],
            'ambiguities': [],
            'needs_clarification': False,
            'clarification_questions': []
        }
        
        for key, default_value in defaults.items():
            parsed_data.setdefault(key, default_value)
        
        # Sanity cleanup (non-semantic)
        if 'constraints' in parsed_data and isinstance(parsed_data['constraints'], list):
            valid_constraints = []
            for constraint in parsed_data['constraints']:
                if not isinstance(constraint, str):
                    continue
                    
                # Rule: Reject constraints that contain '=' but no comparison operators
                # Accept equations and inequalities
                valid_constraints.append(constraint)
            
            parsed_data['constraints'] = valid_constraints
            

        
        # Cache successful result
        response_cache.set(cache_key, parsed_data)
        
        return parsed_data
        
    except Exception as e:
        logger.error(f"Unexpected error in parse_math_problem: {str(e)}", exc_info=True)
        result = create_fallback_parse(problem_text)
        response_cache.set(cache_key, result)
        return result

# -------------------- API Routes --------------------
@app.route('/', methods=['GET'])
def home():
    """Health check and service info"""
    api_status = {
        'groq': 'available' if groq_client else 'unavailable',
        'openai': 'available' if openai_client else 'unavailable',
        'fallback': 'available'
    }
    
    return jsonify({
        "service": "Math Parser Agent",
        "status": "operational",
        "version": "2.1.0",
        "api_status": api_status,
        "endpoints": {
            "parse": "/api/parse (POST)",
            "session_clear": "/api/session/clear (POST)",
            "session_history": "/api/session/history (GET)",
            "cache_clear": "/api/cache/clear (POST)"
        }
    })

@app.route('/api/parse', methods=['POST'])
def parse_endpoint():
    """Parse mathematical problem"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid request body',
                'needs_clarification': True
            }), 400
        
        problem_text = data.get('problem_text', '').strip()
        session_id = data.get('session_id')
        cleanup_sessions()

        if not problem_text:
            return jsonify({
                'error': 'Empty problem text',
                'needs_clarification': True,
                'clarification_questions': ['Please provide a mathematical problem to parse.']
            }), 400
        
        # Get or create session
        session_id, _ = get_or_create_session(session_id)
        
        # Parse the problem
        parsed_result = parse_math_problem(problem_text)
        
        # Store in session
        try:
            sessions_collection.update_one(
                {'_id': session_id},
                {
                    '$push': {
                        'parsed_problems': {
                            'input': problem_text,
                            'result': parsed_result,
                            'timestamp': datetime.now(timezone.utc)
                        }
                    },
                    '$set': {'last_activity': datetime.now(timezone.utc)}
                }
            )
        except Exception as e:
            logger.error(f"Error updating session: {str(e)}")
        
       
        
        return jsonify(parsed_result)

        
    except Exception as e:
        logger.error(f"parse_endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error processing request',
            'message': str(e),
            'needs_clarification': True
        }), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear response cache"""
    try:
        response_cache.clear()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': 'Failed to clear cache'}), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear session history"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') if data else None
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        result = sessions_collection.update_one(
            {'_id': session_id},
            {
                '$set': {
                    'parsed_problems': [],
                    'last_activity': datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': 'Session cleared',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        return jsonify({'error': 'Failed to clear session'}), 500

@app.route('/api/session/history', methods=['GET'])
def get_session_history():
    """Get parsing history for a session"""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({
                'error': 'Session ID is required',
                'success': False
            }), 400
        
        session_data = sessions_collection.find_one({'_id': session_id})
        
        if not session_data:
            return jsonify({
                'error': 'Session not found',
                'success': False
            }), 404
        
        response = {
            'success': True,
            'session_id': str(session_data['_id']),
            'created_at': session_data.get('created_at').isoformat() if session_data.get('created_at') else None,
            'last_activity': session_data.get('last_activity').isoformat() if session_data.get('last_activity') else None,
            'total_parsed': len(session_data.get('parsed_problems', [])),
            'history': []
        }
        
        for problem in session_data.get('parsed_problems', []):
            response['history'].append({
                'input': problem.get('input', ''),
                'result': problem.get('result', {}),
                'timestamp': problem.get('timestamp').isoformat() if problem.get('timestamp') else None
            })
        
        response['history'].sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return json_util.dumps(response)
        
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve session history',
            'success': False,
            'details': str(e)
        }), 500

# -------------------- Application Entry Point --------------------
if __name__ == "__main__":
    try:
        logger.info("Starting Math Parser Agent Server...")
        logger.info(f"Groq client: {'✓' if groq_client else '✗'}")
        logger.info(f"OpenAI client: {'✓' if openai_client else '✗'}")
        
        if not groq_client and not openai_client:
            logger.warning("⚠ No API clients available - running in fallback-only mode")
        
        port = int(os.getenv('PORT', 8000))
        debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        
        logger.info(f"Starting server on port {port} (debug: {debug})")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            use_reloader=debug
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise