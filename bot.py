import os
import json
import logging
import uuid
import threading
import time
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from groq import Groq
from flask_cors import CORS
from flask import render_template, send_from_directory

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_vars = ['FLASK_SECRET', 'GROQ_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# -------------------- Flask App --------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'dev-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# -------------------- File Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, 'prompt', 'system_prompt.txt')

# -------------------- GROQ / LLM Configuration --------------------
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = "llama-3.3-70b-versatile"

# Load system prompt
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

try:
    # Initialize Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("GROQ client initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing GROQ client: {str(e)}")
    groq_client = None
    raise

# -------------------- Session Management --------------------
sessions = {}
session_lock = threading.Lock()
SESSION_TIMEOUT = 1800  # 30 minutes

def cleanup_sessions():
    while True:
        now = datetime.now()
        with session_lock:
            to_delete = [sid for sid, data in sessions.items()
                         if (now - data['last_activity']).total_seconds() > SESSION_TIMEOUT]
            for sid in to_delete:
                del sessions[sid]
        time.sleep(300)

threading.Thread(target=cleanup_sessions, daemon=True).start()

def get_or_create_session(session_id=None):
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        with session_lock:
            sessions[session_id] = {'history': [], 'created_at': datetime.now(), 'last_activity': datetime.now()}
    else:
        with session_lock:
            sessions[session_id]['last_activity'] = datetime.now()
    return session_id, sessions[session_id]['history']

# -------------------- Emotional Analysis --------------------
def analyze_emotional_state(text):
    if not text: return 'neutral'
    text = text.lower()
    positive_words = ['happy','joy','excited','proud','confident','hopeful','relieved','calm','peaceful','grateful','optimistic']
    negative_words = ['sad','depressed','hopeless','worthless','empty','lonely','anxious','nervous','worried','scared','afraid','panicked','angry','frustrated','irritated','annoyed','mad','furious','stressed','overwhelmed','burned out','exhausted','tired','always','never','should','must','failure','disaster']
    pos_count = sum(1 for w in positive_words if f' {w} ' in f' {text} ')
    neg_count = sum(1 for w in negative_words if f' {w} ' in f' {text} ')
    intensity_words = {'very':1.5,'really':1.5,'extremely':2,'incredibly':2,'slightly':0.7,'a bit':0.7,'somewhat':0.8}
    for word,mult in intensity_words.items():
        if f' {word} ' in f' {text} ': pos_count*=mult; neg_count*=mult
    if 'not ' in text or "don't " in text or "can't " in text or "won't " in text:
        pos_count*=0.5; neg_count*=1.3
    if pos_count > neg_count*1.5: return 'positive'
    if neg_count > pos_count*1.5: return 'negative'
    if pos_count>0 or neg_count>0: return 'mixed'
    return 'neutral'

# -------------------- Helper Functions --------------------
def get_volunteer_suggestion():
    """Simulate volunteer suggestion based on availability"""
    volunteers = [
        "Dr. Smith (Counselor, 5 years experience)",
        "Ms. Johnson (Therapist, specializes in anxiety)",
        "Dr. Lee (Psychiatrist, available for video call)",
        "Campus Peer Counselor (Available for in-person)"
    ]
    return random.choice(volunteers)

def get_vr_environment(mood):
    """Get appropriate VR environment based on mood"""
    environments = {
        'anxious': 'calm_meadow',
        'stressed': 'beach_sunset',
        'sad': 'mountain_retreat',
        'angry': 'forest_stream',
        'default': 'peaceful_garden'
    }
    return environments.get(mood, environments['default'])

# -------------------- CBT / LLM Response --------------------
def get_cbt_response(user_input, conversation_history):
    try:
        # Check for specific commands
        if user_input.lower() in ['help', 'options']:
            return """
            How can I help you today? Here are some options:
            - Talk about my feelings
            - Schedule a session
            - Connect with a volunteer
            - Try VR relaxation
            - Emergency help
            """
        
        # Prepare context for the AI
        context = {
            'user_input': user_input,
            'conversation_history': conversation_history[-4:],  # Last 4 exchanges
            'current_time': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'suggested_volunteer': get_volunteer_suggestion(),
            'vr_environment': get_vr_environment('default')
        }
        
        # Prepare messages for the AI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
            Current Time: {context['current_time']}
            User: {user_input}
            
            Please respond as Sattva, the mental health assistant, keeping in mind:
            - Be empathetic and supportive
            - Keep responses concise
            - Offer specific help options when appropriate
            - Use the user's name if known
            """}
        ]
        
        # Add conversation history
        for msg in context['conversation_history']:
            role = "user" if msg['role'] == 'user' else "assistant"
            messages.append({"role": role, "content": msg['content']})
        
        # Call Groq API
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
            stop=None,
        )
        
        # Process the response
        ai_msg = response.choices[0].message.content.strip()
        
        # Add additional resources if needed
        if any(word in user_input.lower() for word in ['emergency', 'help', 'urgent', 'suicide']):
            ai_msg += """
            
            If you're in immediate danger or having thoughts of self-harm, 
            please contact emergency services or a crisis hotline in your country.
            """
            
        return ai_msg
        
    except Exception as e:
        logger.error(f"Error in get_cbt_response: {str(e)}", exc_info=True)
        return "I'm having trouble processing your request. Please try again or contact support if the issue persists."

def agentic_response(user_input, conversation_history):
    """
    Process user input and generate a response using the Sattva system.
    """
    if not user_input or not isinstance(user_input, str):
        return "I didn't catch that. Could you please rephrase?"
    
    # Analyze user's emotional state
    emotion = analyze_emotional_state(user_input)
    
    # Add metadata to the user's message
    user_message = {
        "role": "user",
        "content": user_input,
        "emotion": emotion,
        "timestamp": datetime.now().isoformat(),
        "source": "web"
    }
    
    # Get AI response
    ai_msg = get_cbt_response(user_input, conversation_history)
    
    # Add metadata to the AI's response
    ai_response = {
        "role": "assistant",
        "content": ai_msg,
        "type": "response",
        "timestamp": datetime.now().isoformat(),
        "suggested_actions": [
            "Schedule a session",
            "Try VR relaxation",
            "Talk to a volunteer"
        ]
    }
    
    # Update conversation history
    conversation_history.extend([user_message, ai_response])
    
    # Log the interaction (in a real app, you'd save this to a database)
    logger.info(f"Interaction - Emotion: {emotion}, Response Length: {len(ai_msg)}")
    
    return ai_msg

# -------------------- Flask Routes --------------------
@app.route('/',methods=['GET'])
def home():
    printf("API is working")

@app.route('/api/cbt', methods=['POST'])
def cbt_endpoint():
    try:
        data = request.get_json()
        user_msg = data.get('message','').strip()
        session_id = data.get('session_id')
        
        if not user_msg: 
            return jsonify({'error':'Empty message'}), 400
            
        # Get or create session
        session_id, history = get_or_create_session(session_id)
        
        # Add user message to history
        history.append({
            'role': 'user',
            'content': user_msg,
            'timestamp': datetime.now().isoformat(),
            'source': 'web'  # Could be 'web', 'mobile', 'voice', etc.
        })
        
        # Get AI response
        ai_response = agentic_response(user_msg, history)
        
        # Add AI response to history
        history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat(),
            'type': 'response'
        })
        
        # Prepare response
        response_data = {
            'reply': ai_response,
            'session_id': session_id,
            'history_length': len(history),
            'suggestions': [
                'Schedule a session',
                'Try VR relaxation',
                'Talk to a volunteer',
                'Emergency help'
            ]
        }
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"cbt_endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error processing request',
            'message': str(e),
            'suggestion': 'Please try again or contact support if the issue persists.'
        }), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        if session_id and session_id in sessions:
            with session_lock:
                # Instead of clearing, we could archive the session
                sessions[session_id]['history'] = [
                    {
                        'role': 'system',
                        'content': 'New session started',
                        'timestamp': datetime.now().isoformat()
                    }
                ]
                sessions[session_id]['last_activity'] = datetime.now()
            return jsonify({
                'status': 'success',
                'message': 'Session cleared',
                'session_id': session_id
            })
        return jsonify({'error':'Invalid session ID'}), 400
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        return jsonify({'error': 'Failed to clear session'}), 500

# -------------------- WebSocket --------------------
@socketio.on('connect')
def handle_connect(): print('Client connected')

@socketio.on('disconnect')
def handle_disconnect(): print('Client disconnected')

@socketio.on('audio_data')
def handle_audio(data): emit('audio_ack',{'status':'received'})

# -------------------- Main --------------------
# -------------------- Main --------------------
if __name__ == "__main__":
    try:
        logger.info("Starting GROQ CBT Chatbot Server...")
        # Check if running in production (Render sets the PORT environment variable)
        is_production = os.environ.get('PORT') is not None
        
        if is_production:
            logger.info("Running in production mode")
            port = int(os.environ.get('PORT', 8000))
            socketio.run(app, 
                        host="0.0.0.0", 
                        port=port, 
                        debug=False, 
                        allow_unsafe_werkzeug=True)
        else:  # Running locally
            logger.info("Running in development mode")
            socketio.run(app, 
                       host="0.0.0.0", 
                       port=8000, 
                       debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
