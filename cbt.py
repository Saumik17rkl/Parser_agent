import os
import json
import logging
import uuid
import threading
from flask import Flask, request, jsonify, session, render_template
from flask_socketio import SocketIO
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from dotenv import load_dotenv
from datetime import datetime, timedelta, timedelta as td

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_vars = [
    'FLASK_SECRET',
    'WATSONX_API_KEY',
    'WATSONX_URL',
    'WATSONX_PROJECT_ID'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# -------------------- Flask App --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['FLASK_SECRET']
app.config['PERMANENT_SESSION_LIFETIME'] = td(days=1)

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# -------------------- WatsonX Configuration --------------------
WATSONX_API_KEY = os.environ["WATSONX_API_KEY"]
WATSONX_URL = os.environ["WATSONX_URL"]
WATSONX_PROJECT = os.environ["WATSONX_PROJECT_ID"]
WATSONX_MODEL = "meta-llama/llama-3-3-70b-instruct"
PROMPT_TEMPLATE = ""

def load_prompt_template():
    try:
        with open('prompt/system_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt template: {str(e)}")
        return ""

# Initialize WatsonX client
try:
    logger.info("Initializing WatsonX client...")
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    llm_model = ModelInference(
        model_id=WATSONX_MODEL,
        credentials={
            "apikey": WATSONX_API_KEY,
            "url": WATSONX_URL
        },
        project_id=WATSONX_PROJECT,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 250,
            "min_new_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2
        }
    )
    logger.info("WatsonX client initialized successfully!")
    PROMPT_TEMPLATE = load_prompt_template()
    if not PROMPT_TEMPLATE:
        raise ValueError("Failed to load prompt template")
except Exception as e:
    logger.error(f"Error initializing WatsonX client: {str(e)}")
    llm_model = None
    raise

# Session management
sessions = {}
session_lock = threading.Lock()
SESSION_TIMEOUT = 1800  # 30 minutes of inactivity

def cleanup_sessions():
    """Remove inactive sessions"""
    while True:
        current_time = datetime.now()
        with session_lock:
            to_remove = [sid for sid, data in sessions.items() 
                        if (current_time - data['last_activity']).total_seconds() > SESSION_TIMEOUT]
            for sid in to_remove:
                del sessions[sid]
        time.sleep(300)  # Check every 5 minutes

# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

# -------------------- Emotional Analysis --------------------
def analyze_emotional_state(text):
    if not text:
        return 'neutral'
    text = text.lower()
    positive_words = ['happy', 'joy', 'excited', 'proud', 'confident', 'hopeful', 'relieved', 'calm', 'peaceful', 'grateful', 'optimistic']
    negative_words = ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'lonely', 'anxious', 'nervous', 'worried', 'scared', 'afraid', 'panicked', 'angry', 'frustrated', 'irritated', 'annoyed', 'mad', 'furious', 'stressed', 'overwhelmed', 'burned out', 'exhausted', 'tired', 'always', 'never', 'should', 'must', 'failure', 'disaster']
    
    positive_count = sum(1 for word in positive_words if f' {word} ' in f' {text} ')
    negative_count = sum(1 for word in negative_words if f' {word} ' in f' {text} ')
    
    intensity_words = {'very': 1.5, 'really': 1.5, 'extremely': 2, 'incredibly': 2, 'slightly': 0.7, 'a bit': 0.7, 'somewhat': 0.8}
    for word, multiplier in intensity_words.items():
        if f' {word} ' in f' {text} ':
            positive_count *= multiplier
            negative_count *= multiplier
    
    if 'not ' in text or "don't " in text or "can't " in text or "won't " in text:
        positive_count *= 0.5
        negative_count *= 1.3
    
    if positive_count > negative_count * 1.5:
        return 'positive'
    elif negative_count > positive_count * 1.5:
        return 'negative'
    elif positive_count > 0 or negative_count > 0:
        return 'mixed'
    return 'neutral'

# -------------------- CBT Response --------------------
CBT_PROMPT = """You are {bot_name}, a professional IPT Therapist. Help users identify negative thoughts and develop coping strategies.

Current conversation history:
{chat_history}

User: {user_input}
Therapist:"""

def get_cbt_response(user_input, conversation_history):
    try:
        chat_history = "\n".join([
            f"{'User' if msg['role']=='user' else 'Therapist'}: {msg['content']}" 
            for msg in conversation_history[-6:]
        ])
        prompt = CBT_PROMPT.format(
            bot_name="Dr. Smith",
            chat_history=chat_history,
            user_input=user_input
        )
        if llm_model is None:
            return "Technical difficulties. Please try again later."
        
        response = llm_model.generate(prompt=prompt)
        if isinstance(response, dict):
            ai_message = response.get('results', [{}])[0].get('generated_text', '').strip()
        else:
            ai_message = str(response).strip()
        if 'Therapist:' in ai_message:
            ai_message = ai_message.split('Therapist:')[-1].strip()
        return ai_message
    except Exception as e:
        logger.error(f"Error in get_cbt_response: {str(e)}", exc_info=True)
        return "Error processing your request. Please try again."

def agentic_response(user_input, conversation_history):
    if not user_input or not isinstance(user_input, str):
        return "I didn't catch that. Could you rephrase?"
    try:
        emotion = analyze_emotional_state(user_input)
        conversation_history.append({"role": "user", "content": user_input, "emotion": emotion})
        ai_message = get_cbt_response(user_input, conversation_history)
        conversation_history.append({"role": "assistant", "content": ai_message, "type": "cbt_response"})
        return ai_message
    except Exception as e:
        logger.error(f"Error in agentic_response: {str(e)}", exc_info=True)
        return "Error processing your request. Please try again."

def get_or_create_session(session_id=None):
    """Get or create a new session"""
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        with session_lock:
            sessions[session_id] = {
                'history': [],
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
    else:
        with session_lock:
            sessions[session_id]['last_activity'] = datetime.now()
    
    return session_id, sessions[session_id]['history']

# -------------------- Flask Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cbt', methods=['POST'])
def cbt_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get or create session
        session_id, conversation_history = get_or_create_session(session_id)
        
        # Add user message to history
        conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get AI response
        ai_response = agentic_response(user_message, conversation_history)
        
        # Add AI response to history
        conversation_history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'reply': ai_response,
            'session_id': session_id,
            'history_length': len(conversation_history)
        })
        
    except Exception as e:
        logger.error(f"Error in cbt_endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error processing request', 'details': str(e)}), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear session data"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in sessions:
            with session_lock:
                sessions[session_id]['history'] = []
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Invalid session'}), 400
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    return jsonify({'status': 'success'})

# -------------------- WebSocket Handlers --------------------
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_data')
def handle_audio(data):
    emit('audio_ack', {'status': 'received'})

# -------------------- Main --------------------
def main():
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>AI Voice Chat • Circle UI</title>
            <style>
                :root {
                --bg: #0b0f14;
                --panel: #0f1621;
                --accent: #6ee7ff;
                --accent-2: #a78bfa;
                --text: #e8eef7;
                --muted: #9fb2c8;
                --danger: #ef4444;
                --ok: #10b981;
                }
                * { box-sizing: border-box; }
                html, body { height: 100%; }
                body {
                margin: 0;
                background: radial-gradient(1200px 800px at 50% 30%, #101827 0%, var(--bg) 60%);
                color: var(--text);
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji";
                display: grid;
                grid-template-rows: 1fr auto;
                gap: 24px;
                }
                .wrap { display: grid; place-items: center; padding: 24px; }

                /* Center Circle */
                .circle {
                width: clamp(180px, 22vmin, 320px);
                aspect-ratio: 1/1;
                border-radius: 50%;
                background: radial-gradient(65% 65% at 50% 40%, #1d2840, #0b1220 70%);
                border: 2px solid rgba(255,255,255,0.08);
                box-shadow:
                    0 0 0 2px rgba(174, 223, 255, 0.05) inset,
                    0 10px 40px rgba(0,0,0,0.6),
                    0 0 80px rgba(110,231,255,0.12);
                display: grid; place-items: center;
                position: relative;
                cursor: pointer;
                user-select: none;
                transition: transform .2s ease;
                }
                .circle:hover { transform: scale(1.02); }

                .pulse, .pulse::before, .pulse::after {
                content: "";
                position: absolute; inset: 0;
                border-radius: 50%;
                pointer-events: none;
                }
                .pulse { box-shadow: 0 0 0 0 rgba(110,231,255,0.20); }

                /* Listening animation */
                .listening .pulse {
                animation: pulse 1.6s ease-out infinite;
                }
                @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(110,231,255,0.28); }
                70% { box-shadow: 0 0 0 30px rgba(110,231,255,0.00); }
                100% { box-shadow: 0 0 0 0 rgba(110,231,255,0.00); }
                }

                /* Speaking animation */
                .speaking .pulse {
                animation: speak 1s ease-in-out infinite;
                }
                @keyframes speak {
                0%, 100% { box-shadow: 0 0 0 6px rgba(167,139,250,0.28); }
                50% { box-shadow: 0 0 0 18px rgba(167,139,250,0.00); }
                }

                .mic-icon {
                width: 34%;
                aspect-ratio: 1/1;
                display: grid; place-items: center;
                border-radius: 50%;
                background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.00));
                border: 1px solid rgba(255,255,255,0.06);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35), 0 0 20px rgba(110,231,255,0.12) inset;
                backdrop-filter: blur(6px);
                font-size: clamp(28px, 4vmin, 48px);
                }

                .hint { margin-top: 16px; color: var(--muted); font-size: 0.95rem; text-align: center; }

                .panel {
                display: none;  
                width: min(980px, calc(100% - 24px));
                margin: 0 auto 20px auto;
                background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 16px 16px 8px 16px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.35);
                }

                .row { display: grid; gap: 12px; grid-template-columns: 1fr; }
                @media (min-width: 760px) {
                .row { grid-template-columns: 1fr 1fr; }
                }

                label { font-size: 0.9rem; color: var(--muted); }
                select, input[type="range"], button, .monobox {
                width: 100%;
                background: #0e1523;
                color: var(--text);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 10px 12px;
                outline: none;
                }
                select:focus, input[type="range"]:focus, button:focus { border-color: var(--accent); }

                .monobox { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; min-height: 72px; }

                .status { display: flex; gap: 8px; align-items: center; font-size: 0.95rem; }
                .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--muted); box-shadow: 0 0 0 2px rgba(255,255,255,0.05) inset; }
                .dot.ok { background: var(--ok); }
                .dot.busy { background: var(--accent-2); }
                .dot.err { background: var(--danger); }

                .btnrow { display: flex; gap: 10px; flex-wrap: wrap; }
                button { cursor: pointer; border-radius: 12px; padding: 10px 14px; border: 1px solid rgba(255,255,255,0.08); }
                .primary { background: linear-gradient(90deg, rgba(110,231,255,0.25), rgba(167,139,250,0.25)); }
                .ghost { background: rgba(255,255,255,0.04); }
            </style>
            </head>
            <body>
            <div class="wrap">
                <div id="circle" class="circle" aria-label="Tap to talk" role="button">
                <div class="pulse"></div>
                <div class="mic-icon" id="micIcon">🎤</div>
                </div>
                <div class="hint" id="hint">Tap to talk • Tap again to stop</div>
            </div>

            <div class="panel">
                <div class="row">
                <div>
                    <label for="voiceSelect">Voice: High-pitched Female</label>
                    <select id="voiceSelect" style="display: none;"></select>
                </div>
                <div class="row">
                    <div>
                    <label for="rate">Rate</label>
                    <input type="range" id="rate" min="0.7" max="1.4" step="0.05" value="1" />
                    </div>
                    <div>
                    <label for="pitch">Pitch</label>
                    <input type="range" id="pitch" min="0.7" max="1.5" step="0.05" value="1" />
                    </div>
                    <div>
                    <label for="volume">Volume</label>
                    <input type="range" id="volume" min="0" max="1" step="0.05" value="1" />
                    </div>
                </div>
                </div>

                <div class="row" style="margin-top:12px;">
                <div>
                    <label>User said</label>
                    <div id="youSaid" class="monobox"></div>
                </div>
                <div>
                    <label>AI replied</label>
                    <div id="aiSaid" class="monobox"></div>
                </div>
                </div>

                <div class="row" style="margin-top:12px;">
                <div class="status" id="status"><span class="dot" id="statusDot"></span><span id="statusText">idle</span></div>
                <div class="btnrow">
                    <button class="ghost" id="stopTTS">Stop Voice</button>
                    <button class="ghost" id="clearLog">Clear</button>
                    <button class="primary" id="retryLast">Replay Last Reply</button>
                </div>
                </div>

                <details style="margin-top:12px;">
                <summary>Backend contract (for <code>cbt.py</code>)</summary>
                <pre class="monobox" style="margin-top:8px;">
            POST /api/cbt HTTP/1.1
            Content-Type: application/json

            {
            "message": "string",       // user transcript
            "session_id": "string"     // optional session tracking
            }

            Response 200
            {
            "reply": "string"          // AI's text response (CBT-guided)
            }
                </pre>
                </details>
            </div>

            <script>
                // ====== Utilities ======
                const qs = (sel) => document.querySelector(sel);
                const circle = qs('#circle');
                const youSaid = qs('#youSaid');
                const aiSaid = qs('#aiSaid');
                const voiceSelect = qs('#voiceSelect');
                const statusDot = qs('#statusDot');
                const statusText = qs('#statusText');
                const rateEl = qs('#rate');
                const pitchEl = qs('#pitch');
                const volumeEl = qs('#volume');

                const sessionId = crypto.randomUUID();

                function setStatus(kind, text) {
                statusDot.className = 'dot ' + (kind || '');
                statusText.textContent = text || '';
                }

                // ====== STT (Web Speech API) ======
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognizer = SpeechRecognition ? new SpeechRecognition() : null;
                let listening = false;

                if (recognizer) {
                recognizer.lang = 'en-US';
                recognizer.interimResults = true;
                recognizer.continuous = true;

                let finalTranscript = '';

                recognizer.onstart = () => {
                    finalTranscript = '';
                    setStatus('busy', 'listening…');
                    circle.classList.add('listening');
                };

                recognizer.onresult = (event) => {
                    let interim = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                    const res = event.results[i];
                    if (res.isFinal) finalTranscript += res[0].transcript + ' ';
                    else interim += res[0].transcript;
                    }
                    youSaid.textContent = (finalTranscript + '\n' + (interim ? '(' + interim + ')' : '')).trim();
                };

                recognizer.onerror = (e) => {
                    setStatus('err', 'mic error: ' + e.error);
                    circle.classList.remove('listening');
                    listening = false;
                };

                recognizer.onend = async () => {
                    circle.classList.remove('listening');
                    listening = false;
                    const text = youSaid.textContent.replace(/\([^)]*\)$/,'').trim();
                    if (text) await sendToBackend(text);
                    else setStatus('', 'idle');
                };
                } else {
                setStatus('err', 'SpeechRecognition not supported in this browser');
                }

                async function toggleListening() {
                if (!recognizer) return;
                if (!listening) {
                    window.speechSynthesis.cancel();
                    aiSaid.textContent = '';
                    recognizer.start();
                    listening = true;
                } else {
                    recognizer.stop();
                }
                }

                circle.addEventListener('click', toggleListening);

                // ====== TTS (Web Speech API) ======
                const synth = window.speechSynthesis;
                let voices = [];
                // Set default values for voice settings - higher pitch and faster rate
                rateEl.value = '1.2';  // Slightly faster
                pitchEl.value = '1.4';  // Higher pitch
                volumeEl.value = '1';   // Full volume

                function findBestFemaleVoice(voices) {
                    // Try to find a high-pitched female voice
                    const preferredVoices = [
                        'Samantha', 'Ava', 'Karen', 'Tessa', 'Victoria', 
                        'Serena', 'Allison', 'Zoe', 'Zira', 'Melina'
                    ];
                    
                    // First try to find an exact match from preferred voices
                    for (const name of preferredVoices) {
                        const voice = voices.find(v => v.name.includes(name));
                        if (voice) return voice;
                    }
                    
                    // If no exact match, find any female voice
                    const femaleVoices = voices.filter(v => 
                        v.name.toLowerCase().includes('female') || 
                        v.lang.startsWith('en-') && 
                        (v.name.toLowerCase().includes('samantha') ||
                         v.name.toLowerCase().includes('ava') ||
                         v.name.toLowerCase().includes('karen'))
                    );
                    
                    return femaleVoices[0] || voices[0];
                }

                function populateVoices() {
                    voices = synth.getVoices();
                    if (!voices || !voices.length) return;
                    
                    // Hide the voice selector since we're using a fixed voice
                    voiceSelect.style.display = 'none';
                    document.querySelector('label[for="voiceSelect"]').textContent = 'Voice: High-pitched Female';
                    
                    // Find and set the best female voice
                    const bestVoice = findBestFemaleVoice(voices);
                    if (bestVoice) {
                        pickedVoices = [bestVoice];
                        voiceSelect.innerHTML = '';
                        const opt = document.createElement('option');
                        opt.value = '0';
                        opt.textContent = `${bestVoice.name} (${bestVoice.lang})`;
                        voiceSelect.appendChild(opt);
                    }
                }

                populateVoices();
                if (typeof speechSynthesis !== 'undefined') {
                speechSynthesis.onvoiceschanged = populateVoices;
                }

                function speak(text) {
                if (!text) return;
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance(text);
                const idx = parseInt(voiceSelect.value, 10);
                if (!isNaN(idx) && pickedVoices[idx]) utter.voice = pickedVoices[idx];
                utter.rate = parseFloat(rateEl.value);
                utter.pitch = parseFloat(pitchEl.value);
                utter.volume = parseFloat(volumeEl.value);
                utter.onstart = () => { setStatus('busy', 'speaking…'); circle.classList.add('speaking'); };
                utter.onend = () => { setStatus('', 'idle'); circle.classList.remove('speaking'); };
                utter.onerror = () => { setStatus('err', 'TTS error'); circle.classList.remove('speaking'); };
                synth.speak(utter);
                }

                qs('#stopTTS').addEventListener('click', () => { synth.cancel(); setStatus('', 'idle'); circle.classList.remove('speaking'); });
                qs('#retryLast').addEventListener('click', () => speak(lastReply));
                qs('#clearLog').addEventListener('click', () => { youSaid.textContent = ''; aiSaid.textContent = ''; setStatus('', 'idle'); });

                // ====== Backend wiring ======
                async function sendToBackend(message) {
                setStatus('busy', 'thinking…');
                try {
                    const res = await fetch('/api/cbt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, session_id: sessionId })
                    });
                    if (!res.ok) throw new Error('HTTP ' + res.status);
                    const data = await res.json();
                    const reply = (data && (data.reply || data.text)) || '';
                    aiSaid.textContent = reply;
                    lastReply = reply;
                    speak(reply);
                    setStatus('ok', 'reply ready');
                } catch (err) {
                    console.error(err);
                    setStatus('err', 'backend error');
                }
                }

                // ====== Permissions hint (mic) ======
                (async () => {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
                try {
                    // Pre-warm mic permissions for smoother first run (ignored by some browsers)
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    stream.getTracks().forEach(t => t.stop());
                } catch (_) { /* user may deny; we handle on start */ }
                })();
            </script>
            </body>
            </html>""")
    logger.info("Starting CBT Therapy Server...")
    socketio.run(app, debug=True, use_reloader=True)

if __name__ == "__main__":
    main()