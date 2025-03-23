import os
import sys
import json
import uuid
import time
import threading
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your components
from src.asr.whisper_asr import WhisperASR
from src.mt.streaming_mt import StreamingTranslator
from src.tts.streaming_tts import StreamingTTS
from src.ui.language_utils import ISO_LANGUAGES

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'translation-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
sessions = {}  # Store active sessions
users = {}     # Track users and their socket IDs
session_lock = threading.Lock()

class TranslationSession:
    """Session for P2P translation between two users."""
    
    def __init__(self, session_id, user1_lang='en', user2_lang='es'):
        """Initialize a new translation session."""
        self.session_id = session_id
        self.user1_id = None
        self.user2_id = None
        self.user1_lang = user1_lang
        self.user2_lang = user2_lang
        self.created_at = time.time()
        
        # Initialize components
        self.init_components()
        
        # Text buffers
        self.user1_text = ""
        self.user2_text = ""
        self.user1_translation = ""
        self.user2_translation = ""
        
        print(f"Created session {session_id}: {user1_lang} <-> {user2_lang}")
    
    def init_components(self):
        """Initialize ASR, MT, and TTS components."""
        try:
            # User 1 -> User 2 direction
            self.asr_1to2 = WhisperASR(model_name="tiny")
            self.asr_1to2.set_language(self.user1_lang)
            
            # User 2 -> User 1 direction
            self.asr_2to1 = WhisperASR(model_name="tiny")
            self.asr_2to1.set_language(self.user2_lang)
            
            # MT models
            self.mt_1to2 = StreamingTranslator(
                source_lang=self.user1_lang,
                target_lang=self.user2_lang
            )
            
            self.mt_2to1 = StreamingTranslator(
                source_lang=self.user2_lang,
                target_lang=self.user1_lang
            )
            
            # TTS models
            self.tts_1to2 = StreamingTTS()
            self.tts_1to2.set_language(self.user2_lang)
            
            self.tts_2to1 = StreamingTTS()
            self.tts_2to1.set_language(self.user1_lang)
            
            print(f"Components initialized for session {self.session_id}")
        
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def add_user(self, user_id, position):
        """Add a user to the session at the specified position."""
        if position == 1 and not self.user1_id:
            self.user1_id = user_id
            return True
        elif position == 2 and not self.user2_id:
            self.user2_id = user_id
            return True
        return False
    
    def get_other_user(self, user_id):
        """Get the other user's ID in this session."""
        if user_id == self.user1_id:
            return self.user2_id
        elif user_id == self.user2_id:
            return self.user1_id
        return None
    
    def process_audio(self, user_id, audio_data, sample_rate=16000):
        """Process audio from a user and return translation results."""
        is_user1 = (user_id == self.user1_id)
        
        try:
            # Select components based on direction
            asr = self.asr_1to2 if is_user1 else self.asr_2to1
            mt = self.mt_1to2 if is_user1 else self.mt_2to1
            tts = self.tts_1to2 if is_user1 else self.tts_2to1
            
            # Process with ASR
            asr_result = asr.transcribe_chunk(audio_data, sample_rate)
            source_text = asr_result['full_text']
            
            # Update text buffer
            if is_user1:
                self.user1_text = source_text
            else:
                self.user2_text = source_text
            
            # Only continue if there's new text
            if not asr_result['text']:
                return {
                    'source_text': source_text,
                    'translated_text': is_user1 and self.user1_translation or self.user2_translation,
                    'new_audio': None
                }
            
            # Translate the new text
            mt_result = mt.translate_chunk(asr_result['text'], force=True)
            translated_text = mt_result['full_text']
            
            # Update translation buffer
            if is_user1:
                self.user1_translation = translated_text
            else:
                self.user2_translation = translated_text
            
            # Generate speech if there's new translated text
            audio_b64 = None
            if mt_result['text']:
                audio = tts.synthesize_speech(mt_result['text'])
                
                if audio is not None and len(audio) > 0:
                    # Convert to int16 and encode as base64
                    audio_int16 = (audio * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Return results
            return {
                'source_text': source_text,
                'translated_text': translated_text,
                'new_audio': audio_b64
            }
        
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            return {
                'source_text': is_user1 and self.user1_text or self.user2_text,
                'translated_text': is_user1 and self.user1_translation or self.user2_translation,
                'new_audio': None,
                'error': str(e)
            }

# Create basic HTML template
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/create-session', methods=['POST'])
def create_session():
    """Create a new translation session."""
    try:
        data = request.json
        user1_lang = data.get('user1_lang', 'en')
        user2_lang = data.get('user2_lang', 'es')
        
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Create session
        with session_lock:
            sessions[session_id] = TranslationSession(
                session_id=session_id,
                user1_lang=user1_lang,
                user2_lang=user2_lang
            )
        
        return jsonify({'session_id': session_id})
    
    except Exception as e:
        print(f"Error creating session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/join-session/<session_id>/<position>', methods=['GET'])
def join_session(session_id, position):
    """Join an existing session as user 1 or 2."""
    try:
        position = int(position)
        if position not in [1, 2]:
            return jsonify({'error': 'Invalid position'}), 400
        
        with session_lock:
            if session_id not in sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = sessions[session_id]
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            
            # Add user to session
            if not session.add_user(user_id, position):
                return jsonify({'error': 'Position already taken'}), 400
        
        # Return user info
        language = session.user1_lang if position == 1 else session.user2_lang
        return jsonify({
            'user_id': user_id,
            'session_id': session_id,
            'position': position,
            'language': language
        })
    
    except Exception as e:
        print(f"Error joining session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def get_languages():
    """Return the list of supported languages."""
    return jsonify(ISO_LANGUAGES)

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle a new connection."""
    print(f"Client connected: {request.sid}")
# Add this to your Socket.IO event handlers in p2p_server.py

@socketio.on('leave')
def handle_leave(data):
    """Handle a user leaving a session."""
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    
    if not session_id or not user_id:
        emit('error', {'message': 'Missing session_id or user_id'})
        return
    
    with session_lock:
        if session_id not in sessions:
            emit('error', {'message': 'Session not found'})
            return
        
        session = sessions[session_id]
        
        # Check if this user is part of the session
        if user_id != session.user1_id and user_id != session.user2_id:
            emit('error', {'message': 'User not in this session'})
            return
        
        # Remove user from session but keep the session active
        other_user = None
        if user_id == session.user1_id:
            other_user = session.user2_id
            session.user1_id = None
        elif user_id == session.user2_id:
            other_user = session.user1_id
            session.user2_id = None
        
        # Leave the room
        room = f"session_{session_id}"
        leave_room(room)
        
        print(f"User {user_id} left session {session_id}")
        
        # Notify other user if connected
        if other_user and other_user in users:
            emit('user_left', {
                'user_id': user_id
            }, to=users[other_user])
    
    # Remove from users dict
    if user_id in users:
        del users[user_id]

@socketio.on('disconnect')
def handle_disconnect():
    """Handle a client disconnection."""
    print(f"Client disconnected: {request.sid}")
    
    # Find user_id for this socket
    user_id = None
    for uid, sid in list(users.items()):
        if sid == request.sid:
            user_id = uid
            break
    
    if not user_id:
        return
    
    # Remove from users dict
    del users[user_id]
    
    # Find session this user is part of
    with session_lock:
        for session_id, session in list(sessions.items()):
            if user_id == session.user1_id or user_id == session.user2_id:
                # Get other user
                other_user = None
                if user_id == session.user1_id:
                    other_user = session.user2_id
                    session.user1_id = None
                elif user_id == session.user2_id:
                    other_user = session.user1_id
                    session.user2_id = None
                
                # Notify other user if connected
                if other_user and other_user in users:
                    emit('user_left', {
                        'user_id': user_id
                    }, to=users[other_user])
                
                # If both users gone, remove session after some time
                if not session.user1_id and not session.user2_id:
                    # Could implement a cleanup here if desired
                    pass
                
                break

@socketio.on('disconnect')
def handle_disconnect():
    """Handle a client disconnection."""
    print(f"Client disconnected: {request.sid}")
    
    # Remove from users dict
    for user_id, sid in list(users.items()):
        if sid == request.sid:
            del users[user_id]
            print(f"Removed user {user_id}")

@socketio.on('join')
def handle_join(data):
    """Handle a user joining a session."""
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    
    if not session_id or not user_id:
        emit('error', {'message': 'Missing session_id or user_id'})
        return
    
    with session_lock:
        if session_id not in sessions:
            emit('error', {'message': 'Session not found'})
            return
        
        session = sessions[session_id]
        if user_id != session.user1_id and user_id != session.user2_id:
            emit('error', {'message': 'User not in this session'})
            return
    
    # Store user's socket ID
    users[user_id] = request.sid
    
    # Join the room
    room = f"session_{session_id}"
    join_room(room)
    
    # Determine position
    position = 1 if user_id == session.user1_id else 2
    
    print(f"User {user_id} (position {position}) joined room {room}")
    
    # Notify user
    emit('joined', {
        'session_id': session_id,
        'user_id': user_id,
        'position': position,
        'other_user_connected': bool(session.user1_id and session.user2_id)
    })
    
    # Notify other user if connected
    other_user = session.get_other_user(user_id)
    if other_user and other_user in users:
        emit('user_joined', {
            'position': position
        }, room=room, skip_sid=request.sid)

@socketio.on('audio')
def handle_audio(data):
    """Process audio from a user."""
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    audio_data_b64 = data.get('audio')
    
    if not session_id or not user_id or not audio_data_b64:
        emit('error', {'message': 'Missing required data'})
        return
    
    # Get session
    with session_lock:
        if session_id not in sessions:
            emit('error', {'message': 'Session not found'})
            return
        
        session = sessions[session_id]
        if user_id != session.user1_id and user_id != session.user2_id:
            emit('error', {'message': 'User not in this session'})
            return
    
    try:
        # Decode audio
        audio_bytes = base64.b64decode(audio_data_b64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Process audio
        result = session.process_audio(user_id, audio_data)
        
        # Room ID for this session
        room = f"session_{session_id}"
        
        # Send source text update to sender
        emit('text_update', {
            'type': 'source',
            'text': result['source_text']
        })
        
        # Get recipient
        recipient_id = session.get_other_user(user_id)
        
        # Send translation and audio to recipient
        if recipient_id and recipient_id in users:
            recipient_sid = users[recipient_id]
            
            # Send translation text
            emit('text_update', {
                'type': 'translation',
                'text': result['translated_text']
            }, to=recipient_sid)
            
            # Send audio if available
            if result['new_audio']:
                emit('audio_update', {
                    'audio': result['new_audio']
                }, to=recipient_sid)
    
    except Exception as e:
        print(f"Error handling audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        emit('error', {'message': str(e)})

# Serve static files from the templates directory
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('templates', path)

# Add this new route to your Flask app in p2p_server.py

@app.route('/auto-join-session/<session_id>', methods=['GET'])
def auto_join_session(session_id):
    """Join an existing session by automatically selecting an available position."""
    try:
        with session_lock:
            if session_id not in sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = sessions[session_id]
            
            # Determine which position is available
            position = None
            if session.user1_id is None:
                position = 1
            elif session.user2_id is None:
                position = 2
            else:
                return jsonify({'error': 'Session is full'}), 400
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            
            # Add user to session
            if position == 1:
                session.user1_id = user_id
                your_language = session.user1_lang
                their_language = session.user2_lang
            else:
                session.user2_id = user_id
                your_language = session.user2_lang
                their_language = session.user1_lang
            
            # Map language codes to readable names
            language_names = {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "zh": "Chinese",
                "ja": "Japanese",
                "ko": "Korean",
                "ar": "Arabic",
                "hi": "Hindi"
                # Add more languages as needed
            }
            
            your_language_name = language_names.get(your_language, your_language)
            their_language_name = language_names.get(their_language, their_language)
        
        # Return user info
        return jsonify({
            'user_id': user_id,
            'session_id': session_id,
            'position': position,
            'your_language': your_language_name,
            'their_language': their_language_name
        })
    
    except Exception as e:
        print(f"Error auto-joining session: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Create simple index.html template
def create_index_html():
    """Create the index.html template if it doesn't exist."""
    os.makedirs('templates', exist_ok=True)
    
    html_path = os.path.join('templates', 'index.html')
    if not os.path.exists(html_path):
        with open(html_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>P2P Translation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 600px;
            margin: 0 auto;
            padding: 1rem;
        }
        .hidden {
            display: none;
        }
        .screen {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            overflow-y: auto;
            margin-bottom: 1rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
        }
        .controls {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .mic-button {
            height: 80px;
            border-radius: 40px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .mic-button:active, .mic-button.recording {
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
        }
        .message {
            max-width: 80%;
            padding: 0.75rem;
            border-radius: 0.75rem;
            margin-bottom: 0.5rem;
        }
        .message.sent {
            background-color: #dcf8c6;
            align-self: flex-end;
        }
        .message.received {
            background-color: white;
            align-self: flex-start;
        }
        .session-info {
            background-color: #e9ecef;
            padding: 0.5rem;
            border-radius: 0.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .status-dot {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.25rem;
        }
        .status-dot.connected {
            background-color: #28a745;
        }
        .status-dot.disconnected {
            background-color: #dc3545;
        }
        .language-label {
            font-size: 0.8rem;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>P2P Translation</h1>
    </div>
    
    <!-- Home Screen -->
    <div id="home-screen" class="screen container">
        <div class="d-grid gap-3">
            <button id="create-button" class="btn btn-primary btn-lg">Create New Conversation</button>
            <button id="join-button" class="btn btn-outline-primary btn-lg">Join Conversation</button>
        </div>
    </div>

    <!-- Create Screen -->
    <div id="create-screen" class="screen container hidden">
        <h2 class="mb-4">Create Conversation</h2>
        <form id="create-form">
            <div class="mb-3">
                <label for="your-language" class="form-label">Your Language</label>
                <select id="your-language" class="form-select" required></select>
            </div>
            <div class="mb-3">
                <label for="their-language" class="form-label">Their Language</label>
                <select id="their-language" class="form-select" required></select>
            </div>
            <div class="d-grid gap-2">
                <button type="submit" class="btn btn-primary">Create</button>
                <button type="button" class="btn btn-outline-secondary back-button">Back</button>
            </div>
        </form>
    </div>

    <!-- Join Screen -->
    <div id="join-screen" class="screen container hidden">
        <h2 class="mb-4">Join Conversation</h2>
        <form id="join-form">
            <div class="mb-3">
                <label for="session-id" class="form-label">Conversation Code</label>
                <input type="text" id="session-id" class="form-control form-control-lg text-center" placeholder="Enter code" required>
            </div>
            <div class="mb-3">
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="position" id="position-1" value="1" checked>
                    <label class="form-check-label" for="position-1">Position 1</label>
                </div>
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="position" id="position-2" value="2">
                    <label class="form-check-label" for="position-2">Position 2</label>
                </div>
            </div>
            <div class="d-grid gap-2">
                <button type="submit" class="btn btn-primary">Join</button>
                <button type="button" class="btn btn-outline-secondary back-button">Back</button>
            </div>
        </form>
    </div>

    <!-- Translation Screen -->
    <div id="translation-screen" class="screen container hidden">
        <div class="session-info">
            <div><strong>Session ID: </strong><span id="session-display"></span></div>
            <div>
                <span class="status-dot disconnected" id="connection-dot"></span>
                <span id="connection-status">Waiting for other user...</span>
            </div>
        </div>
        
        <div class="chat-area" id="chat-area">
            <!-- Messages will appear here -->
        </div>
        
        <div class="controls">
            <button id="mic-button" class="mic-button">Hold to Speak</button>
            
            <div id="share-section" class="mt-3">
                <p>Share this session with your conversation partner:</p>
                <div class="input-group">
                    <input type="text" id="share-link" class="form-control" readonly>
                    <button class="btn btn-outline-secondary" id="copy-button">Copy</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <script>
        // DOM Elements
        const homeScreen = document.getElementById('home-screen');
        const createScreen = document.getElementById('create-screen');
        const joinScreen = document.getElementById('join-screen');
        const translationScreen = document.getElementById('translation-screen');
        
        const createButton = document.getElementById('create-button');
        const joinButton = document.getElementById('join-button');
        const backButtons = document.querySelectorAll('.back-button');
        
        const createForm = document.getElementById('create-form');
        const joinForm = document.getElementById('join-form');
        
        const yourLanguageSelect = document.getElementById('your-language');
        const theirLanguageSelect = document.getElementById('their-language');
        const sessionIdInput = document.getElementById('session-id');
        
        const micButton = document.getElementById('mic-button');
        const chatArea = document.getElementById('chat-area');
        
        const sessionDisplay = document.getElementById('session-display');
        const connectionDot = document.getElementById('connection-dot');
        const connectionStatus = document.getElementById('connection-status');
        
        const shareLink = document.getElementById('share-link');
        const copyButton = document.getElementById('copy-button');
        
        // Session state
        let sessionId = null;
        let userId = null;
        let userPosition = null;
        let socket = null;
        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        
        // Initialize the app
        document.addEventListener('DOMContentLoaded', () => {
            // Fetch languages for dropdowns
            fetchLanguages();
            
            // Set up event listeners
            createButton.addEventListener('click', () => showScreen(createScreen));
            joinButton.addEventListener('click', () => showScreen(joinScreen));
            
            backButtons.forEach(button => {
                button.addEventListener('click', () => showScreen(homeScreen));
            });
            
            createForm.addEventListener('submit', handleCreateForm);
            joinForm.addEventListener('submit', handleJoinForm);
            
            micButton.addEventListener('mousedown', startRecording);
            micButton.addEventListener('mouseup', stopRecording);
            micButton.addEventListener('touchstart', (e) => {
                e.preventDefault();
                startRecording();
            });
            micButton.addEventListener('touchend', (e) => {
                e.preventDefault();
                stopRecording();
            });
            
            copyButton.addEventListener('click', copyShareLink);
            
            // Check URL for session info
            checkUrlParameters();
        });
        
        function showScreen(screen) {
            homeScreen.classList.add('hidden');
            createScreen.classList.add('hidden');
            joinScreen.classList.add('hidden');
            translationScreen.classList.add('hidden');
            
            screen.classList.remove('hidden');
        }
        
        async function fetchLanguages() {
            try {
                const response = await fetch('/languages');
                const languages = await response.json();
                
                // Populate dropdowns
                for (const [code, name] of Object.entries(languages)) {
                    const option1 = document.createElement('option');
                    option1.value = code;
                    option1.textContent = name;
                    
                    const option2 = document.createElement('option');
                    option2.value = code;
                    option2.textContent = name;
                    
                    yourLanguageSelect.appendChild(option1);
                    theirLanguageSelect.appendChild(option2);
                }
                
                // Default to English and Spanish
                yourLanguageSelect.value = 'en';
                theirLanguageSelect.value = 'es';
            } catch (error) {
                console.error('Error fetching languages:', error);
                alert('Failed to load languages. Please refresh the page.');
            }
        }
        
        async function handleCreateForm(e) {
            e.preventDefault();
            
            const yourLang = yourLanguageSelect.value;
            const theirLang = theirLanguageSelect.value;
            
            try {
                const response = await fetch('/create-session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        user1_lang: yourLang,
                        user2_lang: theirLang
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    sessionId = data.session_id;
                    
                    // Join as user 1
                    joinSession(sessionId, 1);
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.error || 'Failed to create session'}`);
                }
            } catch (error) {
                console.error('Error creating session:', error);
                alert('Failed to create session. Please try again.');
            }
        }
        
        async function handleJoinForm(e) {
            e.preventDefault();
            
            const session = sessionIdInput.value.trim();
            const position = document.querySelector('input[name="position"]:checked').value;
            
            if (!session) {
                alert('Please enter a session ID');
                return;
            }
            
            joinSession(session, parseInt(position));
        }
        
        async function joinSession(session, position) {
            try {
                const response = await fetch(`/join-session/${session}/${position}`);
                
                if (response.ok) {
                    const data = await response.json();
                    
                    sessionId = data.session_id;
                    userId = data.user_id;
                    userPosition = data.position;
                    
                    // Connect to socket
                    connectSocket();
                    
                    // Show translation screen
                    showScreen(translationScreen);
                    
                    // Update UI
                    sessionDisplay.textContent = sessionId;
                    
                    // Generate share link
                    const shareUrl = `${window.location.origin}?session=${sessionId}&position=${userPosition === 1 ? 2 : 1}`;
                    shareLink.value = shareUrl;
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.error || 'Failed to join session'}`);
                }
            } catch (error) {
                console.error('Error joining session:', error);
                alert('Failed to join session. Please try again.');
            }
        }
        
        function connectSocket() {
            // Disconnect existing socket if any
            if (socket) {
                socket.disconnect();
            }
            
            // Connect to Socket.IO server
            socket = io();
            
            // Set up event handlers
            socket.on('connect', () => {
                console.log('Connected to server');
                
                // Join session room
                socket.emit('join', {
                    session_id: sessionId,
                    user_id: userId
                });
            });
            
            socket.on('disconnect', () => {
                console.log('Disconnected from server');
                connectionDot.classList.remove('connected');
                connectionDot.classList.add('disconnected');
                connectionStatus.textContent = 'Disconnected from server';
            });
            
            socket.on('joined', (data) => {
                console.log('Joined session:', data);
                
                if (data.other_user_connected) {
                    connectionDot.classList.remove('disconnected');
                    connectionDot.classList.add('connected');
                    connectionStatus.textContent = 'Other user connected';
                }
            });
            
            socket.on('user_joined', (data) => {
                console.log('Other user joined:', data);
                connectionDot.classList.remove('disconnected');
                connectionDot.classList.add('connected');
                connectionStatus.textContent = 'Other user connected';
            });
            
            socket.on('text_update', (data) => {
                console.log('Text update:', data);
                updateChatMessages(data);
            });
            
            socket.on('audio_update', (data) => {
                console.log('Audio update received');
                playAudio(data.audio);
            });
            
            socket.on('error', (data) => {
                console.error('Socket error:', data.message);
                alert(`Error: ${data.message}`);
            });
        }
        
        function updateChatMessages(data) {
            if (data.type === 'source') {
                // Update my message
                let messageEl = document.querySelector('.message.sent');
                
                if (!messageEl) {
                    messageEl = document.createElement('div');
                    messageEl.className = 'message sent';
                    chatArea.appendChild(messageEl);
                }
                
                messageEl.textContent = data.text;
            } else if (data.type === 'translation') {
                // Update their message
                let messageEl = document.querySelector('.message.received');
                
                if (!messageEl) {
                    messageEl = document.createElement('div');
                    messageEl.className = 'message received';
                    chatArea.appendChild(messageEl);
                }
                
                messageEl.textContent = data.text;
            }
            
            // Scroll to bottom
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        async function startRecording() {
            if (isRecording || !socket || !socket.connected) return;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.addEventListener('dataavailable', event => {
                    audioChunks.push(event.data);
                });
                
                mediaRecorder.addEventListener('stop', () => {
                    processAudioChunks();
                });
                
                mediaRecorder.start();
                isRecording = true;
                
                // Update UI
                micButton.classList.add('recording');
                micButton.textContent = 'Release to Stop';
            } catch (error) {
                console.error('Error starting recording:', error);
                alert('Failed to access microphone. Please ensure your browser has permission to use the microphone.');
            }
        }
        
        function stopRecording() {
            if (!isRecording || !mediaRecorder) return;
            
            mediaRecorder.stop();
            isRecording = false;
            
            // Update UI
            micButton.classList.remove('recording');
            micButton.textContent = 'Hold to Speak';
        }
        
        async function processAudioChunks() {
            if (audioChunks.length === 0) return;
            
            try {
                // Create blob from chunks
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                // Convert to Float32Array
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioData = await decodeAudioData(arrayBuffer);
                
                // Convert to base64
                const audio_b64 = arrayBufferToBase64(audioData.buffer);
                
                // Send to server
                socket.emit('audio', {
                    session_id: sessionId,
                    user_id: userId,
                    audio: audio_b64
                });
            } catch (error) {
                console.error('Error processing audio:', error);
            }
        }
        
        async function decodeAudioData(arrayBuffer) {
            // Create audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Decode audio data
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Get first channel data
            const audioData = audioBuffer.getChannelData(0);
            
            // Resample to 16kHz if needed
            if (audioBuffer.sampleRate !== 16000) {
                return resampleAudio(audioData, audioBuffer.sampleRate, 16000);
            }
            
            return audioData;
        }
        
        function resampleAudio(audioData, originalSampleRate, targetSampleRate) {
            const ratio = targetSampleRate / originalSampleRate;
            const newLength = Math.floor(audioData.length * ratio);
            const result = new Float32Array(newLength);
            
            for (let i = 0; i < newLength; i++) {
                const index = Math.floor(i / ratio);
                result[i] = audioData[index];
            }
            
            return result;
        }
        
        function arrayBufferToBase64(buffer) {
            const bytes = new Uint8Array(buffer);
            const binary = bytes.reduce((acc, byte) => acc + String.fromCharCode(byte), '');
            return window.btoa(binary);
        }
        
        function playAudio(audio_b64) {
            if (!audio_b64) return;
            
            try {
                // Decode base64
                const binaryString = window.atob(audio_b64);
                const bytes = new Uint8Array(binaryString.length);
                
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                // Convert to audio buffer
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = new Int16Array(bytes.buffer);
                const floatBuffer = new Float32Array(audioBuffer.length);
                
                // Convert Int16 to Float32
                for (let i = 0; i < audioBuffer.length; i++) {
                    floatBuffer[i] = audioBuffer[i] / 32767.0;
                }
                
                // Create buffer source
                const bufferSource = audioContext.createBuffer(1, floatBuffer.length, 16000);
                bufferSource.getChannelData(0).set(floatBuffer);
                
                // Play audio
                const source = audioContext.createBufferSource();
                source.buffer = bufferSource;
                source.connect(audioContext.destination);
                source.start();
            } catch (error) {
                console.error('Error playing audio:', error);
            }
        }
        
        function copyShareLink() {
            shareLink.select();
            document.execCommand('copy');
            alert('Link copied to clipboard!');
        }
        
        function checkUrlParameters() {
            const urlParams = new URLSearchParams(window.location.search);
            const session = urlParams.get('session');
            const position = urlParams.get('position');
            
            if (session) {
                sessionIdInput.value = session;
                
                if (position && (position === '1' || position === '2')) {
                    document.getElementById(`position-${position}`).checked = true;
                }
                
                showScreen(joinScreen);
            }
        }
    </script>
</body>
</html>""")
        print(f"Created index.html in templates directory")

def main():
    """Main function to run the server."""
    # Create HTML template
    create_index_html()
    
    # Run the server
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting server on port {port}")
    print(f"Visit http://localhost:{port} in your browser")
    socketio.run(app, debug=True, host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()

    # At the end of the main() function, before socketio.run:
