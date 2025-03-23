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
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global variables
sessions = {}  # Store active sessions
users = {}     # Track users and their socket IDs
session_lock = threading.Lock()

class TranslationSession:
    """Session for P2P translation between two users."""
    
    def __init__(self, session_id, user1_lang='hi', user2_lang='en'):
        """Initialize a new translation session."""
        self.session_id = session_id
        self.user1_id = None
        self.user2_id = None
        self.user1_lang = user1_lang
        self.user2_lang = user2_lang
        self.user1_text = ""  # Should be reset or replaced each time, not appended
        self.user2_text = ""
        self.user1_translation = ""
        self.user2_translation = ""
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
    
    def process_text(self, user_id, text):
        """Process text input from a user and return translation results."""
        is_user1 = (user_id == self.user1_id)
        
        try:
            # Select components based on direction
            mt = self.mt_1to2 if is_user1 else self.mt_2to1
            tts = self.tts_1to2 if is_user1 else self.tts_2to1
            
            # Update text buffer
            if is_user1:
                self.user1_text = text
            else:
                self.user2_text = text
            
            # Translate the text
            mt_result = mt.translate_chunk(text, force=True)
            translated_text = mt_result['full_text']
            
            # Update translation buffer
            if is_user1:
                self.user1_translation = translated_text
            else:
                self.user2_translation = translated_text
            
            # Generate speech for the translation
            audio_b64 = None
            if translated_text:
                audio = tts.synthesize_speech(translated_text)
                
                if audio is not None and len(audio) > 0:
                    # Convert to int16 and encode as base64
                    audio_int16 = (audio * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Return results
            return {
                'source_text': text,
                'translated_text': translated_text,
                'new_audio': audio_b64
            }
        
        except Exception as e:
            print(f"Error processing text: {str(e)}")
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
        user1_lang = data.get('user1_lang', 'hi')
        user2_lang = data.get('user2_lang', 'en')
        
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

@app.route('/languages', methods=['GET'])
def get_languages():
    """Return the list of supported languages."""
    return jsonify(ISO_LANGUAGES)

# Serve static files from the templates directory
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('templates', path)

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle a new connection."""
    print(f"Client connected: {request.sid}")

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
        # Reset ASR and MT contexts for new utterance
        if user_id == session.user1_id:
            session.asr_1to2.reset_context()
            session.mt_1to2.reset_context()
        else:
            session.asr_2to1.reset_context()
            session.mt_2to1.reset_context()
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_data_b64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Process audio
        result = session.process_audio(user_id, audio_data)
        
        # Rest of the function remains unchanged...
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

@socketio.on('text')
def handle_text(data):
    """Process text input from a user."""
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    text = data.get('text')
    
    if not session_id or not user_id or not text:
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
        # Process text
        result = session.process_text(user_id, text)
        
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
        print(f"Error handling text: {str(e)}")
        import traceback
        print(traceback.format_exc())
        emit('error', {'message': str(e)})

@socketio.on('ping')
def handle_ping(data):
    """Handle ping request."""
    # Simply respond to complete the round trip
    emit('pong', {})

# Create simple index.html template
def create_index_html():
    """Create the index.html template if it doesn't exist."""
    os.makedirs('templates', exist_ok=True)
    
    html_path = os.path.join('templates', 'index.html')
    if not os.path.exists(html_path):
        print("Creating index.html template...")
        # The content will be created from the first HTML artifact

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