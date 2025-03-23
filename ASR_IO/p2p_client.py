import os
import sys
import gradio as gr
import numpy as np
import threading
import base64
import requests
import socketio
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your components
from src.ui.language_utils import ISO_LANGUAGES
from src.tts.streaming_tts import StreamingTTS

# Global variables
sio = socketio.Client()
server_url = "http://localhost:5000"  # Default server URL
session_id = None
user_id = None
current_source_text = ""
current_translated_text = ""
is_recording = False
is_connected = False
audio_playback = None  # Will store the audio playback component

# Initialize TTS component for local playback
tts = None  # Will be initialized on first use

# Socket.IO event handlers
@sio.event
def connect():
    global is_connected
    is_connected = True
    print("Connected to server")

@sio.event
def disconnect():
    global is_connected
    is_connected = False
    print("Disconnected from server")

@sio.event
def joined(data):
    print(f"Successfully joined session: {data}")

@sio.event
def text_update(data):
    global current_source_text, current_translated_text
    
    print(f"Received text update: {data}")
    
    if data["type"] == "source":
        current_source_text = data["text"]
    elif data["type"] == "translation":
        current_translated_text = data["text"]

@sio.event
def audio_update(data):
    print("Received audio update")
    
    # Play audio locally
    if data.get("audio"):
        play_audio(data["audio"])

# Helper functions
def create_session(server_url, your_language, their_language):
    """Create a new translation session."""
    global session_id, user_id
    
    # Convert language names to ISO codes
    your_iso = get_iso_code(your_language)
    their_iso = get_iso_code(their_language)
    
    if not your_iso or not their_iso:
        return f"Error: Couldn't recognize language codes for {your_language} and {their_language}", None, None
    
    try:
        # Make API request to create session
        response = requests.post(
            f"{server_url}/create-session",
            json={"user1_lang": your_iso, "user2_lang": their_iso},
            timeout=10
        )
        
        if response.status_code != 200:
            return f"Error: {response.text}", None, None
        
        data = response.json()
        session_id = data["session_id"]
        
        # Join as user 1
        return join_session(server_url, session_id, "1")
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

def join_session(server_url, session_id, position):
    """Join an existing translation session."""
    global user_id
    
    try:
        # Make API request to join session
        response = requests.get(
            f"{server_url}/join-session/{session_id}/{position}",
            timeout=10
        )
        
        if response.status_code != 200:
            return f"Error: {response.text}", None, None
        
        data = response.json()
        user_id = data["user_id"]
        
        # Set up socket connection
        if sio.connected:
            sio.disconnect()
        
        sio.connect(server_url)
        sio.emit("join", {"session_id": session_id, "user_id": user_id})
        
        # Generate share link for the other user
        other_position = "2" if position == "1" else "1"
        share_link = f"{server_url}?session={session_id}&position={other_position}"
        
        return f"Successfully joined session {session_id} as user {position}", session_id, share_link
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

def get_iso_code(language_name):
    """Convert a language name to ISO code."""
    for iso, name in ISO_LANGUAGES.items():
        if name.lower() == language_name.lower():
            return iso
    return None

def record_audio(duration=2.0):
    """Record audio from the microphone."""
    if not is_connected or not session_id or not user_id:
        return "Not connected to a session"
    
    try:
        import sounddevice as sd
        
        # Audio parameters
        sample_rate = 16000
        channels = 1
        
        # Record audio
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete
        
        # Flatten and normalize audio
        audio_mono = audio_data.flatten()
        
        # Check audio level
        audio_level = np.mean(np.abs(audio_mono))
        if audio_level < 0.01:  # Too quiet
            return "Audio too quiet, please speak louder"
        
        # Convert to base64
        audio_bytes = audio_mono.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Send to server
        sio.emit("audio", {
            "session_id": session_id,
            "user_id": user_id,
            "audio": audio_b64
        })
        
        return "Audio sent successfully"
    
    except Exception as e:
        return f"Error recording audio: {str(e)}"

def start_recording():
    """Start continuous recording."""
    global is_recording
    
    if is_recording:
        return "Already recording"
    
    if not is_connected or not session_id or not user_id:
        return "Not connected to a session"
    
    is_recording = True
    threading.Thread(target=recording_thread, daemon=True).start()
    return "Recording started"

def stop_recording():
    """Stop continuous recording."""
    global is_recording
    
    if not is_recording:
        return "Not recording"
    
    is_recording = False
    return "Recording stopped"

def recording_thread():
    """Thread function for continuous recording."""
    global is_recording
    
    try:
        import sounddevice as sd
        
        # Audio parameters
        sample_rate = 16000
        channels = 1
        chunk_duration = 1.0  # seconds
        chunk_size = int(sample_rate * chunk_duration)
        
        # Start recording stream
        with sd.InputStream(samplerate=sample_rate, channels=channels, blocksize=chunk_size) as stream:
            while is_recording:
                # Read audio chunk
                audio_chunk, overflowed = stream.read(chunk_size)
                
                if overflowed:
                    print("Audio input overflow")
                
                # Process audio
                audio_mono = audio_chunk.flatten()
                
                # Check audio level
                audio_level = np.mean(np.abs(audio_mono))
                if audio_level < 0.01:  # Too quiet
                    continue
                
                # Convert to base64
                audio_bytes = audio_mono.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Send to server
                sio.emit("audio", {
                    "session_id": session_id,
                    "user_id": user_id,
                    "audio": audio_b64
                })
    
    except Exception as e:
        print(f"Error in recording thread: {str(e)}")
        is_recording = False

def play_audio(audio_b64):
    """Play audio from base64 string using local TTS component."""
    global tts, audio_playback
    
    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_b64)
        
        # Convert to float32 array
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Initialize TTS component if not already initialized
        if tts is None:
            # Use your existing TTS component
            device = None  # Use default device
            tts = StreamingTTS(device=device)
        
        # Play audio using your audio playback component
        if audio_playback:
            audio_playback.play(audio_data)
        else:
            # Fallback to saving and playing from file
            import wave
            import sounddevice as sd
            
            # Save to temporary file
            temp_file = "temp_audio.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(16000)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            # Play the file
            try:
                sd.play(audio_data, 16000)
                sd.wait()
            except Exception as e:
                print(f"Error playing audio: {e}")
    
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def get_status():
    """Get current connection status."""
    if not is_connected:
        return "Disconnected"
    elif not session_id:
        return "Connected to server, but not in a session"
    else:
        return f"Connected to session {session_id}"

def get_source_text():
    """Get current source text."""
    return current_source_text

def get_translated_text():
    """Get current translated text."""
    return current_translated_text

# UI functions
def handle_create_click(server_url, your_language, their_language):
    """Handle create button click."""
    status, sid, share = create_session(server_url, your_language, their_language)
    return status, sid or "", share or ""

def handle_join_click(server_url, session_id, position):
    """Handle join button click."""
    status, sid, share = join_session(server_url, session_id, position)
    return status, sid or "", share or ""

def handle_record_start():
    """Handle record button press."""
    return start_recording()

def handle_record_stop():
    """Handle record button release."""
    return stop_recording()

# Create the UI
def create_ui():
    with gr.Blocks(title="P2P Translation") as app:
        with gr.Tabs() as tabs:
            with gr.Tab("Create Session"):
                with gr.Group():
                    gr.Markdown("## Create New Translation Session")
                    
                    server_url_create = gr.Textbox(
                        label="Server URL",
                        value="http://localhost:5000",
                        placeholder="Enter server URL"
                    )
                    
                    with gr.Row():
                        your_language = gr.Dropdown(
                            choices=list(ISO_LANGUAGES.values()),
                            label="Your Language",
                            value="English"
                        )
                        
                        their_language = gr.Dropdown(
                            choices=list(ISO_LANGUAGES.values()),
                            label="Their Language",
                            value="Spanish"
                        )
                    
                    create_btn = gr.Button("Create Session")
                    
                    create_status = gr.Textbox(
                        label="Status",
                        value="Ready"
                    )
                    
                    session_display = gr.Textbox(
                        label="Session ID",
                        value="",
                        interactive=False
                    )
                    
                    share_link = gr.Textbox(
                        label="Share Link",
                        value="",
                        interactive=False
                    )
                    
                    # Event handler
                    create_btn.click(
                        fn=handle_create_click,
                        inputs=[server_url_create, your_language, their_language],
                        outputs=[create_status, session_display, share_link]
                    )
            
            with gr.Tab("Join Session"):
                with gr.Group():
                    gr.Markdown("## Join Existing Session")
                    
                    server_url_join = gr.Textbox(
                        label="Server URL",
                        value="http://localhost:5000",
                        placeholder="Enter server URL"
                    )
                    
                    session_id_input = gr.Textbox(
                        label="Session ID",
                        placeholder="Enter session ID"
                    )
                    
                    position_input = gr.Radio(
                        choices=["1", "2"],
                        label="Position",
                        value="2"
                    )
                    
                    join_btn = gr.Button("Join Session")
                    
                    join_status = gr.Textbox(
                        label="Status",
                        value="Ready"
                    )
                    
                    # Event handler
                    join_btn.click(
                        fn=handle_join_click,
                        inputs=[server_url_join, session_id_input, position_input],
                        outputs=[join_status, gr.Textbox(visible=False), gr.Textbox(visible=False)]
                    )
            
            with gr.Tab("Translation"):
                with gr.Group():
                    gr.Markdown("## Real-Time Translation")
                    
                    connection_status = gr.Textbox(
                        label="Connection Status",
                        value=get_status(),
                        interactive=False
                    )
                    
                    with gr.Row():
                        source_textbox = gr.Textbox(
                            label="Your Speech",
                            placeholder="Speak to see your words here...",
                            value=get_source_text(),
                            lines=5,
                            interactive=False
                        )
                        
                        translated_textbox = gr.Textbox(
                            label="Translated Speech",
                            placeholder="Translated speech will appear here...",
                            value=get_translated_text(),
                            lines=5,
                            interactive=False
                        )
                    
                    with gr.Row():
                        mic_btn = gr.Button("Hold to Speak", variant="primary")
                    
                    recording_status = gr.Textbox(
                        label="Recording Status",
                        value="Not recording"
                    )
                    
                    # Event handlers for record button
                    mic_btn.click(
                        fn=handle_record_start,
                        outputs=[recording_status]
                    )
                    
                    # We can't easily do press-and-hold with Gradio, so we'll add a stop button
                    stop_btn = gr.Button("Stop Speaking")
                    stop_btn.click(
                        fn=handle_record_stop,
                        outputs=[recording_status]
                    )
                    
                    # Update text boxes periodically
                    app.load(
                        fn=get_source_text,
                        inputs=None,
                        outputs=source_textbox,
                        every=1
                    )
                    
                    app.load(
                        fn=get_translated_text,
                        inputs=None,
                        outputs=translated_textbox,
                        every=1
                    )
                    
                    app.load(
                        fn=get_status,
                        inputs=None,
                        outputs=connection_status,
                        every=3
                    )
        
        # Initialize audio components
        try:
            from src.audio.streaming_audio import StreamingAudioPlayback
            global audio_playback
            audio_playback = StreamingAudioPlayback()
            audio_playback.start()
        except Exception as e:
            print(f"Error initializing audio playback: {e}")
    
    return app

# Main function
def main():
    # Create temp directory for audio files
    os.makedirs("temp", exist_ok=True)
    
    # Create the UI
    app = create_ui()
    
    # Launch the app
    app.launch(share=True)
    
    # Clean up on exit
    if audio_playback:
        audio_playback.stop()
    
    if sio.connected:
        sio.disconnect()

if __name__ == "__main__":
    main()