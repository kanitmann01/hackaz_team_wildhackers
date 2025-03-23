# VoiceBridge: Real-Time P2P Translation

<div align="center">

[![Made With][made-with-shield]][made-with-url]
[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]

<img src="https://raw.githubusercontent.com/kanitmann01/hackaz_team_wildhackers/refs/heads/main/logo.png" alt="VoiceBridge Logo" width="200" height="200"/>

**Break language barriers with real-time speech translation**

[View Demo](https://voicebridge.dev) ·
[Report Bug](https://github.com/kanitmann01/hackaz_team_wildhackers/issues) ·
[Explore Documentation](https://github.com/kanitmann01/hackaz_team_wildhackers)

</div>

## 🌟 Highlights

- **Real-time Translation**: Seamless speech-to-speech translation with minimal latency
- **No Third-Party Services**: Fully on-device processing for enhanced privacy
- **12+ Languages**: Support for major world languages including English, Spanish, French, Chinese, Hindi, and more
- **Web-Based Interface**: Works on any device with a modern browser
- **Peer-to-Peer Architecture**: Direct communication with no server-side conversation processing

## 📋 Table of Contents

- [About The Project](#about-the-project)
  - [How It Works](#how-it-works)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🚀 About The Project

VoiceBridge tackles one of humanity's oldest challenges: the language barrier. Our solution enables real-time conversation between people speaking different languages by leveraging cutting-edge AI models for speech recognition, translation, and speech synthesis.

Whether for international business meetings, educational exchanges, tourist interactions, or connecting with family members who speak different languages, VoiceBridge creates a seamless translation experience that feels natural and immediate.

### How It Works

1. **Speech Capture**: The speaker's voice is captured in real-time through their device's microphone
2. **Speech Recognition (ASR)**: The audio is converted to text in the source language
3. **Translation (MT)**: The text is translated to the target language
4. **Speech Synthesis (TTS)**: The translated text is converted to speech
5. **Real-time Playback**: The translated speech is played to the listener

All of this happens in near real-time, with optimizations to minimize latency while maintaining translation quality.

### Built With

- **Frontend**:
  - HTML5, CSS3, JavaScript
  - Socket.IO (WebSockets)
  
- **Backend**:
  - Python 3.9.21
  - Flask + Socket.IO
  - PyTorch
  
- **AI Models**:
  - [Whisper](https://github.com/openai/whisper) (ASR)
  - [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) (Translation)
  - [MMS-TTS](https://github.com/facebook/fairseq/tree/main/examples/mms) (Speech Synthesis)
  
- **Development Tools**:
  - Gradio (Component Testing Interface)
  - SoundDevice (Audio Processing)

## 💻 Getting Started

To get a local copy up and running, follow these steps:

### Prerequisites

- Python 3.9 or higher
- PyTorch 1.12.0 or higher
- CUDA-capable GPU (optional but recommended)

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/kanitmann01/hackaz_team_wildhackers.git
   cd hackaz_team_wildhackers
   ```

2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

3. Download the AI models (first run will download them automatically)
   ```sh
   python scripts/download_model.py
   ```

4. Start the server
   ```sh
   python p2p_server.py
   ```

5. In a separate terminal, start the UI
   ```sh
   python src/ui/app.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:7860
   ```

## 📱 Usage

### Starting a Conversation

1. Choose "Create New Conversation" on the home screen
2. Select your language and your conversation partner's language
3. Share the generated session code with your partner
4. Start speaking while holding the mic button

### Joining a Conversation

1. Choose "Join Conversation" on the home screen
2. Enter the session code shared with you
3. You're connected! Hold the mic button to speak

### Using the Translation Interface

- **Hold-to-Speak**: Press and hold the microphone button while speaking, then release when done
- **Text Fallback**: Switch to text input if the environment is too noisy
- **Quick Phrases**: Tap common phrases for instant translation
- **Audio Levels**: Visual indicators show when audio is being received or played

## ✨ Features

### Core Features

- **Streaming Speech Processing**: Processes speech in small chunks for real-time translation
- **Multi-language Support**: Translate between 12+ languages with high accuracy
- **Context Maintenance**: Preserves conversation context for coherent translation
- **Error Resilience**: Gracefully handles network issues and audio problems
- **Low Latency**: Optimized for minimal delay between speaking and hearing translations

### User Experience

- **Audio Level Visualization**: See when audio is being captured or played
- **Connection Quality Indicators**: Visual feedback on connection status
- **Session Code Sharing**: Easy to share and join conversation sessions
- **Mobile-Friendly Design**: Works well on smartphones and tablets
- **Quick Phrases**: Built-in common phrases for frequent scenarios

## 🏗️ Architecture

VoiceBridge uses a modular pipeline architecture connecting three primary AI components:

```
┌───────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│ Audio     │    │ Speech       │    │ Machine       │    │ Text-to-     │
│ Capture ──┼───►│ Recognition ─┼───►│ Translation ──┼───►│ Speech       │
│           │    │              │    │               │    │              │
└───────────┘    └──────────────┘    └───────────────┘    └──────────────┘
       ▲                                                          │
       │                                                          ▼
       │                                                    ┌──────────────┐
       └────────────────────────────────────────────────────┤ Audio        │
                                                            │ Playback     │
                                                            └──────────────┘
```

- **Socket.IO Server**: Manages communication between peers
- **Audio Processing**: Handles real-time audio capture and playback
- **Pipeline Manager**: Orchestrates the flow between ASR, MT, and TTS components
- **Diagnostics Tool**: Provides testing and debugging capabilities for each component

## 🔮 Roadmap

- [ ] Add support for more languages (goal: 25+ languages)
- [ ] Implement offline mode with downloadable models
- [ ] Create native mobile applications
- [ ] Add conversation recording and transcription
- [ ] Integrate speaker recognition for multi-speaker scenarios
- [ ] Implement end-to-end encryption for enhanced privacy

See the [open issues](https://github.com/kanitmann01/hackaz_team_wildhackers/issues) for a list of proposed features and known issues.

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgements

- [Whisper](https://github.com/openai/whisper) by OpenAI
- [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) by Meta AI
- [MMS-TTS](https://github.com/facebook/fairseq/tree/main/examples/mms) by Meta AI
- [Socket.IO](https://socket.io/)
- [Gradio](https://www.gradio.app/)

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/kanitmann01/hackaz_team_wildhackers.svg?style=for-the-badge
[contributors-url]: https://github.com/kanitmann01/hackaz_team_wildhackers/graphs/contributors
[license-shield]: https://img.shields.io/github/license/kanitmann01/hackaz_team_wildhackers.svg?style=for-the-badge
[license-url]: https://github.com/kanitmann01/hackaz_team_wildhackers/blob/main/LICENSE
[made-with-shield]: https://img.shields.io/badge/Made%20with-Python%20%26%20PyTorch-blue?style=for-the-badge
[made-with-url]: https://pytorch.org/