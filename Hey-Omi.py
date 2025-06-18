from flask import Flask, request, jsonify
import logging
import time
import threading
from collections import defaultdict
import google.generativeai as genai
from pathlib import Path
import openai
import anthropic
from abc import ABC, abstractmethod
from typing import Optional

# === AI PROVIDER CONFIGURATION ===
# Change the model for each provider here as needed.
AI_CONFIG = {
    "gemini": {
        # Gemini API Key
        "api_key": "Your-API-Here",
        # Change the model name below to switch Gemini models
        "model": "gemini-2.0-flash"
    },
    "openai": {
        # OpenAI API Key
        "api_key": "your-openai-key-here",  # Replace with your OpenAI key
        # Change the model name below to switch OpenAI models
        "model": "gpt-3.5-turbo"  # e.g. "gpt-4o", "gpt-4-turbo", etc.
    },
    "anthropic": {
        # Anthropic API Key
        "api_key": "your-anthropic-key-here",  # Replace with your Anthropic key
        # Change the model name below to switch Anthropic models
        "model": "claude-3-opus-20240229"  # e.g. "claude-3-5-sonnet-latest"
    }
}
# === END AI PROVIDER CONFIGURATION ===

# AI Provider Interface
class AIProvider(ABC):
    @abstractmethod
    def get_response(self, text: str) -> str:
        pass

class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def get_response(self, text: str) -> str:
        try:
            response = self.model.generate_content(text, generation_config={
                "temperature": 0.7,
                "candidate_count": 1
            })
            answer = response.text.strip()
            return answer.replace('*', '').replace('**', '').replace('`', '').replace('#', '')
        except Exception as e:
            logging.error(f"Gemini error: {str(e)}")
            return "I'm sorry, I encountered an error processing your request."

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def get_response(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Omi, a helpful AI assistant. Provide clear, concise, and friendly responses."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI error: {str(e)}")
            return "I'm sorry, I encountered an error processing your request."

class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_response(self, text: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            logging.error(f"Anthropic error: {str(e)}")
            return "I'm sorry, I encountered an error processing your request."

class AIManager:
    def __init__(self):
        self.providers = {}
        self.setup_providers()

    def setup_providers(self):
        if AI_CONFIG["gemini"]["api_key"]:
            self.providers["gemini"] = GeminiProvider(
                AI_CONFIG["gemini"]["api_key"],
                AI_CONFIG["gemini"]["model"]
            )
        
        if AI_CONFIG["openai"]["api_key"] != "your-openai-key-here":
            self.providers["openai"] = OpenAIProvider(
                AI_CONFIG["openai"]["api_key"],
                AI_CONFIG["openai"]["model"]
            )
        
        if AI_CONFIG["anthropic"]["api_key"] != "your-anthropic-key-here":
            self.providers["anthropic"] = AnthropicProvider(
                AI_CONFIG["anthropic"]["api_key"],
                AI_CONFIG["anthropic"]["model"]
            )

    def get_response(self, text: str, provider: str = "gemini") -> str:
        if provider not in self.providers:
            return f"AI provider '{provider}' is not configured."
        return self.providers[provider].get_response(text)

ai_manager = AIManager()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRIGGER_PHRASES = ["hey omi", "hey, omi"]
PARTIAL_FIRST = ["hey", "hey,"]
PARTIAL_SECOND = ["omi"]

class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def get_buffer(self, session_id):
        current_time = time.time()
        
        # Cleanup old sessions periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_sessions()
        
        with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {
                    'messages': [],
                    'trigger_detected': False,
                    'trigger_time': 0,
                    'collected_question': [],
                    'response_sent': False,
                    'partial_trigger': False,
                    'partial_trigger_time': 0,
                    'last_activity': current_time
                }
            else:
                self.buffers[session_id]['last_activity'] = current_time
                
        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600  # Remove sessions older than 1 hour
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time

message_buffer = MessageBuffer()
notification_cooldowns = defaultdict(float)
NOTIFICATION_COOLDOWN = 10  # 10 seconds cooldown between notifications
start_time = time.time()

def get_ai_response(text: str, provider: str = "gemini") -> str:
    """Get response from the specified AI provider"""
    try:
        logger.info(f"Sending question to {provider}: {text}")
        response = ai_manager.get_response(text, provider)
        logger.info(f"Received response from {provider}: {response}")
        return response
    except Exception as e:
        logger.error(f"Error getting {provider} response: {str(e)}")
        return "I'm sorry, I encountered an error processing your request."

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        logger.info("Received webhook POST request")
        data = request.json
        logger.info(f"Received data: {data}")
        
        session_id = data.get('session_id')
        uid = request.args.get('uid')
        # Get AI provider from query params, default to gemini
        provider = request.args.get('provider', 'gemini')
        
        logger.info(f"Processing request for session_id: {session_id}, uid: {uid}, provider: {provider}")
        
        if not session_id:
            logger.error("No session_id provided in request body")
            return jsonify({"status": "error", "message": "No session_id provided in request body"}), 400
        
        current_time = time.time()
        buffer_data = message_buffer.get_buffer(session_id)
        has_processed = False
        
        # Process each segment in the transcript
        segments = data.get('segments', [])
        for segment in segments:
            if not segment.get('text') or has_processed:
                continue
                
            text = segment['text'].lower().strip()
            logger.info(f"Processing text segment: '{text}'")
            
            # Check for complete trigger phrases with question
            for trigger in TRIGGER_PHRASES:
                if trigger in text:
                    question = text.split(trigger)[-1].strip()
                    if question:
                        logger.info(f"Found trigger and question: {question}")
                        response = get_ai_response(question, provider)
                        return jsonify({"message": response}), 200
            
            # No immediate question found, check for trigger only
            if any(trigger in text for trigger in TRIGGER_PHRASES) and not buffer_data['trigger_detected']:
                logger.info(f"Complete trigger phrase detected in session {session_id}")
                buffer_data['trigger_detected'] = True
                buffer_data['trigger_time'] = current_time
                notification_cooldowns[session_id] = current_time
                continue
            
            # If trigger was detected, this must be the question
            if buffer_data['trigger_detected'] and not buffer_data['response_sent']:
                logger.info(f"Processing question after trigger: {text}")
                response = get_ai_response(text, provider)
                
                # Reset states
                buffer_data['trigger_detected'] = False
                buffer_data['response_sent'] = True
                has_processed = True
                
                return jsonify({"message": response}), 200
        
        return jsonify({"status": "success"}), 200

@app.route('/webhook/setup-status', methods=['GET'])
def setup_status():
    return jsonify({"is_setup_completed": True}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "active_sessions": len(message_buffer.buffers),
        "uptime": time.time() - start_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
