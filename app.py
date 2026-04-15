from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Ollama runs locally on your Mac
OLLAMA_API = "http://localhost:11434/api/generate"

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

SYSTEM_PROMPTS = {
    "helpful_assistant": "You are a helpful assistant. Be clear and concise.",
    "code_expert": "You are an expert programmer. Provide clear code examples.",
    "creative_writer": "You are a creative storyteller. Use vivid language.",
}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        system_key = data.get("system_prompt", "helpful_assistant")
        temperature = data.get("temperature", 0.7)
        
        if not user_message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        if system_key not in SYSTEM_PROMPTS:
            return jsonify({"error": f"Unknown prompt: {system_key}"}), 400
        
        system_prompt = SYSTEM_PROMPTS[system_key]
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        # Call Ollama
        response = requests.post(
            OLLAMA_API,
            json={
                "model": "mistral",
                "prompt": full_prompt,
                "temperature": temperature,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return jsonify({"error": "Ollama error"}), 500
        
        result = response.json()
        assistant_message = result.get("response", "No response")
        
        return jsonify({
            "response": assistant_message,
            "system_prompt": system_key,
            "model": "mistral (free!)"
        })
    
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama not running! Open Ollama app on your Mac"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5002)