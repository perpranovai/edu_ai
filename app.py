import os
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

app = Flask(__name__)

# Configure API Key
API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6JxZbAQCGeOztAQscZY7_ahbxUGqzFtG8c8-FOh1pcB2A")
client = genai.Client(api_key=API_KEY)

# Auto-detect all active Flash models available to your account
def get_active_models():
    candidates = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    try:
        available = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "flash" in m.name
        ]
        if available:
            return available
    except Exception as e:
        print(f"[Warning] Could not list models: {e}")
    return candidates

MODELS_LIST = get_active_models()
print(f"[*] Active Models Loaded: {MODELS_LIST}")

SYSTEM_INSTRUCTION = (
    "Your name is Perpranov AI. "
    "You are an expert academic tutor. "
    "Always double-check calculations and facts before answering. "
    "If anyone asks your name, say: 'I am Perpranov AI.'"
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    data = request.get_json() or {}
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"response": "Please enter a question."}), 400

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
    )

    last_error = ""
    # Automatically try active models in order
    for model_name in MODELS_LIST:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_query,
                config=config
            )
            if response and response.text:
                return jsonify({"response": response.text})
        except Exception as e:
            last_error = str(e)
            print(f"[-] {model_name} failed: {last_error}")
            continue

    if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
        return jsonify({"response": "⚠️ Free request quota exhausted for today. Please create a new key at aistudio.google.com."})

    return jsonify({"response": f"Error: {last_error}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
