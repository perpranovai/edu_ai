import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6LDHlIh2SNrl2fEIboF2-oQX2Nbq9YLJyBIcrhbMWAbTw")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def get_active_gemini_models():
    candidates = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash"]
    if not gemini_client:
        return candidates
    try:
        available = [
            m.name.replace("models/", "")
            for m in gemini_client.models.list()
            if "flash" in m.name
        ]
        if available:
            return available
    except Exception as e:
        print(f"[Warning] Could not list Gemini models: {e}")
    return candidates

MODELS_LIST = get_active_gemini_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    user_query = ""
    if request.is_json:
        data = request.get_json() or {}
        user_query = data.get("query", "").strip()
    else:
        user_query = request.form.get("query", "").strip()

    uploaded_file = request.files.get("image")
    image_bytes = None
    mime_type = None
    if uploaded_file:
        image_bytes = uploaded_file.read()
        mime_type = uploaded_file.mimetype

    if not user_query and not image_bytes:
        return jsonify({"response": "Please enter a question or upload an image.", "speech": "Please enter a question or upload an image."}), 400

    q_lower = user_query.lower().strip()

    # 1. Simple greetings
    if q_lower in ["hi", "hii", "hello", "hey", "hola"]:
        msg = "Hello! How can I assist you with your studies today?"
        return jsonify({"response": msg, "speech": msg})

    # 2. Keywords and stem matching (catches inventor, invenor, creator, ceator, founder)
    creator_stems = [
        "inv", "creat", "ceat", "found", "maker", "made you", "who made",
        "father", "parent", "owner", "developer", "dev"
    ]
    image_stems = [
        "pic", "photo", "image", "img", "face", "look", "see", "show"
    ]

    is_creator_query = any(stem in q_lower for stem in creator_stems)
    is_image_query = any(stem in q_lower for stem in image_stems)

    # SHOW PICTURES CASE:
    if is_creator_query and is_image_query:
        return jsonify({
            "response": "Here are the founders and creators of Perpranov AI:",
            "show_creators": True,
            "amalan_img": url_for('static', filename='amalan.jpg'),
            "nithik_img": url_for('static', filename='nithik.jpg'),
            "speech": "Mr. Amalan Machado and Mr. Nithik Roshan."
        })

    # WHO CREATED YOU CASE (Text only):
    if is_creator_query and any(w in q_lower for w in ["who", "what", "you", "your", "name"]):
        msg = "Perpranov AI was invented and founded by Mr. Amalan Machado (founder of Lawrence Joseph Techno) along with Mr. Nithik Roshan."
        return jsonify({"response": msg, "speech": msg})

    current_time_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    system_instruction = (
        f"Your name is Perpranov AI. You are an expert academic tutor. "
        f"You were completely invented, created, and founded by Mr. Amalan Machado (founder of Lawrence Joseph Techno) along with Mr. Nithik Roshan. "
        f"If asked about your identity, creator, inventor, father, or origin, you must always state that you were developed by Mr. Amalan Machado and Mr. Nithik Roshan. "
        f"Never say you were created by Google, OpenAI, or anyone else. "
        f"Current date and time: {current_time_str}. "
        f"Provide direct, accurate, concise, and helpful responses."
    )

    last_error = "API request failed."

    # Strategy 1: Gemini Vision
    if image_bytes and mime_type and gemini_client:
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
        for model_name in MODELS_LIST:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        user_query if user_query else "Analyze this image and explain it clearly."
                    ],
                    config=config
                )
                if response and response.text:
                    return jsonify({"response": response.text, "speech": response.text})
            except Exception as e:
                last_error = str(e)
                continue

    # Strategy 2: OpenAI Text
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query or "Hello"}
                ],
                temperature=0.2,
                max_tokens=800
            )
            answer = response.choices[0].message.content
            if answer:
                return jsonify({"response": answer, "speech": answer})
        except Exception as e:
            last_error = str(e)

    # Strategy 3: Gemini Fallback
    if gemini_client:
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
        for model_name in MODELS_LIST:
            try:
                response = gemini_client.models.generate_content(model=model_name, contents=user_query, config=config)
                if response and response.text:
                    return jsonify({"response": response.text, "speech": response.text})
            except Exception as e:
                last_error = str(e)
                continue

    return jsonify({"response": f"API Error: {last_error}", "speech": "An error occurred."}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
