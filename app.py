import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        return jsonify({"response": "Please enter a question or upload an image."}), 400

    q_lower = user_query.lower().strip()

    if q_lower in ["hi", "hii", "hello", "hey", "hola"]:
        return jsonify({"response": "Hello! How can I assist you with your studies today?"})

    creator_keywords = ["invented", "created", "made you", "creator", "founded", "founder", "father", "parent", "dad", "maker"]
    if any(phrase in q_lower for phrase in creator_keywords):
        if any(p in q_lower for p in ["picture", "photo", "show", "image", "pic"]):
            html_response = (
                "Here are the founders and creators of Perpranov AI:<br><br>"
                "<div style='display: flex; gap: 20px; margin-top: 10px;'>"
                "  <div style='text-align: center;'>"
                f"    <img src='{url_for('static', filename='amalan.jpg')}' style='width: 120px; height: 140px; border-radius: 8px; border: 2px solid #ff9f43; object-fit: cover;'><br>"
                "    <span style='font-size: 13px; color: #fff; font-weight: bold; display: block; margin-top: 5px;'>Mr. Amalan Machado</span><br>"
                "    <span style='font-size: 11px; color: #bbb; display: block; margin-top: 3px;'>Founder, Perpranov AI<br>Founder, Lawrence &amp; Joseph Techno</span>"
                "  </div>"
                "  <div style='text-align: center;'>"
                f"    <img src='{url_for('static', filename='nithik.jpg')}' style='width: 120px; height: 140px; border-radius: 8px; border: 2px solid #ff9f43; object-fit: cover;'><br>"
                "    <span style='font-size: 13px; color: #fff; font-weight: bold; display: block; margin-top: 5px;'>Mr. Nithik Roshan</span><br>"
                "    <span style='font-size: 11px; color: #bbb; display: block; margin-top: 3px;'>Co-Founder &amp; Designer, Perpranov AI</span>"
                "  </div>"
                "</div>"
            )

            return jsonify({"response": html_response})
        
        return jsonify({"response": "Perpranov AI was founded by Mr. Amalan Machado, Founder of Perpranov AI and of Lawrence & Joseph Techno, along with Mr. Nithik Roshan, Co-Founder and Designer of Perpranov AI."})

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

    if image_bytes and mime_type and gemini_client:
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2, max_output_tokens=600, thinking_config=types.ThinkingConfig(thinking_budget=0))
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
                    return jsonify({"response": response.text})
            except Exception as e:
                last_error = str(e)
                continue

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
            answer = response.choices.message.content
            if answer:
                return jsonify({"response": answer})
        except Exception as e:
            last_error = str(e)

    if gemini_client:
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2, max_output_tokens=600, thinking_config=types.ThinkingConfig(thinking_budget=0))
        for model_name in MODELS_LIST:
            try:
                response = gemini_client.models.generate_content(model=model_name, contents=user_query, config=config)
                if response and response.text:
                    return jsonify({"response": response.text})
            except Exception as e:
                last_error = str(e)
                continue

    return jsonify({"response": f"API Error: {last_error}"}), 500

if __name__ == '__main__':
    import sys; port = int(sys.argv[sys.argv.index("--port")+1]) if "--port" in sys.argv else 5000; app.run(host="127.0.0.1", port=9999, debug=False)
