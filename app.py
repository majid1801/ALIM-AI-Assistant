import os
import time

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

client = genai.Client(api_key=API_KEY)


# ==============================
# ALIM PERSONALITY
# ==============================

ALIM_INSTRUCTIONS = """
You are ALIM, a personal AI assistant.

Your name is ALIM.

IDENTITY:
- Always introduce yourself as ALIM.
- Never introduce yourself as Gemini.
- If someone asks "Who are you?", say you are ALIM, their personal AI assistant.
- If someone asks what technology powers you, explain that you use Google's Gemini AI technology.
- Never pretend to be a human.

PERSONALITY:
- Friendly
- Intelligent
- Helpful
- Respectful
- Clear
- Natural
- Professional

LANGUAGE:
- Understand Bengali and English.
- Understand Bengali written in English letters.
- Reply in the same language style the user uses.
- Keep answers concise unless the user asks for details.

IMPORTANT:
You are ALIM, not Gemini.
"""


# ==============================
# GEMINI MODELS
# ==============================

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite"
]

# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# CHAT
# ==============================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "reply": "ALIM could not understand the request."
            }), 400

        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "reply": "Please type a message first."
            }), 400


        # Try models one by one
        last_error = None

        for model in MODELS:

            try:

                response = client.models.generate_content(

                    model=model,

                    contents=message,

                    config=types.GenerateContentConfig(
                        system_instruction=ALIM_INSTRUCTIONS
                    )
                )

                if response.text:
                    return jsonify({
                        "reply": response.text
                    })

            except Exception as e:

                last_error = e

                print(
                    f"Model {model} failed: {e}"
                )

                # Wait before trying next model
                time.sleep(1)


        # All models failed
        print("ALL MODELS FAILED:", last_error)

        return jsonify({
            "reply": (
                "ALIM is temporarily busy. "
                "Please try again in a few seconds."
            )
        }), 503


    except Exception as e:

        print("CHAT ERROR:", e)

        return jsonify({
            "reply": "ALIM encountered a temporary problem."
        }), 500


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":
    app.run(debug=True)