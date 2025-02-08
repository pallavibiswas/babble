import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import time

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check if API key is missing
if not HUGGINGFACE_API_KEY:
    raise ValueError("Error: Hugging Face API key is missing!")

# Initialize Firebase
cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "babble-8e814-firebase-adminsdk-fbsvc-5aa39be796.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Default route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Speech Therapy API is running!"})

# Hugging Face API Setup
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def generate_structured_prompt(speech_issue):
    """Force the AI to generate a **fully completed** lesson plan."""
    return f"""
You are a professional speech therapist. Create a **fully detailed lesson plan** to help a person struggling with "{speech_issue}". 
Write a **fully completed** lesson that includes **real exercises**, corrections, and practical strategies.

---

### **Lesson Plan: Improving Pronunciation of "{speech_issue}"**

#### **1. Explanation**
Explain why some people struggle with pronouncing "{speech_issue}". What muscles, tongue positioning, and breathing techniques affect pronunciation?

#### **2. Warm-Up Exercises**
Provide **2-3 breathing and relaxation exercises** that help improve pronunciation. Include:
- Step-by-step instructions
- Duration for each exercise
- How they help in speech improvement

#### **3. Pronunciation Drills**
Provide **3 exercises** for pronunciation practice:
- **Phoneme Practice:** Explain how to isolate and practice specific sounds in "{speech_issue}".
- **Word Repetition:** List **at least 5 words** that reinforce correct pronunciation.
- **Sentence Drills:** Provide **3 full sentences** for pronunciation practice.

#### **4. Common Mistakes & Fixes**
List **2-3 common mistakes** and how to fix them.
| Mistake | Fix |
|---------|-----|
| Example Mistake 1 | Correction technique |
| Example Mistake 2 | How to fix it |

#### **5. Daily Practice Routine**
Provide a structured daily routine:
| Time of Day | Exercise |
|-------------|-----------|
| Morning  | Example Exercise |
| Afternoon | Example Exercise |
| Evening  | Example Exercise |

#### **6. Encouragement & Motivation**
List **2-3 motivational techniques** to help the learner stay consistent. These should include:
- Self-recording techniques
- Progress tracking strategies
- How to build confidence when speaking

---

**Ensure that your response is fully detailed and avoids placeholders. Do NOT repeat the instructions. Just write the full lesson plan.**
"""

@app.route("/generate_lesson", methods=["POST"])
def generate_lesson():
    data = request.json
    user_id = data.get("user_id", "").strip()
    user_input = data.get("speech_issue", "").strip()

    if not user_id:
        return jsonify({"error": "Missing or invalid user_id"}), 400
    if not user_input:
        return jsonify({"error": "Speech issue is required"}), 400

    prompt = generate_structured_prompt(user_input)
    payload = {"inputs": prompt}

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            break
        time.sleep(2 ** attempt)  # Exponential backoff

    if response.status_code != 200:
        print("Hugging Face API Error:", response.status_code, response.text)
        return jsonify({
            "error": "Failed to generate lesson",
            "status_code": response.status_code,
            "details": response.text
        }), 500

    try:
        lesson = response.json()[0]['generated_text']
        db.collection("lessons").document(user_id).set({"lesson": lesson})
        return jsonify({"lesson": lesson})
    except (KeyError, IndexError):
        return jsonify({"error": "Unexpected response format", "details": response.json()}), 500

@app.route("/get_lesson", methods=["GET"])
def get_lesson():
    user_id = request.args.get("user_id", "").strip()

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    doc_ref = db.collection("lessons").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return jsonify({"lesson": doc.to_dict()["lesson"]})
    return jsonify({"error": "No lesson found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
