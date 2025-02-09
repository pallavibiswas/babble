import os
import requests
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap
import time

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Ensure API key exists
if not HUGGINGFACE_API_KEY:
    raise ValueError("Error: Hugging Face API key is missing!")

# Initialize Firebase
cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "babble-8e814-firebase-adminsdk-fbsvc-052260f205.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Model Selection
PRIMARY_MODEL = "tiiuae/falcon-7b-instruct"
BACKUP_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def generate_structured_prompt(speech_issue):
    """Generates a structured and complete lesson plan for a speech issue."""
    return f"""
You are a professional speech therapist. Write a **fully structured and detailed lesson plan** for improving pronunciation of '{speech_issue}'.

---

### **Lesson Plan: Improving Pronunciation of '{speech_issue}'**

#### **1. Goals and Objectives**
- Improve pronunciation of '{speech_issue}'.
- Develop confidence in speaking.
- Recognize and correct common errors.

#### **2. Assessment Methods**
- Student records 10 words before and after the session.
- Instructor evaluates pronunciation improvements.

#### **3. Required Resources**
- Pronunciation handouts
- Audio recordings
- Speech recognition software

#### **4. Activities & Exercises**
1. **Warm-Up (15 min):** Listen and mimic recordings.
2. **Sound Differentiation (20 min):** Compare '{speech_issue}' with similar sounds.
3. **Articulation Training (20 min):** Tongue and lip placement drills.
4. **Vocabulary Practice (15 min):** Read and practice common words.
5. **Conversational Practice (10 min):** Structured dialogue practice.
6. **Self-Evaluation (10 min):** Compare recordings & get instructor feedback.

### **5. Daily Practice Routine**
| Time | Exercise |
|------|---------|
| Morning | Phoneme drills |
| Afternoon | Word repetition |
| Evening | Conversational practice |

### **6. Confidence Building**
- **Self-Recording:** Track progress.
- **Goal Setting:** Weekly check-ins.
- **Confidence Exercises:** Mirror practice.

🚨 **Your response must be a fully structured and detailed lesson plan. DO NOT include disclaimers, bullet points, or outlines. Write in complete sentences with clear explanations for each section. Provide example words and exercises.** 🚨
"""

def request_ai_response(prompt, model):
    """Requests AI-generated text from Hugging Face API with error handling."""
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 4096,
                "temperature": 0.7,  # Increased temperature for more creative responses
                "top_p": 0.9,  # Slightly lowered top_p to keep focus
                "return_full_text": False
            }
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=HEADERS,
            json=payload
        )

        print(f"API Attempt {attempt + 1} (Model: {model}): Status Code {response.status_code}")

        if response.status_code == 200:
            try:
                response_json = response.json()
                print("Full AI Response:", response_json)

                # Validate AI response format
                if isinstance(response_json, list) and len(response_json) > 0:
                    lesson_text = response_json[0].get("generated_text", "").strip()
                    
                    # Remove duplicate sections
                    lines = lesson_text.split("\n")
                    seen = set()
                    filtered_lines = []
                    for line in lines:
                        if line.strip() not in seen:
                            seen.add(line.strip())
                            filtered_lines.append(line)
                    lesson_text = "\n".join(filtered_lines)

                    # Fix Markdown table formatting
                    lesson_text = lesson_text.replace("|------|---------|", "| Time | Exercise |\n|------|---------|")

                    # Ensure valid output
                    if len(lesson_text.split()) > 50 and lesson_text != "---":
                        return lesson_text
                    else:
                        print("Error: AI response too short or invalid:", lesson_text)

            except Exception as e:
                print("Error processing AI response:", str(e))

        time.sleep(2 ** attempt)  # Exponential backoff

    return None

def generate_pdf(lesson_text, pdf_path):
    """Creates a properly formatted PDF using ReportLab."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    margin = 40
    y_position = height - 50

    c.setFont("Helvetica", 12)

    for line in lesson_text.split("\n"):
        wrapped_lines = wrap(line, width=90)
        for wrapped_line in wrapped_lines:
            if y_position < 50:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - 50
            c.drawString(margin, y_position, wrapped_line)
            y_position -= 18

    c.save()

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

    # Try primary model first
    lesson_text = request_ai_response(prompt, PRIMARY_MODEL)

    # If Falcon-7B fails, fall back to Mistral
    if not lesson_text:
        print("⚠️ Falcon-7B failed. Trying Mistral-7B instead...")
        lesson_text = request_ai_response(prompt, BACKUP_MODEL)

    if not lesson_text:
        return jsonify({"error": "AI failed to generate a lesson. Try again later."}), 500

    db.collection("lessons").document(user_id).set({"lesson": lesson_text})

    pdf_filename = f"lesson_{user_id}.pdf"
    pdf_path = os.path.join("pdfs", pdf_filename)
    os.makedirs("pdfs", exist_ok=True)
    generate_pdf(lesson_text, pdf_path)

    return jsonify({"lesson": lesson_text, "pdf_url": f"/download_lesson/{user_id}"})

@app.route("/download_lesson/<user_id>", methods=["GET"])
def download_lesson(user_id):
    """Allows users to download the generated PDF."""
    pdf_path = os.path.join("pdfs", f"lesson_{user_id}.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": "Lesson PDF not found"}), 404
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)