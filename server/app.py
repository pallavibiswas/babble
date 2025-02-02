import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# 🔹 Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# 🔹 Check if API key is missing
if not HUGGINGFACE_API_KEY:
    raise ValueError("Error: Hugging Face API key is missing!")

# 🔹 Initialize Firebase
cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "babble-8e814-firebase-adminsdk-fbsvc-5aa39be796.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# 🔹 Initialize Flask app
app = Flask(__name__)

# 🔹 Default route to check if the API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Mistral AI Speech Therapy API is running!"})

# 🔹 Hugging Face API Setup
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"  # Alternative model
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

@app.route("/generate_lesson", methods=["POST"])
def generate_lesson():
    data = request.json
    user_id = data.get("user_id", "").strip()  # Ensure it's a valid string
    user_input = data.get("speech_issue", "").strip()

    if not user_id:
        return jsonify({"error": "Missing or invalid user_id"}), 400
    if not user_input:
        return jsonify({"error": "Speech issue is required"}), 400

    prompt = f"I have trouble pronouncing {user_input}. Create a personalized lesson plan to improve my pronunciation."

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

    if response.status_code == 200:
        try:
            lesson = response.json()[0]['generated_text']
            db.collection("lessons").document(user_id).set({"lesson": lesson})  # Save lesson to Firebase
            return jsonify({"lesson": lesson})
        except (KeyError, IndexError):
            return jsonify({"error": "Unexpected response format", "details": response.json()}), 500
    else:
        print("Hugging Face API Error:", response.status_code, response.text)
        return jsonify({"error": "Failed to generate lesson", "details": response.text}), 500

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