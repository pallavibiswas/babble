from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    try:
        # Start the real_time_analysis.py script
        subprocess.Popen(["python", "real_time_analysis.py"])  # Run the script in the background
        return jsonify({"status": "started"})
    except Exception as e:
        print(f"Error starting analysis: {e}")
        return jsonify({"status": "failed", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)