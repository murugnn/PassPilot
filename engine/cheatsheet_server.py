# cheatsheet_server.py

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
# Import the new analyzer class
from cheatsheet_analyser import CheatSheetAnalyzer

# --- Initial Setup ---

# Load environment variables (.env file) for the API key
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Google AI client library. This must be done before initializing the analyzer.
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    logging.critical(f"FATAL: Could not configure Google AI. Server cannot start. Error: {e}")
    # In a real application, you might exit or handle this more gracefully.
    exit()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Initialize Cheat Sheet Analyzer on Server Startup ---
# This global instance is created only once, making it efficient.
logging.info("Initializing CheatSheetAnalyzer...")
cheatsheet_analyzer = CheatSheetAnalyzer()
if not cheatsheet_analyzer.model:
    logging.critical("CheatSheetAnalyzer failed to initialize its model. The API will not be functional.")


# --- API Endpoints ---

@app.route('/generate', methods=['POST'])
def generate():
    """
    Endpoint to generate a cheat sheet.
    Expects JSON: {"file_path": "/path/to/your.json", "module": "Module Name"}
    """
    if not cheatsheet_analyzer or not cheatsheet_analyzer.model:
        return jsonify({"error": "Service is unavailable. The AI model is not initialized."}), 503

    data = request.get_json()
    if not data or 'file_path' not in data or 'module' not in data:
        return jsonify({"error": "Request body must contain 'file_path' and 'module'."}), 400

    file_path = data['file_path']
    module = data['module']

    result = cheatsheet_analyzer.generate_for_module(file_path, module)

    # Check if the result contains an error key to return the appropriate status code
    if 'error' in result:
        # Determine status code based on error type
        if "not found" in result["error"]:
            return jsonify(result), 404 # Not Found
        else:
            return jsonify(result), 400 # Bad Request

    return jsonify(result)

@app.route('/modules', methods=['POST'])
def get_modules():
    """
    Endpoint to get the list of available modules from a JSON file.
    Expects JSON: {"file_path": "/path/to/your.json"}
    """
    if not cheatsheet_analyzer:
        return jsonify({"error": "Service is unavailable."}), 503

    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({"error": "Request body must contain 'file_path'."}), 400

    result = cheatsheet_analyzer.get_available_modules(data['file_path'])

    if 'error' in result:
        return jsonify(result), 404 if "not found" in result["error"] else 400

    return jsonify(result)


# --- Main Execution ---

if __name__ == '__main__':
    # Using port 5002 to avoid conflict if the RAG server is also running
    app.run(host='0.0.0.0', port=5002, debug=True)