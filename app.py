import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from the custom v.env file
load_dotenv("v.env")

# Retrieve Gemini API Key
api_key = os.getenv("Gemini_API_Key")
if not api_key:
    # Fallback check
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Warning: Gemini API Key not found. Please set 'Gemini_API_Key' in v.env")

# Configure GenAI SDK
if api_key:
    genai.configure(api_key=api_key)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not api_key:
        return jsonify({'error': 'Gemini API Key is not configured on the server. Please check your v.env file.'}), 500

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    history = data.get('history', [])
    
    try:
        # Initialize the model (using gemini-2.5-flash as the standard efficient model)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Format history for the Gemini API
        # Gemini expects: [{'role': 'user'|'model', 'parts': [text]}]
        gemini_history = []
        for msg in history:
            role = 'user' if msg.get('role') == 'user' else 'model'
            content = msg.get('content', '')
            gemini_history.append({
                'role': role,
                'parts': [content]
            })
            
        # Start a chat session with the conversation history
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_message)
        
        return jsonify({
            'reply': response.text
        })
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'error': f"Gemini API Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
