import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from the custom v.env file
load_dotenv("v.env")

# Retrieve Gemini API Key
api_key = os.getenv("Gemini_API_Key") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Warning: Gemini API Key not found. Please set 'Gemini_API_Key' in v.env")

app = Flask(__name__)

# Initialize the Gemini GenAI Client
client = None
if api_key:
    client = genai.Client(api_key=api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({'error': 'Gemini API Client is not initialized. Please verify your Gemini_API_Key in v.env'}), 500

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    history = data.get('history', [])
    
    try:
        # Convert client history to the format expected by the google-genai SDK
        genai_history = []
        for msg in history:
            role = 'user' if msg.get('role') == 'user' else 'model'
            content = msg.get('content', '')
            genai_history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)]
                )
            )
            
        # Start chat session using the new SDK
        chat_session = client.chats.create(
            model="gemini-2.5-flash",
            history=genai_history
        )
        
        response = chat_session.send_message(user_message)
        
        return jsonify({
            'reply': response.text
        })
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'error': f"Gemini API Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
