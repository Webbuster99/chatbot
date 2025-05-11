# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import os
import json

# Import your LangGraph chatbot logic
from main import graph

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])
        
        # Create state for the graph
        state = {
            "messages": messages,
            "message_type": None
        }

        # Invoke the graph
        result = graph.invoke(state)

        # Process the messages to ensure they're JSON serializable
        def serialize_message(msg):
            # If already a dict, return as is and ensure it has content
            if isinstance(msg, dict):
                # For analysis responses with code, expected_output, and explanation
                if 'code' in msg and 'content' not in msg:
                    # Create a content field that summarizes the analysis
                    code_summary = "Analysis code generated."
                    msg['content'] = code_summary
                return msg
                
            # If LangChain message object, convert to dict
            if hasattr(msg, 'to_dict'):
                msg_dict = msg.to_dict()
                # Ensure content exists
                if 'content' not in msg_dict:
                    msg_dict['content'] = "Response generated."
                return msg_dict
                
            # Fallback: include all attributes that don't start with '_'
            if hasattr(msg, '__dict__'):
                msg_dict = {k: v for k, v in msg.__dict__.items() if not k.startswith('_')}
                # Ensure content exists
                if 'content' not in msg_dict:
                    msg_dict['content'] = "Response generated."
                return msg_dict
                
            # If it's not an object, just return as string content
            return {
                "role": getattr(msg, "role", "assistant"),
                "content": getattr(msg, "content", str(msg))
            }

        # Ensure all messages are properly serialized
        if "messages" in result:
            processed_messages = []
            for m in result["messages"]:
                processed_msg = serialize_message(m)
                
                # Ensure each message has the required 'role' field
                if "role" not in processed_msg:
                    processed_msg["role"] = "assistant"
                    
                # Ensure each message has the required 'content' field
                if "content" not in processed_msg and not any(k in processed_msg for k in ['code', 'expected_output', 'explanation']):
                    processed_msg["content"] = "Response generated."
                    
                processed_messages.append(processed_msg)
                
            result["messages"] = processed_messages
            
        # Print the result for debugging
        print("API Response:", json.dumps(result, indent=2, default=str))
            
        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)