from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import os
import json
from main import get_server_tools, handle_user_query, execute_tool
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize tools on app startup
async def initialize_tools():
    global all_tools
    all_tools = []
    server_files = [f for f in os.listdir("servers") if f.endswith(".py")]
    
    if not server_files:
        raise Exception("No server files found in the 'servers' directory.")
    
    for server_file in server_files:
        server_tools = await get_server_tools(server_file)
        all_tools.extend(server_tools)

# Run async initialization
asyncio.run(initialize_tools())

@app.route('/')
def index():
    return "MCP Tool Service - Use /tools or /query endpoints"

@app.route('/tools', methods=['GET'])
def list_tools():
    return jsonify([
        {
            "server": tool["server_name"],
            "name": tool["tool_name"],
            "description": tool["description"],
            "parameters": tool["input_schema"].get("properties", {})
        }
        for tool in all_tools
    ])

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get('query')
    model = data.get('model', 'gpt-4o-mini')
    
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    try:
        result = asyncio.run(handle_user_query(query, all_tools))
        
        if result["tool_request"]:
            return jsonify({
                "type": "tool_response",
                "server": result["tool_request"]["server_name"],
                "tool": result["tool_request"]["tool_name"],
                "result": result["result"]
            })
        else:
            return jsonify({
                "type": "ai_response",
                "content": result["raw_llm_response"]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/execute', methods=['POST'])
def direct_execute():
    data = request.get_json()
    try:
        result = asyncio.run(execute_tool(
            data['server_name'],
            data['tool_name'],
            data.get('arguments', {})
        ))
        return jsonify(result)
    except KeyError:
        return jsonify({"error": "Missing required parameters"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG') == '1') 