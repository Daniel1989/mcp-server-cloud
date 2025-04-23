import os
import json
import re
from typing import Dict, List, Any, Optional
import argparse
import requests

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("BASE_URL")
)

model_name = "gemini-2.0-flash"


# Storage for all available tools
all_tools = []

# System prompt template for the LLM that selects the appropriate MCP tool
LLM_SYSTEM_PROMPT = """
You are an AI assistant that helps users interact with MCP tools. Based on the user's input, you will determine which tool to use and with what parameters.

Available tools:
{tools_description}

When you've determined which tool to use, respond using EXACTLY the following format:

<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{
  "param1": "value1",
  "param2": "value2"
}}
</arguments>
</use_mcp_tool>

If you cannot determine a suitable tool, respond with a helpful message explaining why and offer alternatives.
"""

# Optional: create a sampling callback
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model=model_name,
        stopReason="endTurn",
    )

async def get_server_tools(server_file: str, is_js: bool = False) -> List[Dict[str, Any]]:
    """Get all tools from a specific server file"""
    if is_js:
        server_params = StdioServerParameters(
            command="npx",  # Executable
            args=['@playwright/mcp@latest'],  # Server script path
            env=None,  # Optional environment variables
        ) 
        server_name = 'default-js-server'
    else:
        server_path = os.path.join("servers", server_file)
        server_name = os.path.splitext(server_file)[0]  # Remove extension
        
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",  # Executable
            args=[server_path],  # Server script path
            env=None,  # Optional environment variables
        )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(
                read, write, sampling_callback=handle_sampling_message
            ) as session:
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                # Add server name to each tool for identification
                server_tools = []
                for tool in tools.tools:
                    tool_info = {
                        "server_name": server_name,
                        "tool_name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    server_tools.append(tool_info)
                
                return server_tools
    except Exception as e:
        print(f"Error connecting to server {server_name}: {e}")
        return []

async def execute_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific tool with the given arguments"""
    server_path = os.path.join("servers", f"{server_name}.py")
    
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",  # Executable
        args=[server_path],  # Server script path
        env=None,  # Optional environment variables
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(
                read, write, sampling_callback=handle_sampling_message
            ) as session:
                # Initialize the connection
                await session.initialize()
                
                # Execute the tool
                print(f"Calling tool {tool_name} with arguments {arguments}")
                result = await session.call_tool(tool_name, arguments)
                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": result.content[0].text,
                    "success": True
                }
    except Exception as e:
        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "error": str(e),
            "success": False
        }

def parse_tool_request(input_text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool request in XML format"""
    pattern = r'<use_mcp_tool>\s*<server_name>(.*?)</server_name>\s*<tool_name>(.*?)</tool_name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</use_mcp_tool>'
    match = re.search(pattern, input_text, re.DOTALL)
    
    if not match:
        return None
    
    server_name = match.group(1).strip()
    tool_name = match.group(2).strip()
    arguments_json = match.group(3).strip()
    
    try:
        arguments = json.loads(arguments_json)
        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing arguments JSON: {e}")
        return None

def format_tools_for_llm(tools: List[Dict[str, Any]]) -> str:
    """Format the tools description for the LLM prompt"""
    formatted_tools = []
    
    for tool in tools:
        server_name = tool["server_name"]
        tool_name = tool["tool_name"]
        description = tool["description"]
        input_schema = tool.get("input_schema", {})
        
        # Format parameters
        params = []
        if isinstance(input_schema, dict) and "properties" in input_schema:
            properties = input_schema["properties"]
            required = input_schema.get("required", [])
            
            for param_name, param_info in properties.items():
                is_required = param_name in required
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                
                params.append(f"- {param_name}: ({param_type}{', required' if is_required else ''}) {param_desc}")
        
        # Create tool description
        tool_desc = f"""
Server: {server_name}
Tool: {tool_name}
Description: {description}
Parameters:
{chr(10).join(params) if params else '  None'}
"""
        formatted_tools.append(tool_desc)
    
    return "\n".join(formatted_tools)

def ask_llm_for_tool_selection(query: str, tools: List[Dict[str, Any]], model: str = model_name) -> str:
    """Ask the LLM to select an appropriate tool based on the user query"""
    # Format tools description for the prompt
    tools_description = format_tools_for_llm(tools)
    system_prompt = LLM_SYSTEM_PROMPT.format(tools_description=tools_description)
    
    
    # Make the API request
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
    )

    message_content = response.choices[0].message.content
    print(message_content)
    return message_content

async def handle_user_query(query: str, tools: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    """Process a user query, ask LLM to select a tool, and execute it"""
    # Ask LLM for tool selection
    merged_tools = all_tools + tools
    llm_response = ask_llm_for_tool_selection(query, merged_tools)
    
    # Check if the response contains a tool request
    tool_request = parse_tool_request(llm_response)
    
    if tool_request:
        print(f"Selected tool: {tool_request['server_name']}.{tool_request['tool_name']}")
        result = await execute_tool(
            tool_request["server_name"],
            tool_request["tool_name"],
            tool_request["arguments"]
        )
        return {
            "tool_request": tool_request,
            "result": result["result"],
            "raw_llm_response": llm_response
        }
    else:
        # LLM didn't select a tool or responded in free form
        return {
            "tool_request": None,
            "result": None,
            "raw_llm_response": llm_response
        }

async def run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI CLI with MCP tool selection")
    parser.add_argument("query", nargs="?", help="User query to process")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--list-tools", action="store_true", help="List all available tools")
    args = parser.parse_args()
    
    # Create servers directory if it doesn't exist
    os.makedirs("servers", exist_ok=True)
    
    # Check if there are any server files
    server_files = [f for f in os.listdir("servers") if f.endswith(".py")]
    
    if not server_files:
        print("No server files found in the 'servers' directory.")
        raise Exception("No server files found in the 'servers' directory.")
    
    # Get tools from all servers
    global all_tools
    for server_file in server_files:
        server_tools = await get_server_tools(server_file)
        all_tools.extend(server_tools)

    server_tools = await get_server_tools('', False)
    all_tools.extend(server_tools)
    
    # List tools if requested
    if args.list_tools:
        print(f"Found {len(all_tools)} tools across {len(server_files)} servers:")
        for tool in all_tools:
            print(f"  - {tool['server_name']}.{tool['tool_name']}: {tool['description']}")
        return
    
    # Handle user query if provided
    if args.query:
        result = await handle_user_query(args.query)
        
        if result["tool_request"]:
            # Tool was selected and executed
            print(json.dumps(result["result"], indent=2))
        else:
            # LLM response without tool selection
            print("AI Response:")
            print(result["raw_llm_response"])
    else:
        # Interactive mode
        print(f"Found {len(all_tools)} tools across {len(server_files)} servers.")
        print("Enter your query (or 'exit' to quit):")
        
        while True:
            try:
                query = input("> ")
                if query.lower() in ("exit", "quit", "q"):
                    break
                
                result = await handle_user_query(query)
                
                if result["tool_request"]:
                    # Tool was selected and executed
                    print(json.dumps(result["result"], indent=2))
                else:
                    # LLM response without tool selection
                    print("AI Response:")
                    print(result["raw_llm_response"])
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
    