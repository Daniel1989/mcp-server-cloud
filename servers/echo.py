from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")

@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"

@mcp.tool()
def reverse(text: str) -> str:
    """Reverse the provided text"""
    return text[::-1]

if __name__ == "__main__":
    mcp.run()
