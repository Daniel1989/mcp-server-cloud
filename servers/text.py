from mcp.server.fastmcp import FastMCP
import re

# Initialize the FastMCP server
mcp = FastMCP("text")

@mcp.tool()
def count_words(text: str) -> int:
    """Count the number of words in the text"""
    words = text.split()
    return len(words)

@mcp.tool()
def find_pattern(text: str, pattern: str) -> list:
    """Find all occurrences of a regex pattern in the text"""
    try:
        matches = re.findall(pattern, text)
        return matches
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {str(e)}")

@mcp.tool()
def to_upper(text: str) -> str:
    """Convert text to uppercase"""
    return text.upper()

@mcp.tool()
def to_lower(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()

@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse the characters in the text"""
    return text[::-1]

if __name__ == "__main__":
    # This will automatically handle the MCP protocol
    mcp.run() 