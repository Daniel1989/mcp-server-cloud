# AI Service Platform

A cloud-ready service platform for AI-powered tool execution with Model Context Protocol (MCP) integration.

## Key Features

- **Cloud Native Architecture**
  - REST API endpoints for all operations
  - Stateless design with persistent tool configuration
  - Horizontal scaling support

- **Unified Tool Gateway**
  - Automatic discovery of MCP tools in `servers/` directory

## Cloud Deployment

### Prerequisites
- Python 3.10+

## API Usage

### Endpoints

**List Available Tools**
```bash
GET /tools
```

**Execute Natural Language Query**
```bash
POST /query
{
  "query": "5+5",
}
```


## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Daniel1989/mcp-server-cloud.git
cd mcp-server-cloud
```

2. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Start development server:
```bash
FLASK_DEBUG=1 python flask.py
```

## Resources
1. python-sdk. https://github.com/modelcontextprotocol/python-sdk
2. cline's prompt -- how to ask ai to select mcp server. https://github.com/cline/cline/blob/main/src/core/prompts/system.ts