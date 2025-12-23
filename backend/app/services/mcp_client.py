"""MCP (Model Context Protocol) client for external tool integration."""

import asyncio
import json
import subprocess
from typing import Any, Optional
from dataclasses import dataclass, field

from ..config import settings
from .websocket import ws_manager


@dataclass
class MCPTool:
    """A tool available from an MCP server."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    server: str = ""


@dataclass
class MCPServer:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class MCPClient:
    """Client for connecting to MCP servers via stdio."""

    def __init__(self):
        self._servers: dict[str, MCPServer] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._tools: dict[str, MCPTool] = {}
        self._request_id = 0
        self._pending_responses: dict[int, asyncio.Future] = {}

    async def start(self) -> None:
        """Initialize MCP client and connect to configured servers."""
        server_configs = settings.mcp_server_configs
        
        for config in server_configs:
            server = MCPServer(
                name=config.get("name", "unknown"),
                command=config.get("command", ""),
                args=config.get("args", []),
                env=config.get("env", {}),
            )
            await self.add_server(server)

    async def stop(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._processes.keys()):
            await self.remove_server(name)

    async def add_server(self, server: MCPServer) -> None:
        """Add and connect to an MCP server."""
        if server.name in self._servers:
            return
        
        self._servers[server.name] = server
        
        try:
            # Start the server process
            process = subprocess.Popen(
                [server.command] + server.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**dict(__import__("os").environ), **server.env},
            )
            self._processes[server.name] = process
            
            # Initialize the connection
            await self._send_request(server.name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ambient-desktop",
                    "version": "1.0.0",
                },
            })
            
            # Discover available tools
            await self._discover_tools(server.name)
            
            await ws_manager.broadcast_log(
                level="info",
                message=f"Connected to MCP server: {server.name}",
                category="mcp",
            )
        except Exception as e:
            await ws_manager.broadcast_log(
                level="error",
                message=f"Failed to connect to MCP server {server.name}: {e}",
                category="mcp",
            )

    async def remove_server(self, name: str) -> None:
        """Disconnect from an MCP server."""
        if name not in self._servers:
            return
        
        # Kill the process
        if name in self._processes:
            process = self._processes.pop(name)
            process.terminate()
            process.wait(timeout=5)
        
        # Remove tools from this server
        self._tools = {k: v for k, v in self._tools.items() if v.server != name}
        
        del self._servers[name]

    async def _send_request(
        self,
        server_name: str,
        method: str,
        params: dict,
    ) -> Any:
        """Send a JSON-RPC request to an MCP server."""
        if server_name not in self._processes:
            raise ValueError(f"Server not connected: {server_name}")
        
        process = self._processes[server_name]
        self._request_id += 1
        request_id = self._request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        process.stdin.write(request_str.encode())
        process.stdin.flush()
        
        # Read response (simplified - real implementation would be async)
        response_line = await asyncio.get_event_loop().run_in_executor(
            None, process.stdout.readline
        )
        
        if not response_line:
            raise RuntimeError("No response from MCP server")
        
        response = json.loads(response_line.decode())
        
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        
        return response.get("result")

    async def _discover_tools(self, server_name: str) -> None:
        """Discover available tools from an MCP server."""
        try:
            result = await self._send_request(server_name, "tools/list", {})
            
            for tool_def in result.get("tools", []):
                tool = MCPTool(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    server=server_name,
                )
                self._tools[f"{server_name}:{tool.name}"] = tool
        except Exception as e:
            await ws_manager.broadcast_log(
                level="warn",
                message=f"Failed to discover tools from {server_name}: {e}",
                category="mcp",
            )

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> Any:
        """Call a tool on an MCP server."""
        full_name = f"{server_name}:{tool_name}"
        if full_name not in self._tools:
            raise ValueError(f"Tool not found: {full_name}")
        
        await ws_manager.broadcast_log(
            level="info",
            message=f"Calling MCP tool: {full_name}",
            category="mcp",
            details={"arguments": arguments},
        )
        
        result = await self._send_request(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        
        return result

    def get_tools(self) -> list[MCPTool]:
        """Get all available tools."""
        return list(self._tools.values())

    def get_tool(self, server_name: str, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool."""
        return self._tools.get(f"{server_name}:{tool_name}")

    @property
    def connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self._processes.keys())


# Global instance
mcp_client = MCPClient()

