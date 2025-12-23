"""File agent for local file operations."""

import os
import json
import glob
from pathlib import Path
from typing import Any, Optional

from .base import BaseAgent
from ..models import AgentType, AgentStatus, Task


class FileAgent(BaseAgent):
    """Agent for local file system operations."""

    agent_type = AgentType.FILE

    def __init__(self, name: Optional[str] = None, base_path: Optional[str] = None):
        super().__init__(name)
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    async def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a file task."""
        action = task.payload.get("action", "read")
        
        handlers = {
            "read": self._action_read,
            "write": self._action_write,
            "append": self._action_append,
            "delete": self._action_delete,
            "list": self._action_list,
            "exists": self._action_exists,
            "mkdir": self._action_mkdir,
            "move": self._action_move,
            "copy": self._action_copy,
            "info": self._action_info,
        }
        
        handler = handlers.get(action)
        if not handler:
            raise ValueError(f"Unknown file action: {action}")
        
        await self.log("info", f"Executing file action: {action}")
        return await handler(task.payload)

    async def _action_read(self, payload: dict) -> dict[str, Any]:
        """Read a file."""
        path = self._resolve_path(payload.get("path", ""))
        encoding = payload.get("encoding", "utf-8")
        
        await self.update_status(AgentStatus.BUSY, summary=f"Reading: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = path.read_text(encoding=encoding)
        
        return {
            "path": str(path),
            "content": content,
            "size": len(content),
        }

    async def _action_write(self, payload: dict) -> dict[str, Any]:
        """Write to a file."""
        path = self._resolve_path(payload.get("path", ""))
        content = payload.get("content", "")
        encoding = payload.get("encoding", "utf-8")
        create_dirs = payload.get("create_dirs", True)
        
        await self.update_status(AgentStatus.BUSY, summary=f"Writing: {path}")
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding=encoding)
        
        return {
            "path": str(path),
            "size": len(content),
            "written": True,
        }

    async def _action_append(self, payload: dict) -> dict[str, Any]:
        """Append to a file."""
        path = self._resolve_path(payload.get("path", ""))
        content = payload.get("content", "")
        encoding = payload.get("encoding", "utf-8")
        
        await self.update_status(AgentStatus.BUSY, summary=f"Appending to: {path}")
        
        with open(path, "a", encoding=encoding) as f:
            f.write(content)
        
        return {
            "path": str(path),
            "appended": len(content),
        }

    async def _action_delete(self, payload: dict) -> dict[str, Any]:
        """Delete a file or directory."""
        path = self._resolve_path(payload.get("path", ""))
        recursive = payload.get("recursive", False)
        
        await self.update_status(AgentStatus.BUSY, summary=f"Deleting: {path}")
        
        if not path.exists():
            return {"path": str(path), "deleted": False, "reason": "not_found"}
        
        if path.is_dir():
            if recursive:
                import shutil
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink()
        
        return {"path": str(path), "deleted": True}

    async def _action_list(self, payload: dict) -> dict[str, Any]:
        """List files in a directory."""
        path = self._resolve_path(payload.get("path", "."))
        pattern = payload.get("pattern", "*")
        recursive = payload.get("recursive", False)
        
        await self.update_status(AgentStatus.BUSY, summary=f"Listing: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        entries = []
        for f in files:
            entries.append({
                "name": f.name,
                "path": str(f),
                "is_dir": f.is_dir(),
                "size": f.stat().st_size if f.is_file() else None,
            })
        
        return {
            "path": str(path),
            "entries": entries,
            "count": len(entries),
        }

    async def _action_exists(self, payload: dict) -> dict[str, Any]:
        """Check if a file or directory exists."""
        path = self._resolve_path(payload.get("path", ""))
        
        return {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else None,
            "is_dir": path.is_dir() if path.exists() else None,
        }

    async def _action_mkdir(self, payload: dict) -> dict[str, Any]:
        """Create a directory."""
        path = self._resolve_path(payload.get("path", ""))
        parents = payload.get("parents", True)
        
        await self.update_status(AgentStatus.BUSY, summary=f"Creating directory: {path}")
        
        path.mkdir(parents=parents, exist_ok=True)
        
        return {"path": str(path), "created": True}

    async def _action_move(self, payload: dict) -> dict[str, Any]:
        """Move/rename a file or directory."""
        src = self._resolve_path(payload.get("source", ""))
        dst = self._resolve_path(payload.get("destination", ""))
        
        await self.update_status(AgentStatus.BUSY, summary=f"Moving: {src} -> {dst}")
        
        import shutil
        shutil.move(str(src), str(dst))
        
        return {"source": str(src), "destination": str(dst), "moved": True}

    async def _action_copy(self, payload: dict) -> dict[str, Any]:
        """Copy a file or directory."""
        src = self._resolve_path(payload.get("source", ""))
        dst = self._resolve_path(payload.get("destination", ""))
        
        await self.update_status(AgentStatus.BUSY, summary=f"Copying: {src} -> {dst}")
        
        import shutil
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        
        return {"source": str(src), "destination": str(dst), "copied": True}

    async def _action_info(self, payload: dict) -> dict[str, Any]:
        """Get file/directory information."""
        path = self._resolve_path(payload.get("path", ""))
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        stat = path.stat()
        
        return {
            "path": str(path),
            "name": path.name,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
        }

    # Convenience methods for programmatic use
    async def read(self, path: str, encoding: str = "utf-8") -> str:
        """Read a file and return its content."""
        result = await self._action_read({"path": path, "encoding": encoding})
        return result["content"]

    async def write(self, path: str, content: str, encoding: str = "utf-8") -> bool:
        """Write content to a file."""
        result = await self._action_write({"path": path, "content": content, "encoding": encoding})
        return result["written"]

    async def list_dir(self, path: str = ".", pattern: str = "*", recursive: bool = False) -> list[dict]:
        """List files in a directory."""
        result = await self._action_list({"path": path, "pattern": pattern, "recursive": recursive})
        return result["entries"]

    async def exists(self, path: str) -> bool:
        """Check if a path exists."""
        result = await self._action_exists({"path": path})
        return result["exists"]

