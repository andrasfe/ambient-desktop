"""LangGraph-based agent workflow orchestration."""

import operator
from typing import Annotated, TypedDict, Literal, Optional, Any
from uuid import UUID
from uuid_extensions import uuid7

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from ..config import settings
from ..services.websocket import ws_manager, EventType


class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    
    # Conversation messages
    messages: Annotated[list[BaseMessage], operator.add]
    
    # Current task information
    task_id: Optional[str]
    task_type: Optional[str]  # "browser", "file", "mcp"
    task_payload: Optional[dict]
    
    # Execution tracking
    subtasks: list[dict]
    current_subtask_index: int
    results: list[dict]
    
    # Agent tracking
    coordinator_id: Optional[str]  # ID of coordinator agent for this request
    
    # Browser state persistence
    selected_tab_index: Optional[int]  # Persist tab selection across subtasks
    
    # Agent metadata
    agent_id: str
    session_id: str
    
    # Control flow
    next_action: Optional[str]
    error: Optional[str]


def create_llm():
    """Create the LLM instance (OpenRouter, Ollama, or any OpenAI-compatible API)."""
    
    # Check if using local LLM (Ollama)
    if settings.is_local_llm:
        return ChatOpenAI(
            model=settings.openrouter_model,
            openai_api_key="ollama",  # Ollama doesn't need a real key
            openai_api_base=settings.openrouter_base_url,
            temperature=0.7,
        )
    
    # Cloud API (OpenRouter)
    return ChatOpenAI(
        model=settings.openrouter_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": "https://ambient-desktop.local",
            "X-Title": "Ambient Desktop Agent",
        },
        temperature=0.7,
    )


def sanitize_for_llm(data: dict) -> dict:
    """Remove sensitive data before sending to cloud LLM.
    
    In privacy mode, we strip actual content and only keep structure.
    """
    if not settings.privacy_mode or settings.is_local_llm:
        return data
    
    sanitized = {}
    for key, value in data.items():
        if key in ("content", "text", "html", "screenshot", "password", "token", "cookie"):
            sanitized[key] = f"[REDACTED - {len(str(value))} chars]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_llm(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_for_llm(v) if isinstance(v, dict) else v for v in value]
        else:
            sanitized[key] = value
    return sanitized


# System prompts
COORDINATOR_SYSTEM_PROMPT = """You are an AI coordinator that orchestrates computer automation tasks.

You can delegate work to specialized agents:

BROWSER agent actions (use these exact action names):
- "navigate" - Go to a URL. params: {"url": "https://..."}
- "click" - Click an element. params: {"selector": "css selector"} or {"text": "button text"}
- "type" - Type into input. params: {"selector": "input selector", "text": "text to type"}
- "extract" - Extract text from page. params: {"selector": "css selector", "target_url": "url pattern to match"} or {} for current page
- "screenshot" - Take screenshot. params: {} or {"selector": "element to capture"}
- "scroll" - Scroll page. params: {"direction": "down|up", "amount": 500} or {"selector": "element to scroll to"}
- "wait" - Wait for element. params: {"selector": "css selector", "timeout": 5000}
- "list_tabs" - List all open browser tabs. params: {}
- "switch_tab" - Switch to a tab. params: {"index": 0} or {"url_contains": "linkedin.com"}

FILE agent actions:
- "read" - Read file. params: {"path": "/path/to/file"}
- "write" - Write file. params: {"path": "/path/to/file", "content": "..."}
- "list" - List directory. params: {"path": "/path/to/dir"}

When a user asks you to do something:
1. Analyze the request
2. Break it into subtasks using the EXACT action names above
3. Specify which agent should handle each subtask

IMPORTANT WORKFLOW RULES:
- To work with a specific website (e.g. LinkedIn), FIRST use switch_tab with url_contains to select that tab
- Then perform extract/click/etc actions
- Example: extract from LinkedIn = switch_tab(url_contains="linkedin") → extract
- For extraction, use {} (empty params) to get full page text if you're not sure about selectors
- Different pages have different content - a feed page is NOT the same as a profile page
- If user asks about "their LinkedIn profile", they likely mean their profile page (linkedin.com/in/...), not the feed

WEBSITE-SPECIFIC TIPS:
- LinkedIn feed (/feed/) shows posts, not profile info
- LinkedIn profile (/in/username) shows name, title, experience, patents, etc.
- To find current user's profile from feed: click profile picture or "Me" menu
- To extract profile info: first navigate to profile page, then extract

LINKEDIN PATENT EXTRACTION:
- Patents are in a collapsible section on LinkedIn profiles
- Step 1: Navigate to the profile URL (linkedin.com/in/username)
- Step 2: Scroll to the Patents section using: {"section": "Patents", "direction": "down"}
- Step 3: Click to expand if needed using text: {"text": "Show all patents"} or {"text": "patents"}
- Step 4: Extract with empty params {} to get all visible text, NOT with complex selectors
- NEVER use complex CSS selectors like [data-section='patents'] - they don't work on LinkedIn

SELECTOR GUIDELINES:
- Prefer text-based clicking: {"text": "Patents"} over {"selector": ".patents-class"}
- For extraction, use {} (empty params) to get full page text
- If you need to scroll to a section, use {"section": "Patents"} not complex selectors
- Simple is better: {"text": "Show all"} works better than {"selector": "button.show-all"}

Respond with a JSON object:
{
    "understanding": "Brief summary of what user wants",
    "subtasks": [
        {
            "agent": "browser|file|mcp",
            "action": "exact action name from list above",
            "params": { ... parameters ... }
        }
    ],
    "response": "What to tell the user"
}

If no action is needed (just chatting), respond with:
{
    "understanding": "User is just chatting",
    "subtasks": [],
    "response": "Your conversational response"
}

IMPORTANT: Use ONLY the action names listed above. Do NOT invent new actions.
NEVER output raw tool-call markup like `<|tool_call_begin|>` / `<|tool_calls_section_begin|>`; always output the JSON plan format described above."""

# Some OpenRouter models sometimes emit raw "tool call" markup (e.g. <|tool_call_begin|>).
# Our workflow expects JSON plans; we defensively parse that markup into subtasks if it appears.


BROWSER_SYSTEM_PROMPT = """You are a browser automation agent. You can:
- navigate: Go to a URL
- click: Click on elements (by selector or text)
- type: Type text into input fields
- extract: Get text/data from the page
- screenshot: Take a screenshot
- scroll: Scroll the page
- wait: Wait for elements or time

Describe what you're doing clearly and report results."""

FILE_SYSTEM_PROMPT = """You are a file system agent. You can:
- read: Read file contents
- write: Write to files
- list: List directory contents
- delete: Delete files
- mkdir: Create directories
- move/copy: Move or copy files

Always confirm actions and report results clearly."""


async def coordinator_node(state: AgentState) -> dict:
    """Coordinator node - analyzes tasks and creates subtask plans."""
    import json
    import re
    
    # Create a visible coordinator agent
    coordinator_id = str(uuid7())
    await ws_manager.broadcast(
        EventType.AGENT_CREATED,
        {
            "id": coordinator_id,
            "type": "coordinator",
            "name": f"coordinator-{coordinator_id[:8]}",
            "status": "busy",
            "summary": "Analyzing request...",
        },
    )
    
    try:
        llm = create_llm()
        
        messages = [SystemMessage(content=COORDINATOR_SYSTEM_PROMPT)] + state["messages"]
        
        await ws_manager.broadcast_log(
            level="info",
            message="Coordinator analyzing request...",
            category="coordinator",
        )
        
        # Update agent status
        await ws_manager.broadcast(
            EventType.AGENT_UPDATE,
            {
                "id": coordinator_id,
                "status": "busy",
                "summary": "Calling LLM...",
            },
        )
        
        response = await llm.ainvoke(messages)
        
        # Parse the response
        try:
            content = response.content
            print(f"[COORDINATOR] Raw LLM response: {content[:500]}...")

            def _parse_tool_call_markup_to_subtasks(raw: str) -> tuple[list[dict], str | None]:
                """
                Parse model-emitted tool-call markup like:
                  <|tool_call_begin|> functions.switch_tab:0 <|tool_call_argument_begin|> {...} <|tool_call_end|>
                into our subtask format.
                """
                if "<|tool_call_begin|>" not in raw and "<|tool_calls_section_begin|>" not in raw:
                    return [], None

                # Keep any user-facing preamble before tool call section.
                prefix = raw.split("<|tool_calls_section_begin|>", 1)[0].strip()

                tool_re = re.compile(
                    r"<\|tool_call_begin\|>\s*functions\.([a-zA-Z0-9_]+)(?::\d+)?\s*"
                    r"<\|tool_call_argument_begin\|>\s*([\s\S]*?)\s*<\|tool_call_end\|>",
                    re.MULTILINE,
                )

                mapping = {
                    "switch_tab": "switch_tab",
                    "list_tabs": "list_tabs",
                    "navigate": "navigate",
                    "click": "click",
                    "type": "type",
                    "extract": "extract",
                    "scroll": "scroll",
                    "wait": "wait",
                    "screenshot": "screenshot",
                }

                subtasks: list[dict] = []
                for m in tool_re.finditer(raw):
                    fn_name = m.group(1)
                    args_raw = m.group(2).strip()
                    try:
                        args = json.loads(args_raw) if args_raw else {}
                    except Exception:
                        # If the args aren't valid JSON, pass them through for debugging.
                        args = {"_raw_args": args_raw}

                    action = mapping.get(fn_name, fn_name)
                    subtasks.append({"agent": "browser", "action": action, "params": args})

                return subtasks, (prefix or None)

            # If the model emitted tool-call markup, convert it into subtasks and proceed.
            tool_subtasks, tool_prefix = _parse_tool_call_markup_to_subtasks(content)
            if tool_subtasks:
                response_text = tool_prefix or "Executing requested browser actions..."
                await ws_manager.broadcast_log(
                    level="info",
                    message=f"Converted tool-call markup into {len(tool_subtasks)} subtask(s)",
                    category="coordinator",
                    details={"source": "tool_call_markup"},
                )
                await ws_manager.broadcast(
                    EventType.AGENT_UPDATE,
                    {
                        "id": coordinator_id,
                        "status": "busy",
                        "summary": f"Delegating {len(tool_subtasks)} task(s)",
                    },
                )
                return {
                    "messages": [AIMessage(content=response_text)],
                    "subtasks": tool_subtasks,
                    "current_subtask_index": 0,
                    "next_action": tool_subtasks[0].get("agent", "end"),
                    "coordinator_id": coordinator_id,
                }
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            print(f"[COORDINATOR] Parsing JSON: {content[:300]}...")
            parsed = json.loads(content.strip())
            subtasks = parsed.get("subtasks", [])
            print(f"[COORDINATOR] Found {len(subtasks)} subtasks")
            response_text = parsed.get("response", content)
            
            await ws_manager.broadcast_log(
                level="info",
                message=f"Created {len(subtasks)} subtask(s)",
                category="coordinator",
                details={"understanding": parsed.get("understanding")},
            )
            
            # Update agent with task info
            await ws_manager.broadcast(
                EventType.AGENT_UPDATE,
                {
                    "id": coordinator_id,
                    "status": "busy" if subtasks else "idle",
                    "summary": f"Delegating {len(subtasks)} task(s)" if subtasks else "Response ready",
                },
            )
            
            # Determine next action
            if subtasks:
                next_action = subtasks[0].get("agent", "end")
            else:
                next_action = "end"
            
            return {
                "messages": [AIMessage(content=response_text)],
                "subtasks": subtasks,
                "current_subtask_index": 0,
                "next_action": next_action,
                "coordinator_id": coordinator_id,
            }
        except json.JSONDecodeError as je:
            print(f"[COORDINATOR] ⚠️ JSON parse failed: {je}")
            print(f"[COORDINATOR] ⚠️ Content was: {content[:300]}...")
            await ws_manager.broadcast(
                EventType.AGENT_UPDATE,
                {
                    "id": coordinator_id,
                    "status": "idle",
                    "summary": "Conversational response",
                },
            )
            # Just a conversational response - no subtasks to execute
            return {
                "messages": [response],
                "subtasks": [],
                "next_action": "end",
                "coordinator_id": coordinator_id,
            }
    except Exception as e:
        await ws_manager.broadcast(
            EventType.AGENT_UPDATE,
            {
                "id": coordinator_id,
                "status": "error",
                "summary": f"Error: {str(e)}",
            },
        )
        raise


async def browser_node(state: AgentState) -> dict:
    """Browser agent node - executes browser automation tasks."""
    from .browser import BrowserAgent
    
    subtasks = state.get("subtasks", [])
    index = state.get("current_subtask_index", 0)
    
    if index >= len(subtasks):
        return {"next_action": "end"}
    
    subtask = subtasks[index]
    
    await ws_manager.broadcast_log(
        level="info",
        message=f"Browser executing: {subtask.get('action')}",
        category="browser",
        details=subtask.get("params"),
    )
    
    # Create browser agent with configured settings
    # Prioritize: CDP (take over existing) > persistent context > fresh browser
    agent = BrowserAgent(
        cdp_url=settings.browser_cdp_url or None,
        user_data_dir=settings.browser_user_data_dir or None,
        headless=settings.browser_headless,
    )
    try:
        await agent.start()
        
        # Restore previously selected tab if available (for multi-step workflows)
        selected_tab_index = state.get("selected_tab_index")
        if selected_tab_index is not None:
            try:
                await agent.use_existing_page(selected_tab_index)
                await ws_manager.broadcast_log(
                    level="debug",
                    message=f"Restored tab selection: index {selected_tab_index}",
                    category="browser",
                )
            except Exception:
                pass  # Tab might not exist anymore
        
        # Map subtask to browser action
        action = subtask.get("action", "")
        params = subtask.get("params", {})
        
        result = {}
        action_lower = action.lower()
        
        # Track tab selection for persistence across subtasks
        new_selected_tab_index = state.get("selected_tab_index")
        
        if "navigate" in action_lower or "go to" in action_lower or "go_to" in action_lower:
            url = params.get("url", params.get("target", ""))
            if url:
                result = await agent.navigate(url)
            else:
                result = {"error": "No URL provided for navigate action"}
        elif "click" in action_lower or "find_element" in action_lower:
            # find_element is treated as click since it's usually followed by interaction
            selector = params.get("selector")
            text = params.get("text", params.get("description", ""))
            result = await agent.click(selector=selector, text=text if not selector else None)
        elif "type" in action_lower or "input" in action_lower or "fill" in action_lower:
            result = await agent.type_text(
                params.get("selector", ""),
                params.get("text", params.get("value", "")),
            )
        elif "extract" in action_lower or "get_text" in action_lower or "scrape" in action_lower:
            # Handle extract, extract_data, get_text, etc.
            # Auto-switch to correct tab if target_url is specified
            target_url = params.get("target_url", params.get("url_contains", ""))
            if target_url:
                pages = await agent.list_pages()
                for page_info in pages:
                    if target_url.lower() in page_info.get("url", "").lower():
                        await agent.use_existing_page(page_info["index"])
                        await ws_manager.broadcast_log(
                            level="info",
                            message=f"Auto-switched to tab: {page_info.get('title', page_info['url'])}",
                            category="browser",
                        )
                        break
            
            selector = params.get("selector")
            all_matches = params.get("all", False)
            result = await agent.extract(
                selector=selector,
                attribute=params.get("attribute"),
                all=all_matches,
            )
            # Include which page was extracted from
            result["extracted_from"] = {
                "url": agent.page.url if agent.page else None,
                "title": await agent.page.title() if agent.page else None,
            }
        elif "screenshot" in action_lower or "capture" in action_lower:
            result = await agent.screenshot()
        elif "scroll" in action_lower:
            # Handle scroll, scroll_to, scroll_to_section, etc.
            # IMPORTANT: params["section"] (e.g. "Patents") is NOT a CSS selector.
            # Use BrowserAgent.scroll(), which supports scrolling to a selector OR a named section.
            result = await agent.scroll(
                direction=params.get("direction", "down"),
                amount=int(params.get("amount", 500) or 500),
                selector=params.get("selector"),
                to_bottom=bool(params.get("to_bottom", False)),
                section=params.get("section"),
            )
        elif "wait" in action_lower:
            selector = params.get("selector")
            timeout = params.get("timeout", 5000)
            if selector:
                await agent.page.wait_for_selector(selector, timeout=timeout)
                result = {"waited_for": selector}
            else:
                import asyncio
                await asyncio.sleep(timeout / 1000)
                result = {"waited_ms": timeout}
        elif "list" in action_lower and ("tab" in action_lower or "page" in action_lower):
            # List open tabs/pages
            pages = await agent.list_pages()
            result = {"tabs": pages, "count": len(pages)}
        elif "switch" in action_lower or "select_tab" in action_lower:
            # Switch to a specific tab by index or URL pattern
            url_contains = params.get("url_contains", params.get("url_pattern", ""))
            page_index = params.get("index", params.get("page_index"))
            
            if url_contains:
                # Find tab by URL pattern
                pages = await agent.list_pages()
                found_index = None
                for page_info in pages:
                    if url_contains.lower() in page_info.get("url", "").lower():
                        found_index = page_info["index"]
                        break
                
                if found_index is not None:
                    result = await agent.use_existing_page(found_index)
                    result["matched_pattern"] = url_contains
                    new_selected_tab_index = found_index
                else:
                    result = {"error": f"No tab found matching: {url_contains}", "available_tabs": pages}
            elif page_index is not None:
                result = await agent.use_existing_page(page_index)
                new_selected_tab_index = page_index
            else:
                result = {"error": "switch_tab requires 'index' or 'url_contains' parameter"}
        elif "current" in action_lower or "url" in action_lower:
            # Get current page info
            result = {
                "url": agent.page.url if agent.page else None,
                "title": await agent.page.title() if agent.page else None,
            }
        elif "evaluate" in action_lower or "javascript" in action_lower or "js" in action_lower:
            # Execute JavaScript
            script = params.get("script", params.get("code", ""))
            if script:
                result = {"result": await agent.page.evaluate(script)}
            else:
                result = {"error": "No script provided"}
        else:
            result = {"error": f"Unknown action: {action}", "hint": "Use: navigate, click, type, extract, screenshot, scroll, wait, list_tabs, switch_tab"}
        
        results = state.get("results", [])
        results.append({"subtask": subtask, "result": result})
        
        # Determine next step
        next_index = index + 1
        if next_index < len(subtasks):
            next_subtask = subtasks[next_index]
            next_action = next_subtask.get("agent", "coordinator")
        else:
            next_action = "summarize"
        
        return_state = {
            "results": results,
            "current_subtask_index": next_index,
            "next_action": next_action,
            "messages": [AIMessage(content=f"Browser completed: {action}")],
        }
        
        # Persist tab selection for multi-step workflows
        if new_selected_tab_index is not None:
            return_state["selected_tab_index"] = new_selected_tab_index
        
        return return_state
    except Exception as e:
        await ws_manager.broadcast_log(
            level="error",
            message=f"Browser error: {str(e)}",
            category="browser",
        )
        return {
            "error": str(e),
            "next_action": "summarize",
            "messages": [AIMessage(content=f"Browser error: {str(e)}")],
        }
    finally:
        await agent.stop()


async def file_node(state: AgentState) -> dict:
    """File agent node - executes file system operations."""
    from .file import FileAgent
    
    subtasks = state.get("subtasks", [])
    index = state.get("current_subtask_index", 0)
    
    if index >= len(subtasks):
        return {"next_action": "end"}
    
    subtask = subtasks[index]
    
    await ws_manager.broadcast_log(
        level="info",
        message=f"File agent executing: {subtask.get('action')}",
        category="file",
        details=subtask.get("params"),
    )
    
    agent = FileAgent()
    try:
        await agent.start()
        
        action = subtask.get("action", "")
        params = subtask.get("params", {})
        
        result = {}
        if "read" in action.lower():
            content = await agent.read(params.get("path", ""))
            result = {"content": content[:1000]}  # Truncate for safety
        elif "write" in action.lower():
            success = await agent.write(
                params.get("path", ""),
                params.get("content", ""),
            )
            result = {"written": success}
        elif "list" in action.lower():
            entries = await agent.list_dir(
                params.get("path", "."),
                params.get("pattern", "*"),
            )
            result = {"entries": entries[:50]}  # Limit results
        elif "exists" in action.lower():
            exists = await agent.exists(params.get("path", ""))
            result = {"exists": exists}
        else:
            result = {"error": f"Unknown action: {action}"}
        
        results = state.get("results", [])
        results.append({"subtask": subtask, "result": result})
        
        next_index = index + 1
        if next_index < len(subtasks):
            next_subtask = subtasks[next_index]
            next_action = next_subtask.get("agent", "coordinator")
        else:
            next_action = "summarize"
        
        return {
            "results": results,
            "current_subtask_index": next_index,
            "next_action": next_action,
            "messages": [AIMessage(content=f"File operation completed: {action}")],
        }
    except Exception as e:
        await ws_manager.broadcast_log(
            level="error",
            message=f"File error: {str(e)}",
            category="file",
        )
        return {
            "error": str(e),
            "next_action": "summarize",
            "messages": [AIMessage(content=f"File error: {str(e)}")],
        }
    finally:
        await agent.stop()


async def summarize_node(state: AgentState) -> dict:
    """Summarize the results of all subtasks."""
    llm = create_llm()
    
    results = state.get("results", [])
    error = state.get("error")
    coordinator_id = state.get("coordinator_id")
    
    if not results and not error:
        # Mark coordinator as complete
        if coordinator_id:
            await ws_manager.broadcast(
                EventType.AGENT_UPDATE,
                {
                    "id": coordinator_id,
                    "status": "idle",
                    "summary": "No tasks to execute",
                },
            )
        return {"next_action": "end"}
    
    # Update coordinator status
    if coordinator_id:
        await ws_manager.broadcast(
            EventType.AGENT_UPDATE,
            {
                "id": coordinator_id,
                "status": "busy",
                "summary": "Summarizing results...",
            },
        )
    
    # Create a summary prompt
    summary_prompt = f"""Summarize what was accomplished:

Results:
{results}

{"Error: " + error if error else ""}

Provide a brief, user-friendly summary of what was done."""

    messages = [
        SystemMessage(content="You are a helpful assistant summarizing task results."),
        HumanMessage(content=summary_prompt),
    ]
    
    response = await llm.ainvoke(messages)
    
    await ws_manager.broadcast_log(
        level="info",
        message="Task execution completed",
        category="coordinator",
    )
    
    # Mark coordinator as complete
    if coordinator_id:
        await ws_manager.broadcast(
            EventType.AGENT_UPDATE,
            {
                "id": coordinator_id,
                "status": "idle",
                "summary": "Task completed",
            },
        )
    
    return {
        "messages": [response],
        "next_action": "end",
    }


def route_next_action(state: AgentState) -> str:
    """Route to the next node based on state."""
    next_action = state.get("next_action", "end")
    
    if next_action == "browser":
        return "browser"
    elif next_action == "file":
        return "file"
    elif next_action == "mcp":
        return "mcp"  # TODO: implement MCP node
    elif next_action == "summarize":
        return "summarize"
    elif next_action == "coordinator":
        return "coordinator"
    else:
        return "end"


def create_agent_graph() -> StateGraph:
    """Create the LangGraph agent workflow."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("browser", browser_node)
    workflow.add_node("file", file_node)
    workflow.add_node("summarize", summarize_node)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    # Add conditional edges from coordinator
    workflow.add_conditional_edges(
        "coordinator",
        route_next_action,
        {
            "browser": "browser",
            "file": "file",
            "summarize": "summarize",
            "coordinator": "coordinator",
            "end": END,
        },
    )
    
    # Add conditional edges from browser
    workflow.add_conditional_edges(
        "browser",
        route_next_action,
        {
            "browser": "browser",
            "file": "file",
            "summarize": "summarize",
            "coordinator": "coordinator",
            "end": END,
        },
    )
    
    # Add conditional edges from file
    workflow.add_conditional_edges(
        "file",
        route_next_action,
        {
            "browser": "browser",
            "file": "file",
            "summarize": "summarize",
            "coordinator": "coordinator",
            "end": END,
        },
    )
    
    # Summarize always ends
    workflow.add_edge("summarize", END)
    
    return workflow


class AgentOrchestrator:
    """High-level orchestrator using LangGraph."""
    
    def __init__(self):
        self.workflow = create_agent_graph()
        self.memory = MemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.memory)
        self.sessions: dict[str, str] = {}  # session_id -> thread_id
    
    def _get_thread_id(self, session_id: str) -> str:
        """Get or create a thread ID for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = str(uuid7())
            print(f"[SESSION] New session: {session_id} -> thread {self.sessions[session_id]}")
        else:
            print(f"[SESSION] Existing session: {session_id} -> thread {self.sessions[session_id]}")
        return self.sessions[session_id]
    
    async def process_message(
        self,
        message: str,
        session_id: str = "default",
    ) -> str:
        """Process a user message through the agent graph."""
        thread_id = self._get_thread_id(session_id)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Always pass full state - LangGraph MemorySaver will handle merging
        input_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": str(uuid7()),
            "session_id": session_id,
            "next_action": None,
            "error": None,
            "coordinator_id": None,
            "selected_tab_index": None,
        }
        
        try:
            print(f"[ORCHESTRATOR] Running graph.ainvoke for session {session_id}...")
            # Run the graph
            final_state = await self.graph.ainvoke(input_state, config)
            print(f"[ORCHESTRATOR] Graph completed!")
            
            # Extract the response
            messages = final_state.get("messages", [])
            print(f"[ORCHESTRATOR] Got {len(messages)} messages in final state")
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    print(f"[ORCHESTRATOR] Returning response: {last_message.content[:100]}...")
                    return last_message.content
            
            print(f"[ORCHESTRATOR] No messages, returning 'Task completed.'")
            return "Task completed."
        except Exception as e:
            print(f"[ORCHESTRATOR] ❌ Exception: {e}")
            await ws_manager.broadcast_log(
                level="error",
                message=f"Orchestrator error: {str(e)}",
                category="coordinator",
            )
            raise
    
    async def stream_message(
        self,
        message: str,
        session_id: str = "default",
    ):
        """Stream responses from the agent graph."""
        thread_id = self._get_thread_id(session_id)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Always pass full state
        input_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "task_id": None,
            "task_type": None,
            "task_payload": None,
            "subtasks": [],
            "current_subtask_index": 0,
            "results": [],
            "agent_id": str(uuid7()),
            "session_id": session_id,
            "next_action": None,
            "error": None,
            "coordinator_id": None,
            "selected_tab_index": None,
        }
        
        try:
            async for event in self.graph.astream_events(input_state, config, version="v2"):
                kind = event.get("event")
                
                if kind == "on_chat_model_stream":
                    content = event.get("data", {}).get("chunk", {})
                    if hasattr(content, "content") and content.content:
                        yield content.content
        except Exception as e:
            await ws_manager.broadcast_log(
                level="error",
                message=f"Stream error: {str(e)}",
                category="coordinator",
            )
            raise


# Global orchestrator instance
orchestrator = AgentOrchestrator()

