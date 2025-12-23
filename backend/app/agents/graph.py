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
from .browser import BrowserAgent


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

CRITICAL RULE: Check the CURRENT BROWSER STATE section below before planning any actions.
If the page you need is ALREADY OPEN, just extract - DO NOT navigate or click unnecessarily!

BROWSER agent actions:
- "extract" - Extract text from current page. params: {} (empty = get all visible text)
- "switch_tab" - Switch to a tab. params: {"url_contains": "linkedin.com"} or {"index": 0}
- "scroll" - Scroll page. params: {"direction": "down"} or {"section": "Patents"}
- "click" - Click an element. params: {"text": "button text"} (prefer text over selectors)
- "navigate" - Go to a URL. params: {"url": "https://..."}
- "type" - Type into input. params: {"selector": "input", "text": "text"}
- "edit_fields" - Generic edit in a form: find item by text and set/swap fields by semantic-ish field name matching.
  params: {"match_text": "...", "swap": ["Field A", "Field B"]} or {"match_text": "...", "fields": {"Field": "New value"}}
- "screenshot" - Take screenshot. params: {}
- "list_tabs" - List open tabs. params: {}

FILE agent actions:
- "read" - Read file. params: {"path": "/path/to/file"}
- "write" - Write file. params: {"path": "/path/to/file", "content": "..."}
- "list" - List directory. params: {"path": "/path/to/dir"}

DECISION FLOW:
1. READ the "CURRENT BROWSER STATE" section - especially "Active URL" and "All open tabs"
2. CHECK if the content user wants is on a DIFFERENT tab than the Active tab
3. If content is on a different tab → MUST use "switch_tab" FIRST
4. Then use "extract" to get the content

CRITICAL TAB SWITCHING:
- Look at "Active URL" - this is where actions will happen
- Look at "All open tabs" - user might want content from a different tab
- If user asks for "linkedin" content but Active URL is NOT linkedin → switch_tab first!
- Example: Active=openrouter.ai, user wants LinkedIn → [switch_tab, extract]

EXAMPLES:
User: "extract from linkedin" (Active URL: openrouter.ai, Tab 3 is linkedin)
→ [{"agent": "browser", "action": "switch_tab", "params": {"url_contains": "linkedin"}},
   {"agent": "browser", "action": "extract", "params": {}}]

User: "get text from current page" (any active URL)
→ [{"agent": "browser", "action": "extract", "params": {}}]

Respond with a JSON object:
{
    "understanding": "Brief summary of what user wants",
    "subtasks": [
        {
            "agent": "browser|file",
            "action": "action name",
            "params": { ... }
        }
    ],
    "response": "What to tell the user"
}

EXAMPLES:

User: "extract patents from linkedin" (and LinkedIn profile is already the active tab)
→ {"subtasks": [{"agent": "browser", "action": "extract", "params": {}}]}

User: "get info from linkedin" (but active tab is CNN)
→ {"subtasks": [{"agent": "browser", "action": "switch_tab", "params": {"url_contains": "linkedin"}}, {"agent": "browser", "action": "extract", "params": {}}]}

IMPORTANT: 
- Use ONLY the action names listed above
- Check current browser state BEFORE planning navigation
- Prefer minimal actions - if content is already visible, just extract!
- It's OK (and expected) for subagents to perform MULTIPLE steps in sequence (one step at a time) to complete a request.
- NEVER output raw tool-call markup"""

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
        # ---- Fast-path: deterministic planning for generic "edit/swap fields" requests ----
        # We do this without an LLM to avoid the common failure mode of "extracting" instead of editing.
        latest_user_text = ""
        try:
            # messages is a list[BaseMessage]; HumanMessage has .content
            if state.get("messages"):
                latest_user_text = state["messages"][-1].content or ""
        except Exception:
            latest_user_text = ""

        def _extract_match_text(text: str) -> str | None:
            # Prefer quoted strings first
            m = re.search(r'"([^"]{3,100})"', text)
            if m:
                return m.group(1).strip()
            m = re.search(r"'([^']{3,100})'", text)
            if m:
                return m.group(1).strip()
            # Otherwise try to find an identifier-like token (letters+digits with dashes/underscores)
            m = re.search(r"\b[A-Za-z]{1,6}[-_ ]?\d{3,}[-_A-Za-z0-9]{0,20}\b", text)
            if m:
                return m.group(0).strip()
            return None

        def _extract_swap_fields(text: str) -> tuple[str, str] | None:
            # swap X with Y
            m = re.search(r"\bswap\s+(?:the\s+)?(.+?)\s+with\s+(?:the\s+)?(.+?)(?:\s+values|\s+fields|\s*$)", text, re.IGNORECASE)
            if m:
                a = m.group(1).strip(" .,:;\"'()[]{}")
                b = m.group(2).strip(" .,:;\"'()[]{}")
                if a and b:
                    return a, b
            # swap X and Y
            m = re.search(r"\bswap\s+(?:the\s+)?(.+?)\s+(?:and|<->)\s+(?:the\s+)?(.+?)(?:\s+values|\s+fields|\s*$)", text, re.IGNORECASE)
            if m:
                a = m.group(1).strip(" .,:;\"'()[]{}")
                b = m.group(2).strip(" .,:;\"'()[]{}")
                if a and b:
                    return a, b
            return None

        wants_edit = bool(re.search(r"\b(edit|pencil|change|update|modify|fix)\b", latest_user_text, re.IGNORECASE))
        wants_swap = bool(re.search(r"\bswap\b", latest_user_text, re.IGNORECASE))
        swap_pair = _extract_swap_fields(latest_user_text) if wants_swap else None
        match_text = _extract_match_text(latest_user_text) if (wants_edit or wants_swap) else None

        if (wants_swap and swap_pair and match_text) or (wants_edit and match_text and swap_pair):
            subtasks: list[dict] = []
            # If the user referenced a specific tab/site, switch to it first.
            if re.search(r"\blinkedin\b", latest_user_text, re.IGNORECASE):
                subtasks.append({"agent": "browser", "action": "switch_tab", "params": {"url_contains": "linkedin"}})
            else:
                # Generic: if user says "active tab" or "current tab", don't switch.
                pass

            a, b = swap_pair
            subtasks.append(
                {
                    "agent": "browser",
                    "action": "edit_fields",
                    "params": {
                        "match_text": match_text,
                        "swap": [a, b],
                    },
                }
            )

            await ws_manager.broadcast_log(
                level="info",
                message="Using deterministic edit/swap plan (no LLM) for reliability.",
                category="coordinator",
                details={"match_text": match_text, "swap": [a, b]},
            )
            await ws_manager.broadcast(
                EventType.AGENT_UPDATE,
                {
                    "id": coordinator_id,
                    "status": "busy",
                    "summary": f"Delegating {len(subtasks)} task(s)",
                },
            )
            return {
                "messages": [AIMessage(content="Got it — I'll open the editor for that item and swap the two fields.")],
                "subtasks": subtasks,
                "current_subtask_index": 0,
                "next_action": subtasks[0].get("agent", "end"),
                "coordinator_id": coordinator_id,
            }

        llm = create_llm()
        
        # Get current browser context to help coordinator understand what's already visible
        browser_context = ""
        try:
            browser_agent = BrowserAgent()
            await browser_agent.start()
            pages = await browser_agent.list_pages()  # Returns list of dicts
            await browser_agent.stop()
            
            if pages and len(pages) > 0:
                current_page = pages[0]  # Active tab (first in list)
                all_tabs = "\n".join([f"  - Tab {p['index']}: {p.get('title', 'untitled')} ({p.get('url', '')})" for p in pages])
                browser_context = f"""
CURRENT BROWSER STATE:
- Active tab: {current_page.get('title', 'unknown')}
- Active URL: {current_page.get('url', 'unknown')}
- All open tabs:
{all_tabs}

IMPORTANT: Check if the content user wants is ALREADY visible on the active tab.
If yes, just "extract" with empty params {{}}. If on wrong tab, use "switch_tab" first.
"""
                print(f"[COORDINATOR] Browser context: Active tab = {current_page.get('url')}")
                print(f"[COORDINATOR] Full browser_context:\n{browser_context}")
        except Exception as e:
            import traceback
            print(f"[COORDINATOR] Could not get browser context: {e}")
            traceback.print_exc()
        
        system_content = COORDINATOR_SYSTEM_PROMPT + browser_context
        messages = [SystemMessage(content=system_content)] + state["messages"]
        
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
            print(f"[COORDINATOR] ⚠️ Content length: {len(content)}, error at pos: {je.pos}")
            
            # Try to parse just up to the error position (likely extra data after valid JSON)
            if je.pos and je.pos > 10:
                try:
                    # Find the last closing brace before the error
                    truncated = content[:je.pos].rstrip()
                    parsed = json.loads(truncated)
                    subtasks = parsed.get("subtasks", [])
                    response_text = parsed.get("response", "Executing tasks...")
                    print(f"[COORDINATOR] ✅ Recovered JSON (truncated at {je.pos}) with {len(subtasks)} subtasks")
                    
                    if subtasks:
                        await ws_manager.broadcast(
                            EventType.AGENT_UPDATE,
                            {"id": coordinator_id, "status": "busy", "summary": f"Delegating {len(subtasks)} task(s)"},
                        )
                        return {
                            "messages": [AIMessage(content=response_text)],
                            "subtasks": subtasks,
                            "current_subtask_index": 0,
                            "next_action": subtasks[0].get("agent", "end"),
                            "coordinator_id": coordinator_id,
                        }
                except Exception as recover_err:
                    print(f"[COORDINATOR] ⚠️ Truncated JSON recovery failed: {recover_err}")
            
            # Last resort: try to find balanced braces
            try:
                depth = 0
                start = content.find('{')
                if start >= 0:
                    for i, c in enumerate(content[start:], start):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                extracted = content[start:i+1]
                                parsed = json.loads(extracted)
                                subtasks = parsed.get("subtasks", [])
                                response_text = parsed.get("response", "Executing tasks...")
                                print(f"[COORDINATOR] ✅ Recovered JSON (balanced braces) with {len(subtasks)} subtasks")
                                
                                if subtasks:
                                    await ws_manager.broadcast(
                                        EventType.AGENT_UPDATE,
                                        {"id": coordinator_id, "status": "busy", "summary": f"Delegating {len(subtasks)} task(s)"},
                                    )
                                    return {
                                        "messages": [AIMessage(content=response_text)],
                                        "subtasks": subtasks,
                                        "current_subtask_index": 0,
                                        "next_action": subtasks[0].get("agent", "end"),
                                        "coordinator_id": coordinator_id,
                                    }
                                break
            except Exception as brace_err:
                print(f"[COORDINATOR] ⚠️ Balanced brace recovery failed: {brace_err}")
            await ws_manager.broadcast(
                EventType.AGENT_UPDATE,
                {
                    "id": coordinator_id,
                    "status": "idle",
                    "summary": "Conversational response",
                },
            )
            # Remove from Active Agents
            await ws_manager.broadcast(
                EventType.AGENT_REMOVED,
                {"id": coordinator_id},
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
        elif "edit_fields" in action_lower or ("edit" in action_lower and "field" in action_lower):
            # Generic edit: locate an item by match_text and set/swap fields
            match_text = params.get("match_text")
            if not match_text:
                result = {"error": "edit_fields requires match_text"}
            else:
                result = await agent.edit_fields(
                    match_text=match_text,
                    fields=params.get("fields") or None,
                    swap=params.get("swap") or None,
                    open_edit_texts=params.get("open_edit_texts") or None,
                    save_texts=params.get("save_texts") or None,
                    timeout=int(params.get("timeout", 15000) or 15000),
                )
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
        # Mark coordinator as complete and remove
        if coordinator_id:
            await ws_manager.broadcast(
                EventType.AGENT_UPDATE,
                {
                    "id": coordinator_id,
                    "status": "idle",
                    "summary": "No tasks to execute",
                },
            )
            await ws_manager.broadcast(
                EventType.AGENT_REMOVED,
                {"id": coordinator_id},
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
    
    # Create a summary prompt that preserves data
    # Convert results to a more readable format
    results_str = str(results)
    
    # Check if results contain extracted text (likely has actual content to show)
    has_extracted_text = any(
        isinstance(r, dict) and ("text" in r or "results" in r)
        for r in results if isinstance(r, dict)
    )
    
    if has_extracted_text:
        # User likely wants to see the actual data
        summary_prompt = f"""The user requested data extraction. Here are the results:

{results_str[:80000]}

Create a response that:
1. Briefly describes what was done (1-2 sentences)
2. INCLUDES THE ACTUAL EXTRACTED DATA - lists, titles, text, etc.
3. If there's a list of items (like patent titles, names, etc.), format them clearly
4. Don't just say "found X items" - SHOW the items

{"Error encountered: " + error if error else ""}"""
    else:
        summary_prompt = f"""Summarize what was accomplished:

Results:
{results_str[:20000]}

{"Error: " + error if error else ""}

Provide a brief, user-friendly summary of what was done."""

    messages = [
        SystemMessage(content="You are a helpful assistant. When data was extracted, ALWAYS include the actual data in your response, not just a summary of it."),
        HumanMessage(content=summary_prompt),
    ]
    
    response = await llm.ainvoke(messages)
    summary_text = response.content if hasattr(response, "content") else str(response)
    
    await ws_manager.broadcast_log(
        level="info",
        message="Task execution completed",
        category="coordinator",
    )
    
    # Broadcast the summary as a chat message to ALL clients
    # This ensures delivery even if the original WebSocket had issues
    session_id = state.get("session_id", "default")
    await ws_manager.broadcast(
        EventType.CHAT_MESSAGE,
        {
            "role": "assistant",
            "content": summary_text,
            "session_id": session_id,
        },
    )
    print(f"[SUMMARIZE] Broadcast summary: {summary_text[:100]}...")
    
    # Mark coordinator as complete and remove from UI
    if coordinator_id:
        await ws_manager.broadcast(
            EventType.AGENT_UPDATE,
            {
                "id": coordinator_id,
                "status": "idle",
                "summary": "Task completed",
            },
        )
        # Remove coordinator from Active Agents after a brief moment
        import asyncio
        await asyncio.sleep(0.5)
        await ws_manager.broadcast(
            EventType.AGENT_REMOVED,
            {"id": coordinator_id},
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

