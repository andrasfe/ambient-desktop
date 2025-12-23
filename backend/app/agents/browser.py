"""Browser agent using Playwright for web automation."""

import asyncio
import base64
from typing import Any, Optional
from pathlib import Path
from difflib import SequenceMatcher

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from .base import BaseAgent
from ..models import AgentType, AgentStatus, Task
from ..config import settings


class BrowserAgent(BaseAgent):
    """Agent for browser automation using Playwright.
    
    Supports three modes:
    1. Fresh browser (headless) - default, isolated session
    2. Persistent context - saves cookies/logins between sessions
    3. CDP connection - takes over your existing browser session
    """

    agent_type = AgentType.BROWSER

    def __init__(
        self,
        name: Optional[str] = None,
        headless: bool = True,
        cdp_url: Optional[str] = None,  # e.g., "http://localhost:9222"
        user_data_dir: Optional[str] = None,  # For persistent sessions
    ):
        super().__init__(name)
        self.headless = headless
        self.cdp_url = cdp_url
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._owns_browser = True  # False when connecting to existing browser

    async def start(self) -> None:
        """Start the browser agent."""
        await super().start()
        
        self._playwright = await async_playwright().start()
        
        if self.cdp_url:
            # Connect to existing browser via CDP
            await self._connect_to_existing_browser()
        elif self.user_data_dir:
            # Use persistent browser context
            await self._start_persistent_browser()
        else:
            # Launch fresh browser
            await self._start_fresh_browser()

    async def _connect_to_existing_browser(self) -> None:
        """Connect to an existing browser via Chrome DevTools Protocol."""
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
            self._owns_browser = False
            
            # Get all contexts from the browser
            contexts = self._browser.contexts
            print(f"[BROWSER] CDP connected, found {len(contexts)} contexts")
            
            if contexts:
                self._context = contexts[0]
                pages = self._context.pages
                print(f"[BROWSER] Context has {len(pages)} pages")
                
                # Find a non-blank, non-extension page to use as default
                selected_page = None
                for i, page in enumerate(pages):
                    url = page.url
                    print(f"[BROWSER]   Page {i}: {url[:80]}")
                    # Skip blank, extension, and chrome internal pages
                    if url and not url.startswith(('about:', 'chrome:', 'chrome-extension:')):
                        if selected_page is None:
                            selected_page = page
                            print(f"[BROWSER]   -> Selected as default")
                
                if selected_page:
                    self._page = selected_page
                elif pages:
                    self._page = pages[0]
                else:
                    self._page = await self._context.new_page()
            else:
                print("[BROWSER] No contexts found, creating new one")
                self._context = await self._browser.new_context()
                self._page = await self._context.new_page()
            
            await self.log("info", f"Connected to existing browser via CDP: {self.cdp_url}")
            await self.log("info", f"Current URL: {self._page.url}")
        except Exception as e:
            await self.log("error", f"Failed to connect to browser: {e}")
            raise

    async def _start_persistent_browser(self) -> None:
        """Start browser with persistent storage (keeps cookies/logins)."""
        user_data_path = Path(self.user_data_dir).expanduser()
        user_data_path.mkdir(parents=True, exist_ok=True)
        
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_path),
            headless=self.headless,
            viewport={"width": 1280, "height": 800},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        
        await self.log("info", f"Browser launched with persistent context: {user_data_path}")

    async def _start_fresh_browser(self) -> None:
        """Start a fresh browser instance."""
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        self._page = await self._context.new_page()
        
        await self.log("info", "Fresh browser launched")

    async def stop(self) -> None:
        """Stop the browser agent."""
        # Only close browser if we own it (not CDP connection)
        if self._owns_browser:
            if self._browser:
                await self._browser.close()
        else:
            # For CDP, just disconnect (don't close user's browser)
            if self._browser:
                await self._browser.close()  # This disconnects without closing
        
        if self._playwright:
            await self._playwright.stop()
        
        await super().stop()

    async def use_existing_page(self, page_index: int = 0) -> dict[str, Any]:
        """Switch to an existing page/tab in the connected browser."""
        if not self._context:
            raise RuntimeError("Browser not connected")
        
        pages = self._context.pages
        if page_index >= len(pages):
            raise ValueError(f"Page index {page_index} out of range (have {len(pages)} pages)")
        
        self._page = pages[page_index]
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "page_index": page_index,
            "total_pages": len(pages),
        }

    async def list_pages(self) -> list[dict[str, Any]]:
        """List all open pages/tabs in the connected browser."""
        if not self._context:
            return []
        
        pages = []
        for i, page in enumerate(self._context.pages):
            url = page.url
            # Skip blank, extension, and chrome internal pages
            if url.startswith(('about:', 'chrome:', 'chrome-extension:')):
                continue
            try:
                title = await page.title()
            except:
                title = "Unknown"
            pages.append({
                "index": i,
                "url": url,
                "title": title,
            })
        return pages

    async def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a browser task."""
        action = task.payload.get("action", "navigate")
        
        handlers = {
            "navigate": self._action_navigate,
            "click": self._action_click,
            "type": self._action_type,
            "screenshot": self._action_screenshot,
            "extract": self._action_extract,
            "wait": self._action_wait,
            "scroll": self._action_scroll,
            "evaluate": self._action_evaluate,
            "edit_fields": self._action_edit_fields,
        }
        
        handler = handlers.get(action)
        if not handler:
            raise ValueError(f"Unknown browser action: {action}")
        
        await self.log("info", f"Executing browser action: {action}")
        return await handler(task.payload)

    async def _action_navigate(self, payload: dict) -> dict[str, Any]:
        """Navigate to a URL."""
        url = payload.get("url")
        if not url:
            raise ValueError("URL required for navigate action")
        
        await self.update_status(AgentStatus.BUSY, summary=f"Navigating to {url}")
        
        try:
            response = await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)  # 15 second timeout
            
            return {
                "url": self._page.url,
                "title": await self._page.title(),
                "status": response.status if response else None,
            }
        except Exception as e:
            return {
                "error": f"Navigation failed: {str(e)}",
                "url": url,
            }

    async def _action_click(self, payload: dict) -> dict[str, Any]:
        """Click on an element."""
        selector = payload.get("selector")
        text = payload.get("text")
        timeout = payload.get("timeout", 15000)  # 15 second default for dynamic sites
        
        await self.update_status(AgentStatus.BUSY, summary=f"Clicking: {selector or text}")
        
        try:
            if text:
                locator = self._page.get_by_text(text, exact=False)
                # Check if element exists
                if await locator.count() == 0:
                    return {
                        "error": f"No element found with text: {text}",
                        "url": self._page.url,
                        "suggestion": "Try a different text or use a CSS selector",
                    }
                await locator.first.click(timeout=timeout)
            elif selector:
                locator = self._page.locator(selector)
                # Check if element exists
                if await locator.count() == 0:
                    return {
                        "error": f"No element found matching: {selector}",
                        "url": self._page.url,
                        "suggestion": "Check the selector or try text-based clicking",
                    }
                await locator.first.click(timeout=timeout)
            else:
                return {"error": "Selector or text required for click action"}
            
            # Wait a moment for page to react
            await asyncio.sleep(0.5)
            
            return {
                "clicked": selector or text,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as e:
            return {
                "error": f"Click failed: {str(e)}",
                "target": selector or text,
                "url": self._page.url,
            }

    async def _action_type(self, payload: dict) -> dict[str, Any]:
        """Type text into an element."""
        selector = payload.get("selector")
        text = payload.get("text", "")
        clear = payload.get("clear", True)
        
        if not selector:
            raise ValueError("Selector required for type action")
        
        if clear:
            await self._page.fill(selector, text)
        else:
            await self._page.type(selector, text)
        
        await self.update_status(AgentStatus.BUSY, summary=f"Typed into: {selector}")
        
        return {"typed": True, "selector": selector}

    async def _action_screenshot(self, payload: dict) -> dict[str, Any]:
        """Take a screenshot."""
        full_page = payload.get("full_page", False)
        selector = payload.get("selector")
        
        if selector:
            element = self._page.locator(selector)
            screenshot = await element.screenshot()
        else:
            screenshot = await self._page.screenshot(full_page=full_page)
        
        # Return as base64
        b64_screenshot = base64.b64encode(screenshot).decode("utf-8")
        
        await self.update_status(AgentStatus.BUSY, summary="Screenshot taken")
        
        return {
            "screenshot": b64_screenshot,
            "format": "png",
        }

    async def _action_extract(self, payload: dict) -> dict[str, Any]:
        """Extract text or data from the page."""
        selector = payload.get("selector")
        attribute = payload.get("attribute")
        all_matches = payload.get("all", False)
        timeout = payload.get("timeout", 15000)  # 15 second default for dynamic sites
        full_page = payload.get("full_page", True)  # Auto-scroll and expand by default
        
        await self.update_status(AgentStatus.BUSY, summary=f"Extracting from: {selector or 'page'}")
        
        # Wait for network to settle and page to be ready
        try:
            await self._page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # Continue even if network doesn't settle
        
        # Auto-scroll and expand for full page extraction
        if full_page and not selector:
            await self._load_full_page()
        
        # Get page info for context
        page_info = {
            "url": self._page.url,
            "title": await self._page.title(),
        }
        
        if not selector:
            # Extract all visible text (limit to 100KB for safety)
            text = await self._page.inner_text("body")
            max_len = 100000  # 100KB for full page extractions
            return {"text": text[:max_len], "truncated": len(text) > max_len, **page_info}
        
        try:
            # Check if element exists with timeout
            locator = self._page.locator(selector)
            count = await locator.count()
            
            if count == 0:
                # Element not found - return helpful info
                return {
                    "error": f"No elements found matching: {selector}",
                    "suggestion": "Try a different selector or extract full page text",
                    **page_info,
                }
            
            if all_matches:
                if attribute:
                    results = [await locator.nth(i).get_attribute(attribute) for i in range(min(count, 50))]
                else:
                    results = [await locator.nth(i).inner_text() for i in range(min(count, 50))]
                
                return {"results": results, "count": count, "returned": min(count, 50), **page_info}
            else:
                element = locator.first
                
                # Wait for element to be visible with timeout
                try:
                    await element.wait_for(state="visible", timeout=timeout)
                except Exception:
                    pass  # Element might exist but not be "visible" in playwright terms
                
                if attribute:
                    result = await element.get_attribute(attribute)
                else:
                    result = await element.inner_text()
                
                return {"result": result, **page_info}
                
        except Exception as e:
            return {
                "error": f"Extraction failed: {str(e)}",
                "selector": selector,
                **page_info,
            }

    async def _load_full_page(self) -> None:
        """Scroll through page and click 'show more' buttons to load all content."""
        import asyncio
        
        # Common "show more" button patterns (generic, not site-specific)
        more_button_patterns = [
            "text=/show\\s*(all|more)/i",
            "text=/load\\s*more/i",
            "text=/see\\s*(all|more)/i",
            "text=/view\\s*(all|more)/i",
            "text=/expand/i",
            "[aria-label*='more' i]",
            "[aria-label*='expand' i]",
            "button:has-text('more')",
            "a:has-text('more')",
        ]
        
        await self.update_status(AgentStatus.BUSY, summary="Loading full page content...")
        
        # Scroll to bottom progressively to trigger lazy loading
        prev_height = 0
        scroll_attempts = 0
        max_scrolls = 20  # Limit scrolling to prevent infinite loops
        
        while scroll_attempts < max_scrolls:
            # Get current scroll height
            current_height = await self._page.evaluate("document.body.scrollHeight")
            
            if current_height == prev_height:
                break  # No more content to load
            
            prev_height = current_height
            
            # Scroll down
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)  # Wait for lazy content to load
            
            # Try to click any "show more" buttons that appeared
            for pattern in more_button_patterns:
                try:
                    locator = self._page.locator(pattern)
                    if await locator.count() > 0:
                        # Click all visible "more" buttons
                        for i in range(min(await locator.count(), 5)):  # Max 5 clicks per pattern
                            try:
                                btn = locator.nth(i)
                                if await btn.is_visible():
                                    await btn.click(timeout=2000)
                                    await asyncio.sleep(0.3)
                            except Exception:
                                pass
                except Exception:
                    pass
            
            scroll_attempts += 1
        
        # Scroll back to top
        await self._page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.3)
        
        # Final wait for any remaining content
        try:
            await self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

    async def _action_wait(self, payload: dict) -> dict[str, Any]:
        """Wait for an element or time."""
        selector = payload.get("selector")
        timeout = payload.get("timeout", 15000)  # 15 second default
        state = payload.get("state", "visible")  # visible, hidden, attached, detached
        
        await self.update_status(AgentStatus.BUSY, summary=f"Waiting for: {selector or f'{timeout}ms'}")
        
        if selector:
            try:
                await self._page.wait_for_selector(selector, state=state, timeout=timeout)
                return {"waited_for": selector, "state": state, "found": True}
            except Exception as e:
                return {"waited_for": selector, "state": state, "found": False, "error": str(e)}
        else:
            await asyncio.sleep(timeout / 1000)
            return {"waited_ms": timeout}

    async def _action_scroll(self, payload: dict) -> dict[str, Any]:
        """Scroll the page."""
        direction = payload.get("direction", "down")
        amount = payload.get("amount", 500)
        selector = payload.get("selector")
        to_bottom = payload.get("to_bottom", False)
        section = payload.get("section")  # e.g., "patents", "experience"
        
        await self.update_status(AgentStatus.BUSY, summary=f"Scrolling {direction}")
        
        try:
            if section:
                # Try to find a section by common patterns (LinkedIn, etc.)
                section_selectors = [
                    f"[id*='{section}' i]",
                    f"[class*='{section}' i]",
                    f"section:has-text('{section}')",
                    f"div:has-text('{section}')",
                    f"h2:has-text('{section}')",
                    f"h3:has-text('{section}')",
                ]
                for sel in section_selectors:
                    try:
                        locator = self._page.locator(sel).first
                        if await locator.count() > 0:
                            await locator.scroll_into_view_if_needed()
                            await asyncio.sleep(1)  # Wait for content to load
                            return {"scrolled_to_section": section, "selector": sel, "url": self._page.url}
                    except Exception:
                        continue
                # Section not found, scroll down to look for it
                for _ in range(5):
                    await self._page.mouse.wheel(0, 800)
                    await asyncio.sleep(0.5)
                return {"scrolled_looking_for": section, "note": "Section may not exist on this page"}
                
            elif selector:
                element = self._page.locator(selector)
                if await element.count() > 0:
                    await element.first.scroll_into_view_if_needed()
                else:
                    return {"error": f"Element not found: {selector}"}
            elif to_bottom:
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                delta_y = amount if direction == "down" else -amount
                await self._page.mouse.wheel(0, delta_y)
            
            # Wait for dynamic content to load
            await asyncio.sleep(1)
            
            return {"scrolled": direction, "amount": amount, "url": self._page.url}
        except Exception as e:
            return {"error": f"Scroll failed: {str(e)}"}

    async def _action_evaluate(self, payload: dict) -> dict[str, Any]:
        """Execute JavaScript on the page."""
        script = payload.get("script")
        
        if not script:
            raise ValueError("Script required for evaluate action")
        
        await self.update_status(AgentStatus.BUSY, summary="Evaluating JavaScript")
        
        result = await self._page.evaluate(script)
        
        return {"result": result}

    async def _action_edit_fields(self, payload: dict) -> dict[str, Any]:
        """Generic edit helper: open an item's edit UI by matching text, then set/swap fields by label.

        Params:
          - match_text (str, required): text that identifies the item to edit (e.g., an ID / number / title)
          - fields (dict[str, str], optional): field_name -> new value (field_name is matched fuzzily)
          - swap (list[str], optional): two field names to swap values between (names are matched fuzzily)
          - open_edit_texts (list[str], optional): button texts to open editor (defaults include Edit/Update/Change)
          - save_texts (list[str], optional): button texts to save changes (defaults include Save/Done/Apply/Update)
          - timeout (int ms, optional): per-step timeout (default 15000)
        """
        import asyncio

        if not self._page:
            raise RuntimeError("Browser page not initialized")

        match_text = payload.get("match_text")
        if not match_text:
            return {"error": "match_text is required for edit_fields"}

        fields: dict[str, str] = payload.get("fields") or {}
        swap_labels = payload.get("swap") or []
        timeout = payload.get("timeout", 15000)

        open_edit_texts = payload.get(
            "open_edit_texts",
            ["Edit", "Update", "Change", "Modify", "Pencil"],
        )
        save_texts = payload.get(
            "save_texts",
            ["Save", "Done", "Apply", "Update", "OK", "Confirm"],
        )

        await self.update_status(AgentStatus.BUSY, summary=f"Editing fields for item matching: {match_text[:40]}")

        # Locate the item by text
        item_locator = self._page.get_by_text(match_text, exact=False)
        if await item_locator.count() == 0:
            return {"error": f"Could not find item containing text: {match_text}", "url": self._page.url}

        # Best-effort: open edit UI near the item.
        opened = False
        last_open_error: str | None = None

        # 0) Prefer clicking an edit/pencil control near the matched item (generic DOM search).
        # This follows the user's intent: "pencil next to it" on list/detail pages.
        try:
            clicked_near_item = await self._page.evaluate(
                """
                (matchText) => {
                  const norm = (s) => (s || '').toString().replace(/\\s+/g,' ').trim().toLowerCase();
                  const target = norm(matchText);
                  // Find an element whose text includes matchText
                  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
                  let node = walker.currentNode;
                  let found = null;
                  while (node) {
                    const t = norm(node.innerText);
                    if (t && target && t.includes(target)) { found = node; break; }
                    node = walker.nextNode();
                  }
                  if (!found) return { clicked: false, reason: 'match_not_found' };
                  // Walk up a few ancestors and look for an edit control
                  const re = /(edit|pencil|modify|change|update)/i;
                  let cur = found;
                  for (let depth = 0; depth < 7 && cur; depth++) {
                    // candidates: buttons/links or anything with role=button
                    const candidates = Array.from(cur.querySelectorAll('button, a, [role=\"button\"]'));
                    for (const el of candidates) {
                      const aria = el.getAttribute('aria-label') || '';
                      const title = el.getAttribute('title') || '';
                      const txt = el.innerText || '';
                      if (re.test(aria) || re.test(title) || re.test(txt)) {
                        el.click();
                        return { clicked: true, via: 'near_item', hint: aria || title || txt };
                      }
                      // icon-only buttons sometimes only have svg; keep as fallback if label/title indicates edit
                      if ((aria || title) && re.test(aria + ' ' + title)) {
                        el.click();
                        return { clicked: true, via: 'near_item_icon', hint: aria || title };
                      }
                    }
                    cur = cur.parentElement;
                  }
                  return { clicked: false, reason: 'no_edit_control_near_item' };
                }
                """,
                match_text,
            )
            if isinstance(clicked_near_item, dict) and clicked_near_item.get("clicked"):
                opened = True
        except Exception as e:
            last_open_error = str(e)

        # 1) Try common edit button texts anywhere (generic)
        if not opened:
            for t in open_edit_texts:
                try:
                    btn = self._page.get_by_role("button", name=t)
                    if await btn.count() > 0:
                        await btn.first.click(timeout=3000)
                        opened = True
                        break
                except Exception as e:
                    last_open_error = str(e)

        # 2) Try aria-label patterns (generic)
        if not opened:
            for sel in [
                "[aria-label*='edit' i]",
                "[aria-label*='update' i]",
                "button:has-text('Edit')",
                "button:has-text('edit')",
            ]:
                try:
                    loc = self._page.locator(sel)
                    if await loc.count() > 0:
                        await loc.first.click(timeout=3000)
                        opened = True
                        break
                except Exception as e:
                    last_open_error = str(e)

        # Wait a beat for dialogs/panels to appear if something was clicked.
        if opened:
            await asyncio.sleep(0.5)

        def _norm(s: str) -> str:
            return " ".join((s or "").lower().replace("_", " ").replace("-", " ").split())

        def _token_score(a: str, b: str) -> float:
            a_toks = set(_norm(a).split())
            b_toks = set(_norm(b).split())
            if not a_toks or not b_toks:
                return 0.0
            return len(a_toks & b_toks) / max(len(a_toks), len(b_toks))

        def _sim(a: str, b: str) -> float:
            return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

        def _score(target: str, hint: str) -> float:
            # Heuristic similarity: char similarity + token overlap
            return 0.65 * _sim(target, hint) + 0.35 * _token_score(target, hint)

        async def _load_field_candidates() -> list[dict[str, Any]]:
            """Return metadata for inputs on the page so we can fuzzy-match field names locally.

            We intentionally keep this generic: label text, aria-label, placeholder, name/id, plus nearby text.
            """
            return await self._page.evaluate(
                """
                () => {
                  const els = Array.from(document.querySelectorAll('input, textarea, select, [contenteditable="true"]'));
                  const txt = (x) => (x || '').toString().replace(/\\s+/g,' ').trim();
                  const labelFor = (el) => {
                    const id = el.getAttribute('id');
                    if (id) {
                      const lab = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                      if (lab) return txt(lab.innerText);
                    }
                    const wrap = el.closest('label');
                    if (wrap) return txt(wrap.innerText);
                    // common pattern: a label preceding input in same container
                    const container = el.closest('div, section, form');
                    if (container) {
                      const lab = container.querySelector('label');
                      if (lab) return txt(lab.innerText);
                    }
                    return '';
                  };
                  return els.map((el, index) => {
                    const aria = txt(el.getAttribute('aria-label'));
                    const ph = txt(el.getAttribute('placeholder'));
                    const name = txt(el.getAttribute('name'));
                    const id = txt(el.getAttribute('id'));
                    const lab = labelFor(el);
                    const parent = el.parentElement;
                    const nearby = parent ? txt(parent.innerText).slice(0, 160) : '';
                    return { index, aria, placeholder: ph, name, id, label: lab, nearby };
                  });
                }
                """
            )

        field_candidates: list[dict[str, Any]] | None = None

        async def _resolve_field(field_name: str):
            # 1) Try Playwright label association first
            by_label = self._page.get_by_label(field_name, exact=False)
            if await by_label.count() > 0:
                return by_label

            # 2) Fuzzy match against input metadata gathered from the page
            nonlocal field_candidates
            if field_candidates is None:
                field_candidates = await _load_field_candidates()

            best_idx: int | None = None
            best_score = 0.0
            for cand in field_candidates:
                hints = [
                    cand.get("label", ""),
                    cand.get("aria", ""),
                    cand.get("placeholder", ""),
                    cand.get("name", ""),
                    cand.get("id", ""),
                    cand.get("nearby", ""),
                ]
                cand_best = 0.0
                for h in hints:
                    if h:
                        cand_best = max(cand_best, _score(field_name, h))
                if cand_best > best_score:
                    best_score = cand_best
                    best_idx = int(cand.get("index", -1))

            # Threshold to avoid random mismatches
            if best_idx is None or best_idx < 0 or best_score < 0.55:
                raise RuntimeError(f"Field not found (semantic match failed) for: {field_name}")

            inputs = self._page.locator("input, textarea, select, [contenteditable='true']")
            return inputs.nth(best_idx)

        # Helper to get and set by label/semantic match
        async def _get_value(label: str) -> str:
            el = await _resolve_field(label)
            try:
                return await el.first.input_value()
            except Exception:
                # Fallback for non-input elements
                return (await el.first.inner_text()) or ""

        async def _set_value(label: str, value: str) -> None:
            el = await _resolve_field(label)
            # Most forms use <input>/<textarea>
            try:
                await el.first.fill(value, timeout=timeout)
            except Exception:
                # Fallback: click + type (best effort)
                await el.first.click(timeout=timeout)
                await self._page.keyboard.press("Control+A")
                await self._page.keyboard.type(value)

        before: dict[str, str] = {}
        after: dict[str, str] = {}

        try:
            # Collect current values for swap targets and explicit fields
            labels_to_read = set(fields.keys())
            if isinstance(swap_labels, list) and len(swap_labels) == 2:
                labels_to_read.update([swap_labels[0], swap_labels[1]])

            for lbl in labels_to_read:
                try:
                    before[lbl] = await _get_value(lbl)
                except Exception:
                    # Ignore missing reads; will raise on set
                    pass

            # Apply swap if requested
            if isinstance(swap_labels, list) and len(swap_labels) == 2:
                a, b = swap_labels[0], swap_labels[1]
                a_val = before.get(a, await _get_value(a))
                b_val = before.get(b, await _get_value(b))
                await _set_value(a, b_val)
                await _set_value(b, a_val)
                after[a] = b_val
                after[b] = a_val

            # Apply explicit field updates
            for lbl, val in fields.items():
                await _set_value(lbl, val)
                after[lbl] = val

            # Click a save/confirm action if present
            saved = False
            for t in save_texts:
                try:
                    btn = self._page.get_by_role("button", name=t)
                    if await btn.count() > 0 and await btn.first.is_visible():
                        await btn.first.click(timeout=timeout)
                        saved = True
                        break
                except Exception:
                    continue

            return {
                "match_text": match_text,
                "opened_editor": opened,
                "open_error": None if opened else last_open_error,
                "saved": saved,
                "before": before,
                "after": after,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as e:
            return {
                "error": f"edit_fields failed: {str(e)}",
                "match_text": match_text,
                "opened_editor": opened,
                "open_error": None if opened else last_open_error,
                "before": before,
                "after": after,
                "url": self._page.url,
            }

    async def edit_fields(
        self,
        match_text: str,
        fields: dict[str, str] | None = None,
        swap: list[str] | None = None,
        open_edit_texts: list[str] | None = None,
        save_texts: list[str] | None = None,
        timeout: int = 15000,
    ) -> dict[str, Any]:
        """Convenience wrapper for edit_fields."""
        return await self._action_edit_fields(
            {
                "match_text": match_text,
                "fields": fields or {},
                "swap": swap or [],
                "open_edit_texts": open_edit_texts,
                "save_texts": save_texts,
                "timeout": timeout,
            }
        )

    # Convenience methods for programmatic use
    async def navigate(self, url: str) -> dict[str, Any]:
        """Navigate to a URL."""
        return await self._action_navigate({"url": url})

    async def click(self, selector: str = None, text: str = None) -> dict[str, Any]:
        """Click on an element."""
        return await self._action_click({"selector": selector, "text": text})

    async def type_text(self, selector: str, text: str, clear: bool = True) -> dict[str, Any]:
        """Type text into an element."""
        return await self._action_type({"selector": selector, "text": text, "clear": clear})

    async def screenshot(self, full_page: bool = False, selector: str = None) -> dict[str, Any]:
        """Take a screenshot."""
        return await self._action_screenshot({"full_page": full_page, "selector": selector})

    async def extract(self, selector: str = None, attribute: str = None, all: bool = False) -> dict[str, Any]:
        """Extract text or data from the page."""
        return await self._action_extract({"selector": selector, "attribute": attribute, "all": all})

    async def scroll(
        self,
        direction: str = "down",
        amount: int = 500,
        selector: str | None = None,
        to_bottom: bool = False,
        section: str | None = None,
    ) -> dict[str, Any]:
        """Scroll the page.

        Supports generic scrolling as well as scrolling to a selector or a named section.
        """
        return await self._action_scroll(
            {
                "direction": direction,
                "amount": amount,
                "selector": selector,
                "to_bottom": to_bottom,
                "section": section,
            }
        )

    @property
    def page(self) -> Optional[Page]:
        """Get the current page."""
        return self._page

    @property
    def is_ready(self) -> bool:
        """Check if the browser is ready."""
        return self._browser is not None and self._page is not None

