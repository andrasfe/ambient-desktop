"""Unit tests for browser agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.browser import BrowserAgent
from app.models import AgentType


class TestBrowserAgent:
    """Tests for BrowserAgent."""

    def test_init_default(self):
        """Test default initialization."""
        agent = BrowserAgent()
        assert agent.agent_type == AgentType.BROWSER
        assert agent.headless is True
        assert agent.cdp_url is None
        assert agent.user_data_dir is None
        assert agent._owns_browser is True

    def test_init_with_cdp(self):
        """Test initialization with CDP URL."""
        agent = BrowserAgent(cdp_url="http://localhost:9222")
        assert agent.cdp_url == "http://localhost:9222"

    def test_init_with_persistent(self):
        """Test initialization with persistent context."""
        agent = BrowserAgent(user_data_dir="/tmp/browser-profile")
        assert agent.user_data_dir == "/tmp/browser-profile"

    def test_init_headless_false(self):
        """Test initialization with headless=False."""
        agent = BrowserAgent(headless=False)
        assert agent.headless is False

    @pytest.mark.asyncio
    async def test_list_pages_no_context(self):
        """Test list_pages when no context is set."""
        agent = BrowserAgent()
        pages = await agent.list_pages()
        assert pages == []

    @pytest.mark.asyncio
    async def test_list_pages_with_pages(self):
        """Test list_pages with mock pages."""
        agent = BrowserAgent()
        
        # Create mock pages
        mock_page1 = AsyncMock()
        mock_page1.url = "https://example.com"
        mock_page1.title = AsyncMock(return_value="Example")
        
        mock_page2 = AsyncMock()
        mock_page2.url = "https://google.com"
        mock_page2.title = AsyncMock(return_value="Google")
        
        # Create mock context
        mock_context = MagicMock()
        mock_context.pages = [mock_page1, mock_page2]
        agent._context = mock_context
        
        pages = await agent.list_pages()
        
        assert len(pages) == 2
        assert pages[0]["url"] == "https://example.com"
        assert pages[0]["title"] == "Example"
        assert pages[1]["url"] == "https://google.com"
        assert pages[1]["title"] == "Google"

    @pytest.mark.asyncio
    async def test_use_existing_page(self):
        """Test switching to existing page."""
        agent = BrowserAgent()
        
        # Create mock pages
        mock_page1 = AsyncMock()
        mock_page1.url = "https://example.com"
        mock_page1.title = AsyncMock(return_value="Example")
        
        mock_page2 = AsyncMock()
        mock_page2.url = "https://google.com"
        mock_page2.title = AsyncMock(return_value="Google")
        
        # Create mock context
        mock_context = MagicMock()
        mock_context.pages = [mock_page1, mock_page2]
        agent._context = mock_context
        
        result = await agent.use_existing_page(1)
        
        assert result["url"] == "https://google.com"
        assert result["title"] == "Google"
        assert result["page_index"] == 1
        assert result["total_pages"] == 2
        assert agent._page == mock_page2

    @pytest.mark.asyncio
    async def test_use_existing_page_invalid_index(self):
        """Test switching to invalid page index."""
        agent = BrowserAgent()
        
        mock_context = MagicMock()
        mock_context.pages = []
        agent._context = mock_context
        
        with pytest.raises(ValueError, match="out of range"):
            await agent.use_existing_page(5)

    @pytest.mark.asyncio
    async def test_navigate(self):
        """Test navigation action."""
        agent = BrowserAgent()
        
        mock_response = MagicMock()
        mock_response.status = 200
        
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        
        agent._page = mock_page
        agent.update_status = AsyncMock()
        
        result = await agent.navigate("https://example.com")
        
        mock_page.goto.assert_called_once_with("https://example.com", wait_until="domcontentloaded", timeout=15000)
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_click_by_selector(self):
        """Test click by selector."""
        agent = BrowserAgent()
        
        mock_locator = AsyncMock()
        mock_locator.count = AsyncMock(return_value=1)
        mock_locator.first = AsyncMock()
        mock_locator.first.click = AsyncMock()

        mock_page = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)
        
        agent._page = mock_page
        agent.update_status = AsyncMock()
        
        result = await agent.click(selector="#button")
        
        mock_page.locator.assert_called_once_with("#button")
        mock_locator.first.click.assert_called_once()
        assert result["clicked"] == "#button"

    @pytest.mark.asyncio
    async def test_click_by_text(self):
        """Test click by text."""
        agent = BrowserAgent()
        
        mock_locator = AsyncMock()
        mock_locator.count = AsyncMock(return_value=1)
        mock_locator.first = AsyncMock()
        mock_locator.first.click = AsyncMock()
        
        mock_page = AsyncMock()
        mock_page.get_by_text = MagicMock(return_value=mock_locator)
        
        agent._page = mock_page
        agent.update_status = AsyncMock()
        
        result = await agent.click(text="Submit")
        
        mock_page.get_by_text.assert_called_once_with("Submit", exact=False)
        mock_locator.first.click.assert_called_once()
        assert result["clicked"] == "Submit"

    @pytest.mark.asyncio
    async def test_type_text(self):
        """Test typing text."""
        agent = BrowserAgent()
        
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        
        agent._page = mock_page
        agent.update_status = AsyncMock()
        
        result = await agent.type_text("#input", "Hello World")
        
        mock_page.fill.assert_called_once_with("#input", "Hello World")
        assert result["typed"] is True

    @pytest.mark.asyncio
    async def test_screenshot(self):
        """Test screenshot."""
        agent = BrowserAgent()
        
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"fake_image_data")
        
        agent._page = mock_page
        agent.update_status = AsyncMock()
        
        result = await agent.screenshot()
        
        mock_page.screenshot.assert_called_once_with(full_page=False)
        assert result["format"] == "png"
        assert "screenshot" in result

    @pytest.mark.asyncio
    async def test_extract_all_text(self):
        """Test extracting all visible text."""
        agent = BrowserAgent()
        
        mock_page = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="Page content here")
        
        agent._page = mock_page
        agent.update_status = AsyncMock()
        
        result = await agent.extract()
        
        mock_page.inner_text.assert_called_once_with("body")
        assert result["text"] == "Page content here"

    @pytest.mark.asyncio
    async def test_edit_fields_semantic_match_fills(self):
        """edit_fields should fall back to semantic-ish matching when get_by_label finds nothing."""
        agent = BrowserAgent()

        # Mock minimal page APIs used by edit_fields.
        mock_page = AsyncMock()

        # match_text exists
        mock_item = AsyncMock()
        mock_item.count = AsyncMock(return_value=1)
        mock_page.get_by_text = MagicMock(return_value=mock_item)

        # No edit button is required for this unit test; make clicks no-op / absent.
        mock_btn = AsyncMock()
        mock_btn.count = AsyncMock(return_value=0)
        mock_page.get_by_role = MagicMock(return_value=mock_btn)

        # get_by_label finds nothing, forcing fallback
        mock_empty_loc = AsyncMock()
        mock_empty_loc.count = AsyncMock(return_value=0)
        mock_page.get_by_label = MagicMock(return_value=mock_empty_loc)

        # Evaluate returns candidate inputs metadata (index 0 is "Title", index 1 is "Patent number")
        mock_page.evaluate = AsyncMock(
            return_value=[
                {"index": 0, "label": "Title", "aria": "", "placeholder": "", "name": "title", "id": "title", "nearby": ""},
                {"index": 1, "label": "Patent number", "aria": "", "placeholder": "", "name": "patentNumber", "id": "patentNumber", "nearby": ""},
            ]
        )

        # locator(...) returns a collection we can nth() into.
        mock_inputs = MagicMock()
        input0 = AsyncMock()
        input0.first = input0
        input0.fill = AsyncMock()
        input0.click = AsyncMock()
        input0.input_value = AsyncMock(return_value="Original Title")

        input1 = AsyncMock()
        input1.first = input1
        input1.fill = AsyncMock()
        input1.click = AsyncMock()
        input1.input_value = AsyncMock(return_value="US-123")

        mock_inputs.nth = MagicMock(side_effect=[input0, input1])
        mock_page.locator = MagicMock(return_value=mock_inputs)

        # Other methods referenced
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.keyboard.type = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")

        agent._page = mock_page
        agent.update_status = AsyncMock()

        res = await agent._action_edit_fields(
            {
                "match_text": "US-123",
                "fields": {"Patent No": "US-123"},  # fuzzy match to "Patent number"
                "timeout": 15000,
            }
        )

        assert res.get("error") is None
        # Should have filled the 2nd field (index 1)
        assert input1.fill.called

    def test_is_ready_false(self):
        """Test is_ready when browser not started."""
        agent = BrowserAgent()
        assert agent.is_ready is False

    def test_is_ready_true(self):
        """Test is_ready when browser is started."""
        agent = BrowserAgent()
        agent._browser = MagicMock()
        agent._page = MagicMock()
        assert agent.is_ready is True

    @pytest.mark.asyncio
    async def test_stop_owned_browser(self):
        """Test stopping owned browser."""
        agent = BrowserAgent()
        agent._owns_browser = True
        
        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock()
        agent._browser = mock_browser
        
        mock_playwright = AsyncMock()
        mock_playwright.stop = AsyncMock()
        agent._playwright = mock_playwright
        
        agent.update_status = AsyncMock()
        
        await agent.stop()
        
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cdp_connection(self):
        """Test stopping CDP connection (disconnect without closing)."""
        agent = BrowserAgent(cdp_url="http://localhost:9222")
        agent._owns_browser = False
        
        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock()
        agent._browser = mock_browser
        
        mock_playwright = AsyncMock()
        mock_playwright.stop = AsyncMock()
        agent._playwright = mock_playwright
        
        agent.update_status = AsyncMock()
        
        await agent.stop()
        
        # Still calls close (which disconnects for CDP)
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

