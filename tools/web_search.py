"""Web Search Tool - automatic web search without user interaction.

Uses DuckDuckGo (free) or Tavily API for search.
"""
from typing import List
from dataclasses import dataclass
import httpx
from urllib.parse import quote_plus

from config import settings


@dataclass
class SearchResult:
    """A single search result"""
    title: str
    url: str
    snippet: str
    source: str = ""


class WebSearchTool:
    """
    Performs web searches automatically when queries need current information.
    No user interaction required - the router decides when to search.
    All methods are synchronous.
    """
    
    def __init__(self):
        self.settings = settings
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.settings.tavily_api_key:
            return self._search_tavily(query, max_results)
        else:
            return self._search_duckduckgo(query, max_results)
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo (free, no API key)"""
        try:
            # Try using new ddgs library first
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=r.get('title', ''),
                        url=r.get('href', r.get('link', '')),
                        snippet=r.get('body', r.get('snippet', '')),
                        source="DuckDuckGo"
                    ))
            return results
            
        except ImportError:
            return self._search_duckduckgo_html(query, max_results)
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return self._search_duckduckgo_html(query, max_results)
    
    def _search_duckduckgo_html(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback DuckDuckGo search via HTML"""
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            with httpx.Client(timeout=15.0) as client:
                response = client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                
                if response.status_code != 200:
                    return []
                
                results = []
                html = response.text
                
                import re
                pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+)</a>'
                matches = re.findall(pattern, html, re.DOTALL)
                
                for result_url, title, snippet in matches[:max_results]:
                    results.append(SearchResult(
                        title=title.strip(),
                        url=result_url,
                        snippet=snippet.strip(),
                        source="DuckDuckGo"
                    ))
                
                return results
            
        except Exception as e:
            print(f"DuckDuckGo HTML search error: {e}")
            return []
    
    def _search_tavily(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Tavily API (better quality, requires API key)"""
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": True,
                        "search_depth": "basic"
                    }
                )
                
                if response.status_code != 200:
                    return self._search_duckduckgo(query, max_results)
                
                data = response.json()
                results = []
                
                if data.get("answer"):
                    results.append(SearchResult(
                        title="AI Summary",
                        url="",
                        snippet=data["answer"],
                        source="Tavily AI"
                    ))
                
                for r in data.get("results", [])[:max_results]:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        source="Tavily"
                    ))
                
                return results
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return self._search_duckduckgo(query, max_results)
    
    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results as context for the LLM"""
        if not results:
            return ""
        
        lines = ["[Web Search Results]"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.title}")
            if r.url:
                lines.append(f"   Source: {r.url}")
            lines.append(f"   {r.snippet}")
        
        return "\n".join(lines)
