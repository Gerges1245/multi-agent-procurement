import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class SearchInput(BaseModel):
    query: str = Field(description="A specific product search query.")


class ProductSearchTool(BaseTool):
    name: str = "search_egypt_products"
    description: str = (
        "Search B.TECH and 2B for products in Egypt. "
        "Returns source URLs and search snippets, not verified prices."
    )
    args_schema: type[BaseModel] = SearchInput

    evidence: list[dict] = Field(default_factory=list, exclude=True)
    calls: int = Field(default=0, exclude=True)

    def _run(self, query: str) -> str:
        if self.calls >= 3:
            return json.dumps({
                "error": "Search limit reached. Use existing results."
            })

        self.calls += 1
        domains = ["btech.com", "2b.com.eg"]

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=domains,
            include_answer=False,
        )

        results = []
        for item in response.get("results", []):
            url = item.get("url", "")
            host = (urlparse(url).hostname or "").lower()

            if not any(
                host == domain or host.endswith("." + domain)
                for domain in domains
            ):
                continue

            results.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("content", "")[:1800],
            })

        record = {
            "query": query,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
        self.evidence.append(record)

        return json.dumps(record, ensure_ascii=False)