"""
tools/fetch_tools.py

Web fetching and URL validation tools.
Used by: WebScraper agent (fetch_page, validate_url)
         GrantAnalyzer agent (fetch_page)
"""

import os
import requests
from bs4 import BeautifulSoup
from agents import function_tool, RunContextWrapper
from models.schemas import UrlValidationResult
from pipeline.context import PipelineContext

@function_tool
def fetch_page(url: str) -> str:
    """
    Fetch and extract readable text from a web page URL.
    Use this to retrieve content from an organization's website or a grant page.
    Returns clean text with structure preserved. If the page is unreachable,
    returns the error string — do not raise an exception.
    """
    try:
        response = requests.get(
            url,
            timeout=10,                                         # give up after 10 seconds
            headers={"User-Agent": "Mozilla/5.0"}               # some sites block requests with no user agent
        )
        response.raise_for_status()                             # raises an error if status is 4xx or 5xx

        soup = BeautifulSoup(response.text, "html.parser")      # parse the raw HTML

        # remove tags that aren't readable content
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()                                     # remove the tag and its contents

        return soup.get_text(separator="\n", strip=True)        # return clean text, one line per element

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"                # return error as string, never crash


@function_tool
def validate_url(url: str) -> UrlValidationResult:
    """
    Check whether a URL is reachable and appears to be a legitimate organization website.
    Use once with the submitted URL before any other processing.
    Flag as invalid if: HTTP error, timeout, social media domain, personal blog,
    news article, or Wikipedia. Do NOT flag as invalid just because the site is
    small or uses a free website builder — many legitimate nonprofits do.
    """
    # domains that are never legitimate or websites
    BLOCKED_DOMAINS = [
        "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
        "wikipedia.org", "youtube.com", "tiktok.com",
    ]

    try:
        # check for blocked domains before making any network request
        for domain in BLOCKED_DOMAINS:
            if domain in url:
                return UrlValidationResult(
                    is_valid=False,
                    reason=f"URL appears to be a {domain} page, not an organization website."
                )
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()         # will throw if 4xx or 5xx

        return UrlValidationResult(is_valid=True, reason="URL is reachable and appears valid.")

    except requests.exceptions.Timeout:
        return UrlValidationResult(is_valid=False, reason="URL timed out after 10 seconds.")
    except requests.exceptions.HTTPError as e:
        return UrlValidationResult(is_valid=False, reason=f"HTTP error: {str(e)}")
    except Exception as e:
        return UrlValidationResult(is_valid=False, reason=f"Could not reach URL: {str(e)}")