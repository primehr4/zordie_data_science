from typing import Dict, Any, List, Optional
import logging
import time
import requests
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """
    Base class for all platform-specific data extractors.
    Defines the common interface and utility methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    @abstractmethod
    def extract_data(self, url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data from a platform-specific URL
        
        Args:
            url: The URL to extract data from
            metadata: Optional metadata about the URL from the link crawler
            
        Returns:
            Dictionary containing extracted data
        """
        pass
    
    def _make_request(self, url: str, headers: Dict[str, str] = None, 
                     params: Dict[str, Any] = None, timeout: int = 10) -> Optional[requests.Response]:
        """
        Make an HTTP request with error handling and rate limiting
        
        Args:
            url: URL to request
            headers: Optional additional headers
            params: Optional query parameters
            timeout: Request timeout in seconds
            
        Returns:
            Response object or None if request failed
        """
        try:
            # Apply rate limiting
            time.sleep(self.config.get("rate_limit_delay", 1))
            
            # Merge headers
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Make request
            response = self.session.get(
                url, 
                headers=request_headers,
                params=params,
                timeout=timeout,
                verify=self.config.get("verify_ssl", True)
            )
            
            # Check for successful response
            if response.status_code == 200:
                return response
            else:
                self.logger.warning(f"Request to {url} failed with status code {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making request to {url}: {e}")
            return None
    
    def _extract_from_api(self, api_url: str, headers: Dict[str, str] = None, 
                         params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Extract data from a JSON API
        
        Args:
            api_url: API URL to request
            headers: Optional additional headers
            params: Optional query parameters
            
        Returns:
            Parsed JSON response or None if request failed
        """
        response = self._make_request(api_url, headers, params)
        
        if response:
            try:
                return response.json()
            except Exception as e:
                self.logger.error(f"Error parsing JSON from {api_url}: {e}")
                return None
        
        return None
    
    def _safe_get(self, data: Dict[str, Any], *keys, default=None):
        """
        Safely get a nested value from a dictionary
        
        Args:
            data: Dictionary to extract from
            *keys: Keys to traverse
            default: Default value if key doesn't exist
            
        Returns:
            Value at the specified keys or default
        """
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current