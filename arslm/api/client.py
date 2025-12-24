"""
API Client for ARSLM.

Client for interacting with ARSLM REST API.
"""

import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class ARSLMClient:
    """
    Client for ARSLM API.
    
    Example:
        >>> client = ARSLMClient("http://localhost:8000")
        >>> response = client.chat("Hello!", session_id="user123")
        >>> print(response['text'])
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize ARSLM client.
        
        Args:
            base_url: Base URL of ARSLM API
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        self.session = requests.Session()
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'
        self.session.headers['Content-Type'] = 'application/json'
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            raise ARSLMClientError(f"API request failed: {str(e)}")
    
    def chat(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 1.0,
        max_length: int = 100
    ) -> Dict[str, Any]:
        """
        Send chat message.
        
        Args:
            message: User message
            session_id: Session identifier
            context: Additional context
            temperature: Sampling temperature
            max_length: Maximum response length
            
        Returns:
            API response with 'text' field
        """
        payload = {
            'message': message,
            'session_id': session_id,
            'temperature': temperature,
            'max_length': max_length
        }
        
        if context:
            payload['context'] = context
        
        return self._request('POST', '/api/v1/chat', json=payload)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_return_sequences: Number of sequences
            
        Returns:
            Generated text(s)
        """
        payload = {
            'prompt': prompt,
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'num_return_sequences': num_return_sequences
        }
        
        return self._request('POST', '/api/v1/generate', json=payload)
    
    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get conversation history.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages
            
        Returns:
            Conversation history
        """
        params = {'session_id': session_id}
        if limit:
            params['limit'] = limit
        
        return self._request('GET', '/api/v1/history', params=params)
    
    def clear_history(self, session_id: str) -> Dict[str, Any]:
        """
        Clear conversation history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success confirmation
        """
        return self._request(
            'DELETE',
            f'/api/v1/history/{session_id}'
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        return self._request('GET', '/health')
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model metadata
        """
        return self._request('GET', '/api/v1/model/info')


class ARSLMClientError(Exception):
    """Exception raised for ARSLM client errors."""
    pass


# Async client
try:
    import aiohttp
    import asyncio
    
    class AsyncARSLMClient:
        """Async version of ARSLM client."""
        
        def __init__(
            self,
            base_url: str = "http://localhost:8000",
            api_key: Optional[str] = None,
            timeout: int = 30
        ):
            self.base_url = base_url.rstrip('/')
            self.api_key = api_key
            self.timeout = aiohttp.ClientTimeout(total=timeout)
            self.headers = {'Content-Type': 'application/json'}
            
            if api_key:
                self.headers['Authorization'] = f'Bearer {api_key}'
        
        async def _request(
            self,
            method: str,
            endpoint: str,
            **kwargs
        ) -> Dict[str, Any]:
            """Make async HTTP request."""
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            ) as session:
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
        
        async def chat(
            self,
            message: str,
            session_id: str,
            **kwargs
        ) -> Dict[str, Any]:
            """Async chat."""
            payload = {
                'message': message,
                'session_id': session_id,
                **kwargs
            }
            return await self._request('POST', '/api/v1/chat', json=payload)
        
        async def generate(
            self,
            prompt: str,
            **kwargs
        ) -> Dict[str, Any]:
            """Async generation."""
            payload = {'prompt': prompt, **kwargs}
            return await self._request('POST', '/api/v1/generate', json=payload)

except ImportError:
    # aiohttp not available
    AsyncARSLMClient = None