"""
Module for interacting with Large Language Models (specifically OpenAI).
"""

import logging
import json
import re
import os
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with OpenAI's language models.
    This class centralizes all OpenAI API calls, providing consistent
    error handling, response parsing, and configuration.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM service.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
        """
        # Set API key as environment variable if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Default configuration
        self.default_model = "gpt-4o-mini"
        self.default_temperature = 0.7
        self.default_max_tokens = 1500
        self.embedding_model = "text-embedding-3-small"
    
    def generate_text(self, 
                     prompt: str, 
                     system_message: Optional[str] = None, 
                     model: Optional[str] = None,
                     temperature: Optional[float] = None, 
                     max_tokens: Optional[int] = None,
                     log_prompt: bool = True) -> str:
        """
        Generate text using the OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message for role setup
            model: LLM model to use (defaults to default_model)
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            log_prompt: Whether to log the prompt for debugging
            
        Returns:
            The generated text response
        """
        # Log the prompt if enabled
        if log_prompt:
            logger.info(f"LLM Prompt ({len(prompt)} chars):")
            if len(prompt) > 500:
                logger.info(f"{prompt[:250]}...{prompt[-250:]}")
            else:
                logger.info(prompt)
        
        # Set up the messages for the API call
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Use provided values or defaults
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        try:
            # Log the API call details
            logger.info(f"Calling OpenAI API with: model={model}, temperature={temperature}, max_tokens={max_tokens}")
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract and return the response content
            result = response.choices[0].message.content.strip()
            logger.info(f"OpenAI API call completed successfully - Response length: {len(result)} chars")
            
            return result
            
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
            
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    def generate_json(self, 
                     prompt: str, 
                     system_message: Optional[str] = None,
                     fallback_response: Optional[Dict[str, Any]] = None,
                     clean_json: bool = True,
                     model: Optional[str] = None,
                     temperature: Optional[float] = None, 
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate text and parse as JSON.
        
        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message for role setup
            fallback_response: Response to return if JSON parsing fails
            clean_json: Whether to attempt cleaning common JSON issues
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens to generate
            
        Returns:
            The parsed JSON response as a dictionary
        """
        try:
            # Generate the text response
            response_text = self.generate_text(
                prompt=prompt,
                system_message=system_message or "You will respond with valid JSON only.",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse the response as JSON
            return self._parse_json(response_text, clean=clean_json)
            
        except Exception as e:
            logger.error(f"Error generating JSON with OpenAI: {e}")
            # Return fallback response if provided, otherwise re-raise
            if fallback_response is not None:
                logger.info(f"Using fallback JSON response")
                return fallback_response
            raise
    
    def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Create embeddings for the given texts.
        
        Args:
            texts: A single text string or list of text strings
            
        Returns:
            List of embedding vectors
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts using {self.embedding_model}")
            
            # Make the API call
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            
            # Extract and return the embeddings
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def _parse_json(self, json_text: str, clean: bool = True) -> Dict[str, Any]:
        """
        Parse a string as JSON, with optional cleaning for common issues.
        
        Args:
            json_text: The JSON string to parse
            clean: Whether to attempt cleaning common JSON issues
            
        Returns:
            Parsed JSON as a dictionary
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        # Extract JSON from markdown code blocks if present
        if json_text.startswith("```json"):
            json_text = json_text.split("```json", 1)[1]
            if "```" in json_text:
                json_text = json_text.split("```", 1)[0]
        elif json_text.startswith("```"):
            json_text = json_text.split("```", 1)[1]
            if "```" in json_text:
                json_text = json_text.split("```", 1)[0]
                
        # Basic cleanup
        json_text = json_text.strip()
        
        # Optional advanced cleaning for common JSON issues
        if clean:
            # Fix trailing commas which are invalid in JSON but common in JavaScript
            json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)
            
            # Fix missing quote marks around keys
            json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)
            
            # Fix single quotes to double quotes if needed
            if "'" in json_text and '"' not in json_text:
                json_text = json_text.replace("'", '"')
        
        # Parse JSON
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Problematic JSON text: {json_text}")
            raise 