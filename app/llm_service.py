"""
LLM Service Module - Provides a unified interface for interacting with OpenAI's LLM APIs.

This module centralizes OpenAI API interactions across the application, handling:
- Client initialization and configuration
- API calls with standardized error handling
- Consistent logging
- JSON response parsing and cleaning
- Fallback mechanisms for API failures

Usage Examples:
-------------
# Basic text generation:
llm_service = LLMService()
response = llm_service.generate_text(
    prompt="Tell me about Paris",
    system_message="You are a helpful travel guide."
)

# JSON generation with error handling:
llm_service = LLMService()
data = llm_service.generate_json(
    prompt="List the top 3 attractions in Paris",
    system_message="You are a data provider that only returns valid JSON.",
    fallback_response={"attractions": []}
)
"""

import logging
import json
import re
import os
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)
openai_logger = logging.getLogger('openai.prompt')

class LLMService:
    """
    A service class that provides a unified interface for interacting with OpenAI's LLM APIs.
    
    This class encapsulates the common patterns for making requests to OpenAI,
    handling responses, and dealing with errors. It provides methods for generating
    both free-form text and structured JSON responses.
    
    Attributes:
        client: The OpenAI client instance
        default_model: The default model to use for requests
        default_temperature: The default temperature setting for generation
        default_max_tokens: The default maximum number of tokens for generation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLMService with optional API key.
        
        Args:
            api_key: Optional API key for OpenAI. If not provided, it will use the 
                    OPENAI_API_KEY environment variable.
        """
        # Set API key as environment variable if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
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
            prompt: The user prompt to send to the API
            system_message: Optional system message to set the behavior of the model
            model: The model to use (defaults to self.default_model)
            temperature: The temperature for generation (defaults to self.default_temperature)
            max_tokens: Maximum tokens to generate (defaults to self.default_max_tokens)
            log_prompt: Whether to log the prompt (defaults to True)
            
        Returns:
            The generated text as a string, or None if an error occurred
            
        Example:
            service = LLMService()
            response = service.generate_text(
                prompt="Explain quantum computing",
                system_message="You are a physics professor."
            )
        """
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Log using the special OpenAI logger if enabled
        if log_prompt:
            openai_logger.info(f"Generating text with {model} (temp={temperature:.1f}, max_tokens={max_tokens})")
            
            if system_message:
                openai_logger.info(f"System message: {system_message}")
                
            openai_logger.info(f"User message: {prompt}")
        
        try:
            logger.info(f"Calling OpenAI API with model {model}")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # Log a truncated version of the response
                if log_prompt:
                    openai_logger.info(f"Response: {content}")
                
                return content
            else:
                logger.warning("OpenAI API returned empty response")
                return None
                
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {str(e)}")
            return None
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return None
    
    def generate_json(self, 
                     prompt: str, 
                     system_message: Optional[str] = None,
                     fallback_response: Optional[Dict[str, Any]] = None,
                     clean_json: bool = True,
                     model: Optional[str] = None,
                     temperature: Optional[float] = None, 
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate text and parse it as JSON.
        
        This method adds specific instructions to the system message to ensure the
        model returns properly formatted JSON. It also includes JSON parsing and cleaning.
        
        Args:
            prompt: The user prompt to send to the API
            system_message: Optional base system message
            fallback_response: Response to return if API call or JSON parsing fails
            clean_json: Whether to attempt JSON cleaning if parsing fails (default: True)
            model: The model to use (defaults to self.default_model)
            temperature: The temperature setting (defaults to self.default_temperature)
            max_tokens: Maximum tokens to generate (defaults to self.default_max_tokens)
            
        Returns:
            The parsed JSON as a Python dict/list, or fallback_response if an error occurred
            
        Example:
            service = LLMService()
            data = service.generate_json(
                prompt="List the capital cities of North America",
                fallback_response={"cities": []}
            )
        """
        # Set default fallback response if none provided
        if fallback_response is None:
            fallback_response = {"error": "Failed to generate response"}
        
        # Add JSON formatting instructions to system message
        json_system_msg = "You must respond with valid JSON only. No explanations or text outside of the JSON object."
        if system_message:
            system_message = f"{system_message}\n\n{json_system_msg}"
        else:
            system_message = json_system_msg
            
        # Add JSON formatting hint to the prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. The response must have a valid JSON structure."
        
        response_text = self.generate_text(
            prompt=json_prompt,
            system_message=system_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if not response_text:
            logger.warning("Failed to generate JSON response, using fallback")
            return fallback_response
            
        # Log first 100 chars of response for debugging
        logger.debug(f"Response from LLM (first 100 chars): {response_text[:100]}...")
        
        # Try parsing the JSON
        parsed_json = self._parse_json(response_text, fallback_response, clean_json)
        
        # Check if parsing returned the fallback
        if parsed_json == fallback_response:
            logger.warning("JSON parsing failed, using fallback response")
            # Try an alternative parsing strategy as a last resort
            try:
                # Look for anything that looks like a JSON object or array
                json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                if json_match:
                    potential_json = json_match.group(1).strip()
                    logger.debug(f"Attempting to parse extracted JSON: {potential_json[:100]}...")
                    return json.loads(potential_json)
            except Exception as e:
                logger.debug(f"Alternative parsing failed: {e}")
        
        return parsed_json
    
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
    
    def _parse_json(self, json_text: str, fallback_response: Optional[Dict[str, Any]] = None, clean: bool = True) -> Dict[str, Any]:
        """
        Parse text as JSON, with optional cleaning for common issues.
        
        Args:
            json_text: The JSON string to parse
            fallback_response: Response to return if parsing fails
            clean: Whether to attempt cleaning if parsing fails (default: True)
            
        Returns:
            The parsed JSON as a Python dict/list, or fallback_response if parsing failed
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
        
        # First attempt direct parsing without cleaning
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            # If direct parsing fails, try with cleaning if enabled
            if not clean:
                logger.error(f"JSON parsing error (no cleaning attempted): {e}")
                logger.error(f"Problematic JSON text: {json_text}")
                return fallback_response
                
            # Try advanced cleaning methods
            logger.debug(f"Initial JSON parsing failed, attempting cleaning...")
            
            # Fix trailing commas which are invalid in JSON but common in JavaScript
            json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)
            
            # Fix missing quote marks around keys
            json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)
            
            # Fix single quotes to double quotes if needed
            if "'" in json_text and '"' not in json_text:
                json_text = json_text.replace("'", '"')
                
            # Remove any JavaScript comments
            json_text = re.sub(r'//.*?\n|/\*.*?\*/', '', json_text, flags=re.DOTALL)
            
            # Try second parsing attempt after cleaning
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error after cleaning: {e}")
                logger.error(f"Problematic JSON text after cleaning: {json_text}")
                return fallback_response 