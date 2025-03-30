# LLM Service Refactoring

This document describes the refactoring of OpenAI/LLM usage in the TV Recommender application to improve code quality, reduce duplication, and enhance maintainability.

## Overview

The application previously had duplicate code for OpenAI interactions in multiple classes, primarily in `Recommender` and `ProfileBuilder`. This refactoring centralizes all LLM interactions into a dedicated `LLMService` class, which provides standardized methods for:

1. Initializing the OpenAI client
2. Generating text responses
3. Generating and parsing JSON responses
4. Creating embeddings
5. Error handling and logging

## Changes Made

### 1. New LLMService Class

Created a new service class (`app/llm_service.py`) that encapsulates all OpenAI interactions with the following features:

- **Consistent Initialization**: Standardized API key handling and client setup
- **Text Generation**: Unified method for generating text with customizable parameters
- **JSON Generation**: Enhanced method that handles JSON parsing and cleaning
- **Embeddings**: Support for creating text embeddings
- **Error Handling**: Comprehensive error handling with proper logging
- **JSON Parsing**: Robust JSON parsing with cleanup for common issues

### 2. Refactored Recommender Class

The `Recommender` class was modified to:

- Use `LLMService` instead of direct OpenAI interactions
- Remove duplicate code for API calls and JSON parsing
- Maintain identical functionality while reducing code complexity

### 3. Refactored ProfileBuilder Class

The `ProfileBuilder` class was modified to:

- Use `LLMService` for clustering operations
- Simplify the taste cluster generation process
- Maintain backward compatibility with the existing codebase

## Benefits

This refactoring provides several key benefits:

1. **Reduced Code Duplication**: Eliminated approximately 150 lines of duplicated code
2. **Improved Error Handling**: Standardized error handling across all LLM interactions
3. **Better Maintainability**: Changes to LLM behavior can now be made in one place
4. **Enhanced Logging**: Consistent logging for LLM operations
5. **Future-Proofing**: Easier to update or swap out LLM providers in the future
6. **Simplified Testing**: LLM interactions can be mocked or tested in isolation

## Usage

To use the new `LLMService` in your code:

```python
from app.llm_service import LLMService

# Initialize the service
llm_service = LLMService(api_key="your_openai_api_key")  # Or omit for env var

# Generate text
response = llm_service.generate_text(
    prompt="Your prompt here",
    system_message="Optional system message",
    model="gpt-4o-mini",  # Optional, defaults to gpt-4o-mini
    temperature=0.7,      # Optional, defaults to 0.7
    max_tokens=1500       # Optional, defaults to 1500
)

# Generate and parse JSON
json_response = llm_service.generate_json(
    prompt="Generate JSON data for...",
    system_message="You will respond with valid JSON only.",
    fallback_response={"error": "Failed to generate"}  # Optional fallback
)

# Create embeddings
embeddings = llm_service.create_embeddings("Text to embed")
# Or for multiple texts:
embeddings = llm_service.create_embeddings(["Text 1", "Text 2", "Text 3"])
```

## Future Improvements

Potential future enhancements:

1. Add support for streaming responses
2. Implement retry logic for transient errors
3. Add caching for common requests
4. Add support for additional LLM providers
5. Create a factory pattern for different types of prompts 