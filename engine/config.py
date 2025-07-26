"""
Configuration file for the Exam Analysis Engine.
Adjust these settings based on your deployment needs.
"""

# Model Configuration
MODEL_CONFIG = {
    'name': 'all-MiniLM-L6-v2',  # Sentence transformer model
    'cache_folder': './models',   # Where to cache the model
}

# Search Configuration
SEARCH_CONFIG = {
    'default_similarity_threshold': 0.5,  # Default similarity threshold
    'max_similarity_threshold': 0.95,     # Maximum allowed threshold
    'min_similarity_threshold': 0.1,      # Minimum allowed threshold
    'default_top_k': 20,                   # Default number of results
    'max_top_k': 100,                      # Maximum allowed results
}

# Topic Detection Configuration
TOPIC_CONFIG = {
    'default_similarity_threshold': 0.7,  # Default for topic clustering
    'default_min_frequency': 2,           # Minimum frequency for topics
    'max_clusters': 50,                   # Maximum topic clusters to return
}

# Data Configuration
DATA_CONFIG = {
    'folder_path': './exam_data',         # Default data folder
    'supported_formats': ['.json'],       # Supported file formats
    'encoding': 'utf-8',                  # File encoding
}

# API Configuration
API_CONFIG = {
    'cors_origins': ['*'],                # CORS allowed origins
    'rate_limit': '100/hour',            # Rate limiting (if needed)
    'max_query_length': 500,             # Maximum query length
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'exam_analyzer.log'
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'batch_size': 32,                    # Batch size for embeddings
    'max_questions': 10000,              # Maximum questions to process
    'embedding_dimension': 384,          # Expected embedding dimension
}

# Validation schemas for API requests
VALIDATION_SCHEMAS = {
    'query_request': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'minLength': 1, 'maxLength': 500},
            'similarity_threshold': {'type': 'number', 'minimum': 0.1, 'maximum': 0.95},
            'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 100}
        },
        'required': ['query']
    }
}

def get_model_config():
    """Get model configuration."""
    return MODEL_CONFIG.copy()

def get_search_config():
    """Get search configuration."""
    return SEARCH_CONFIG.copy()

def get_topic_config():
    """Get topic detection configuration."""
    return TOPIC_CONFIG.copy()

def get_data_config():
    """Get data configuration."""
    return DATA_CONFIG.copy()

def get_api_config():
    """Get API configuration."""
    return API_CONFIG.copy()

def validate_query_params(similarity_threshold, top_k):
    """
    Validate query parameters.

    Args:
        similarity_threshold: Similarity threshold value
        top_k: Number of results to return

    Returns:
        tuple: (is_valid, error_message)
    """
    if not (SEARCH_CONFIG['min_similarity_threshold'] <=
            similarity_threshold <=
            SEARCH_CONFIG['max_similarity_threshold']):
        return False, f"Similarity threshold must be between {SEARCH_CONFIG['min_similarity_threshold']} and {SEARCH_CONFIG['max_similarity_threshold']}"

    if not (1 <= top_k <= SEARCH_CONFIG['max_top_k']):
        return False, f"top_k must be between 1 and {SEARCH_CONFIG['max_top_k']}"

    return True, None

def validate_topic_params(similarity_threshold, min_frequency):
    """
    Validate topic detection parameters.

    Args:
        similarity_threshold: Similarity threshold for clustering
        min_frequency: Minimum frequency for topics

    Returns:
        tuple: (is_valid, error_message)
    """
    if not (0.5 <= similarity_threshold <= 0.95):
        return False, "Topic similarity threshold must be between 0.5 and 0.95"

    if min_frequency < 2:
        return False, "Minimum frequency must be at least 2"

    return True, None