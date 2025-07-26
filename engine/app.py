from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from exam_analyzer import ExamAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global analyzer instance (will be initialized on startup)
analyzer = None


def initialize_analyzer():
    """
    Initialize the exam analyzer with data.
    Call this once when the server starts.
    """
    global analyzer

    try:
        logger.info("Initializing ExamAnalyzer...")
        analyzer = ExamAnalyzer()

        # Load data from JSON files
        data_folder = os.getenv('DATA_FOLDER', './exam_data')
        if not os.path.exists(data_folder):
            logger.warning(f"Data folder {data_folder} not found. Creating it...")
            os.makedirs(data_folder)
            logger.warning("Please add your JSON files to the exam_data folder and restart the server.")
            return False

        analyzer.load_json_files(data_folder)
        analyzer.build_embeddings()

        logger.info("ExamAnalyzer initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    global analyzer

    status = {
        'status': 'healthy',
        'analyzer_ready': analyzer is not None and analyzer.is_fitted
    }

    if analyzer:
        stats = analyzer.get_stats()
        status['total_questions'] = stats.get('total_questions', 0)

    return jsonify(status)


@app.route('/query', methods=['POST'])
def semantic_query():
    """
    POST /query - Perform semantic search on exam questions.

    Expected JSON payload:
    {
        "query": "string",
        "similarity_threshold": 0.5,  // optional, default 0.5
        "top_k": 20  // optional, default 20
    }

    Returns:
    {
        "query": "original query",
        "results": [...],
        "module_distribution": {...},
        "marks_distribution": {...},
        "total_matches": int
    }
    """
    global analyzer

    if not analyzer or not analyzer.is_fitted:
        return jsonify({
            'error': 'Analyzer not initialized. Please restart the server.',
            'status': 'not_ready'
        }), 503

    try:
        # Parse request data
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query'
            }), 400

        query = data['query'].strip()
        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400

        # Extract optional parameters
        similarity_threshold = data.get('similarity_threshold', 0.5)
        top_k = data.get('top_k', 20)

        # Validate parameters
        if not (0 <= similarity_threshold <= 1):
            return jsonify({
                'error': 'similarity_threshold must be between 0 and 1'
            }), 400

        if not (1 <= top_k <= 100):
            return jsonify({
                'error': 'top_k must be between 1 and 100'
            }), 400

        # Perform semantic search
        logger.info(f"Processing query: '{query}' (threshold: {similarity_threshold}, top_k: {top_k})")

        search_results = analyzer.semantic_search(
            query=query,
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )

        # Get distributions
        module_distribution = analyzer.get_module_distribution(search_results)
        marks_distribution = analyzer.get_marks_distribution(search_results)

        response = {
            'query': query,
            'results': search_results,
            'module_distribution': module_distribution,
            'marks_distribution': marks_distribution,
            'total_matches': len(search_results),
            'parameters': {
                'similarity_threshold': similarity_threshold,
                'top_k': top_k
            }
        }

        logger.info(f"Found {len(search_results)} matches for query: '{query}'")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/topics', methods=['GET'])
def frequent_topics():
    """
    GET /topics - Get frequently appearing topics across all exams.

    Query parameters:
    - similarity_threshold: float (default 0.7)
    - min_frequency: int (default 2)

    Returns:
    {
        "frequent_topics": [...],
        "total_clusters": int,
        "parameters": {...}
    }
    """
    global analyzer

    if not analyzer or not analyzer.is_fitted:
        return jsonify({
            'error': 'Analyzer not initialized. Please restart the server.',
            'status': 'not_ready'
        }), 503

    try:
        # Extract query parameters
        similarity_threshold = float(request.args.get('similarity_threshold', 0.7))
        min_frequency = int(request.args.get('min_frequency', 2))

        # Validate parameters
        if not (0 <= similarity_threshold <= 1):
            return jsonify({
                'error': 'similarity_threshold must be between 0 and 1'
            }), 400

        if min_frequency < 2:
            return jsonify({
                'error': 'min_frequency must be at least 2'
            }), 400

        logger.info(f"Detecting frequent topics (threshold: {similarity_threshold}, min_freq: {min_frequency})")

        # Detect frequent topics
        frequent_topics_list = analyzer.detect_frequent_topics(
            similarity_threshold=similarity_threshold,
            min_frequency=min_frequency
        )

        response = {
            'frequent_topics': frequent_topics_list,
            'total_clusters': len(frequent_topics_list),
            'parameters': {
                'similarity_threshold': similarity_threshold,
                'min_frequency': min_frequency
            }
        }

        logger.info(f"Found {len(frequent_topics_list)} frequent topic clusters")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error detecting frequent topics: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def dataset_stats():
    """
    GET /stats - Get general statistics about the loaded dataset.

    Returns:
    {
        "total_questions": int,
        "total_courses": int,
        "courses": {...},
        "modules": {...},
        "marks_distribution": {...},
        "years": {...}
    }
    """
    global analyzer

    if not analyzer:
        return jsonify({
            'error': 'Analyzer not initialized. Please restart the server.',
            'status': 'not_ready'
        }), 503

    try:
        stats = analyzer.get_stats()
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'POST /query',
            'GET /topics',
            'GET /stats'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end'
    }), 500


if __name__ == '__main__':
    # Initialize the analyzer on startup
    logger.info("Starting Exam Analysis API server...")

    success = initialize_analyzer()
    if not success:
        logger.warning("Starting server without fully initialized analyzer")

    # Run the Flask app
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))

    logger.info(f"Server starting on port {port} (debug={debug_mode})")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )