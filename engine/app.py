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


# In app.py, add this new endpoint

@app.route('/pass-simulation', methods=['POST'])
def pass_simulation():
    """
    POST /pass-simulation - Runs a Monte Carlo simulation for pass probability.

    Expected JSON payload (same as /pass-strategy):
    {
        "studied_topics": ["Topic A", "Topic B"],
        "internal_marks": 32,
        "external_pass_threshold": 40,
        "overall_pass_threshold": 75
    }
    """
    global analyzer

    if not analyzer or not analyzer.is_fitted:
        return jsonify({'error': 'Analyzer not initialized'}), 503

    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'Invalid JSON payload'}), 400

        # --- Input validation (same as before) ---
        studied_topics = data.get('studied_topics', [])
        internal_marks = data.get('internal_marks', 0)
        external_pass_threshold = data.get('external_pass_threshold', 0)
        overall_pass_threshold = data.get('overall_pass_threshold', 0)

        # --- Target calculation (same as before) ---
        marks_needed_for_overall = overall_pass_threshold - internal_marks
        target_external_marks = max(external_pass_threshold, marks_needed_for_overall)

        logger.info(f"Running simulation for target: {target_external_marks} marks.")

        # --- Call the NEW simulation function ---
        simulation_results = analyzer.run_pass_simulation(
            studied_topics=studied_topics,
            target_marks=target_external_marks
        )

        # Add context to the response
        simulation_results['inputs'] = {
            'internal_marks': internal_marks,
            'studied_topics': studied_topics,
            'calculated_target_marks': target_external_marks
        }

        return jsonify(simulation_results)

    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


## The old @app.route('/topics') should be replaced with this one.
@app.route('/topics', methods=['GET'])
def analyze_topics():
    """
    GET /topics - Get an analysis of pre-labeled topics from the exam data.

    Query parameters:
    - min_frequency: int (default 2) - The minimum number of questions a topic must have to be included.

    Returns:
    {
        "topics": [...],
        "total_topics": int,
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
        min_frequency = int(request.args.get('min_frequency', 2))
        if min_frequency < 1:
            return jsonify({'error': 'min_frequency must be at least 1'}), 400

        logger.info(f"Analyzing topics with min_freq: {min_frequency}")

        # Call the new, refactored analysis function
        topic_list = analyzer.get_topic_analysis(min_frequency=min_frequency)

        response = {
            'topics': topic_list,
            'total_topics': len(topic_list),
            'parameters': {
                'min_frequency': min_frequency
            }
        }

        logger.info(f"Returning {len(topic_list)} topics.")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error analyzing topics: {str(e)}", exc_info=True)
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
            'GET /stats',
            'POST /pass-strategy',
            'POST /pass-simulation'  # <-- Add this line
        ]
    }), 404


# In app.py, add this new endpoint

@app.route('/pass-strategy', methods=['POST'])
def pass_strategy():
    """
    POST /pass-strategy - Generates a targeted study plan to pass an exam.

    Expected JSON payload:
    {
        "studied_topics": ["Topic A", "Topic B"], // list of strings
        "internal_marks": 32,                    // number
        "external_exam_total": 100,              // number
        "external_pass_threshold": 40,           // number (min marks for the paper)
        "overall_pass_threshold": 75             // number (min combined marks)
    }

    Returns:
    A personalized study strategy.
    """
    global analyzer

    if not analyzer or not analyzer.is_fitted:
        return jsonify({
            'error': 'Analyzer not initialized. Please restart the server.',
            'status': 'not_ready'
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        # --- Validate inputs ---
        studied_topics = data.get('studied_topics')
        internal_marks = data.get('internal_marks')
        external_pass_threshold = data.get('external_pass_threshold')
        overall_pass_threshold = data.get('overall_pass_threshold')

        if not all(isinstance(val, list) for val in [studied_topics]):
            return jsonify({'error': '`studied_topics` must be a list'}), 400

        if not all(isinstance(val, (int, float)) for val in
                   [internal_marks, external_pass_threshold, overall_pass_threshold]):
            return jsonify({'error': 'Marks and thresholds must be numbers'}), 400

        # --- Core Logic: Determine the actual target score for the external exam ---
        # The student needs to satisfy TWO conditions:
        # 1. Get at least 40 on the external paper.
        # 2. Get a combined score of at least 75.
        # This means the required score is the HIGHER of the two thresholds.

        marks_needed_for_overall = overall_pass_threshold - internal_marks

        # The target is the maximum of the standalone threshold and the one needed for the overall total.
        target_external_marks = max(external_pass_threshold, marks_needed_for_overall)

        logger.info(
            f"Calculating pass strategy. Studied: {len(studied_topics)} topics. Target: {target_external_marks} marks.")

        # --- Call the analyzer to get the strategy ---
        strategy = analyzer.get_pass_strategy(
            studied_topics=studied_topics,
            min_pass_marks=target_external_marks
        )

        # Add context to the response
        strategy['inputs'] = {
            'internal_marks': internal_marks,
            'studied_topics': studied_topics,
            'calculated_target_marks': target_external_marks
        }

        return jsonify(strategy)

    except Exception as e:
        logger.error(f"Error calculating pass strategy: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


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