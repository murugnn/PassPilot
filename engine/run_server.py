#!/usr/bin/env python3
"""
Server runner script with automatic setup and validation.
This handles model downloading, data validation, and server startup.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")

    required_packages = [
        'flask',
        'flask_cors',
        'sentence_transformers',
        'sklearn',
        'numpy',
        'transformers',
        'torch'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} found")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} not found")

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

    logger.info("All dependencies satisfied!")
    return True


def setup_directories():
    """Create necessary directories."""
    logger.info("Setting up directories...")

    directories = [
        'exam_data',
        'models',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory ensured: {directory}")


def validate_data_files():
    """Validate JSON data files in exam_data directory."""
    logger.info("Validating data files...")

    data_dir = Path('exam_data')
    json_files = list(data_dir.glob('*.json'))

    if not json_files:
        logger.warning("No JSON files found in exam_data directory")
        create_sample_data()
        return False

    valid_files = 0
    total_questions = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            required_fields = ['courseCode', 'courseName', 'questions']
            if all(field in data for field in required_fields):
                questions = data.get('questions', [])
                if questions and all('question' in q for q in questions):
                    valid_files += 1
                    total_questions += len(questions)
                    logger.info(f"✅ Valid file: {json_file.name} ({len(questions)} questions)")
                else:
                    logger.warning(f"⚠️ File has invalid question structure: {json_file.name}")
            else:
                logger.warning(f"⚠️ File missing required fields: {json_file.name}")

        except Exception as e:
            logger.error(f"❌ Error reading {json_file.name}: {str(e)}")

    logger.info(f"Validation complete: {valid_files}/{len(json_files)} valid files, {total_questions} total questions")
    return valid_files > 0


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")

    sample_data = {
        "courseCode": "CST 206",
        "courseName": "OPERATING SYSTEMS",
        "month": "April",
        "year": 2025,
        "scheme": "2019 Scheme",
        "questions": [
            {
                "marks": "3",
                "module": "Module 1",
                "question": "Define user mode and kernel mode. Why are two modes of operations required for the system?"
            },
            {
                "marks": "3",
                "module": "Module 3",
                "question": "What is a resource-allocation graph? Explain with an example."
            },
            {
                "marks": "5",
                "module": "Module 2",
                "question": "Explain different process scheduling algorithms used in operating systems."
            },
            {
                "marks": "7",
                "module": "Module 4",
                "question": "Describe the concept of deadlock. Explain methods for deadlock prevention and avoidance."
            },
            {
                "marks": "10",
                "module": "Module 5",
                "question": "Compare and contrast different file system organization methods. Discuss their advantages and disadvantages."
            },
            {
                "marks": "5",
                "module": "Module 1",
                "question": "Explain the concept of system calls. How do they provide interface between user programs and operating system?"
            },
            {
                "marks": "7",
                "module": "Module 2",
                "question": "What is virtual memory? Explain the concept of paging and segmentation."
            },
            {
                "marks": "3",
                "module": "Module 3",
                "question": "Define semaphores. How are they used for process synchronization?"
            }
        ]
    }

    # Create additional sample files for better testing
    sample_data_2 = {
        "courseCode": "CST 204",
        "courseName": "DATA STRUCTURES",
        "month": "December",
        "year": 2024,
        "scheme": "2019 Scheme",
        "questions": [
            {
                "marks": "3",
                "module": "Module 1",
                "question": "Define data structure. Explain the difference between linear and non-linear data structures."
            },
            {
                "marks": "5",
                "module": "Module 2",
                "question": "Explain stack operations with implementation using arrays."
            },
            {
                "marks": "7",
                "module": "Module 3",
                "question": "Write an algorithm for binary search. Analyze its time complexity."
            },
            {
                "marks": "10",
                "module": "Module 4",
                "question": "Explain different tree traversal methods with examples."
            },
            {
                "marks": "5",
                "module": "Module 2",
                "question": "What is a queue? Implement queue using linked list."
            }
        ]
    }

    # Write sample files
    try:
        with open('exam_data/sample_os_2025.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)

        with open('exam_data/sample_ds_2024.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data_2, f, indent=2, ensure_ascii=False)

        logger.info("✅ Sample data files created successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {str(e)}")
        return False


def download_model():
    """Download the sentence transformer model if not already cached."""
    logger.info("Checking/downloading sentence transformer model...")

    try:
        from sentence_transformers import SentenceTransformer

        # This will download the model if not already cached
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Model ready")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False


def start_server():
    """Start the Flask server."""
    logger.info("Starting the Flask server...")

    try:
        # Import and run the Flask app
        from app import app, initialize_analyzer

        # Initialize analyzer
        logger.info("Initializing analyzer...")
        if not initialize_analyzer():
            logger.error("Failed to initialize analyzer")
            return False

        # Get configuration
        debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        port = int(os.getenv('PORT', 5000))
        host = os.getenv('HOST', '0.0.0.0')

        logger.info(f"Starting server on {host}:{port} (debug={debug_mode})")

        # Start the server
        app.run(
            host=host,
            port=port,
            debug=debug_mode
        )

        return True

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        return False


def main():
    """Main setup and run function."""
    print("🚀 Exam Analysis Engine - Server Setup")
    print("=" * 50)

    # Step 1: Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Step 2: Setup directories
    setup_directories()

    # Step 3: Validate data files
    if not validate_data_files():
        logger.warning("No valid data files found. Sample data created.")
        logger.info("Add your JSON files to the exam_data directory for production use.")

    # Step 4: Download/check model
    if not download_model():
        sys.exit(1)

    print("\n✅ Setup complete! Starting server...")
    print("=" * 50)

    # Step 5: Start server
    if not start_server():
        sys.exit(1)


if __name__ == "__main__":
    main()