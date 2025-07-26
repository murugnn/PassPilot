"""
Test script for the Exam Analysis API.
Run this to verify your API is working correctly.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL (adjust if needed)
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")

    try:
        response = requests.get(f"{BASE_URL}/health")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint."""
    print("\n📊 Testing stats endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/stats")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats retrieved successfully")
            print(f"   Total questions: {data.get('total_questions', 'N/A')}")
            print(f"   Total courses: {data.get('total_courses', 'N/A')}")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
        return False

def test_query_endpoint():
    """Test the semantic query endpoint."""
    print("\n🔍 Testing query endpoint...")

    test_queries = [
        {
            "query": "What is operating system process scheduling?",
            "similarity_threshold": 0.5,
            "top_k": 5
        },
        {
            "query": "Define deadlock in operating systems",
            "similarity_threshold": 0.4,
            "top_k": 10
        }
    ]

    for i, test_query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: '{test_query['query']}'")

        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json=test_query,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Query successful")
                print(f"      Matches found: {data.get('total_matches', 0)}")
                print(f"      Modules: {list(data.get('module_distribution', {}).get('modules', {}).keys())}")

                # Show top result if available
                if data.get('results'):
                    top_result = data['results'][0]
                    print(f"      Top match ({top_result.get('similarity_percentage', 0)}%): {top_result.get('question', '')[:100]}...")

            else:
                print(f"   ❌ Query failed: {response.status_code}")
                print(f"      Response: {response.text}")

        except Exception as e:
            print(f"   ❌ Query error: {str(e)}")

def test_topics_endpoint():
    """Test the frequent topics endpoint with updated features."""
    print("\n📈 Testing topics endpoint...")

    try:
        # Test with default parameters
        response = requests.get(f"{BASE_URL}/topics")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Topics retrieved successfully")
            print(f"   Total clusters: {data.get('total_clusters', 0)}")

            # Show top topics if available
            if data.get('frequent_topics'):
                print("   Top frequent topics:")

                for i, topic in enumerate(data['frequent_topics'][:3], 1):
                    keywords = ', '.join(topic.get('topic_keywords', []))
                    cohesion = topic.get('cohesion_score', None)
                    freq = topic.get('frequency', 0)
                    examples = topic.get('examples', [])

                    print(f"      {i}. Frequency: {freq}")
                    print(f"         Keywords: {keywords}")
                    print(f"         Cohesion Score: {round(cohesion, 3) if cohesion is not None else 'N/A'}")

                    if examples:
                        print(f"         Example Q: {examples[0].get('question', '')[:80]}...")
                    else:
                        print("         ⚠️ No example questions found.")

                    # Validation checks
                    if not keywords or not isinstance(keywords, str) and not isinstance(keywords, list):
                        print(f"         ❌ Invalid topic keywords format")
                    if cohesion is not None and not (0.0 <= cohesion <= 1.0):
                        print(f"         ❌ Invalid cohesion score: {cohesion}")

            else:
                print("   ❌ No topic clusters returned")

            return True
        else:
            print(f"❌ Topics failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Topics error: {str(e)}")
        return False


def test_error_handling():
    """Test error handling scenarios."""
    print("\n⚠️ Testing error handling...")

    # Test invalid query (empty)
    print("   Testing empty query...")
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": ""},
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 400:
            print("   ✅ Empty query handled correctly")
        else:
            print(f"   ❌ Empty query not handled: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Empty query test error: {str(e)}")

    # Test invalid similarity threshold
    print("   Testing invalid similarity threshold...")
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": "test", "similarity_threshold": 1.5},
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 400:
            print("   ✅ Invalid threshold handled correctly")
        else:
            print(f"   ❌ Invalid threshold not handled: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Invalid threshold test error: {str(e)}")

    # Test invalid endpoint
    print("   Testing invalid endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/invalid")

        if response.status_code == 404:
            print("   ✅ Invalid endpoint handled correctly")
        else:
            print(f"   ❌ Invalid endpoint not handled: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Invalid endpoint test error: {str(e)}")

def run_performance_test():
    """Run a basic performance test."""
    print("\n⚡ Running performance test...")

    queries = [
        "operating system process",
        "memory management",
        "file system",
        "deadlock detection",
        "cpu scheduling"
    ]

    start_time = time.time()
    successful_queries = 0

    for query in queries:
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": query, "top_k": 5},
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                successful_queries += 1

        except Exception as e:
            print(f"   ❌ Performance test query failed: {str(e)}")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"   Performance Results:")
    print(f"      Queries processed: {successful_queries}/{len(queries)}")
    print(f"      Total time: {total_time:.2f} seconds")
    print(f"      Average per query: {total_time/len(queries):.2f} seconds")

def create_sample_data():
    """Create sample JSON data for testing if none exists."""
    print("\n📁 Creating sample data for testing...")

    import os

    # Create exam_data directory if it doesn't exist
    os.makedirs('exam_data', exist_ok=True)

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
                "question": "Define user mode and kernel mode. Why the two modes of operations are required for the system?"
            },
            {
                "marks": "3",
                "module": "Module 3",
                "question": "What is a resource-allocation graph?"
            },
            {
                "marks": "5",
                "module": "Module 2",
                "question": "Explain process scheduling algorithms in operating systems."
            },
            {
                "marks": "7",
                "module": "Module 4",
                "question": "Describe the concept of deadlock and methods for deadlock prevention."
            },
            {
                "marks": "10",
                "module": "Module 5",
                "question": "Compare and contrast different file system organization methods."
            }
        ]
    }

    # Write sample data to file
    with open('exam_data/sample_exam_2025.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)

    print("   ✅ Sample data created in exam_data/sample_exam_2025.json")

def main():
    """Run all tests."""
    print("🚀 Starting Exam Analysis API Tests")
    print("=" * 50)

    # Check if sample data exists, create if not
    if not os.path.exists('exam_data') or not os.listdir('exam_data'):
        create_sample_data()
        print("   Please restart your API server to load the sample data.")
        return

    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)

    # Run tests
    tests_passed = 0
    total_tests = 4

    if test_health_check():
        tests_passed += 1

    if test_stats_endpoint():
        tests_passed += 1

    test_query_endpoint()  # This runs multiple sub-tests
    tests_passed += 1

    if test_topics_endpoint():
        tests_passed += 1

    # Additional tests
    test_error_handling()
    run_performance_test()

    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Summary: {tests_passed}/{total_tests} main tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    import os
    main()