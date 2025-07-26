import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExamAnalyzer:
    """
    Intelligent exam analysis engine with semantic search capabilities.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the analyzer with a sentence transformer model.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Storage for processed data
        self.questions_data: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_fitted = False

    def load_json_files(self, folder_path: str) -> None:
        """
        Load and process all JSON files from the specified folder.

        Args:
            folder_path: Path to folder containing JSON files
        """
        logger.info(f"Loading JSON files from: {folder_path}")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

        if not json_files:
            raise ValueError(f"No JSON files found in {folder_path}")

        logger.info(f"Found {len(json_files)} JSON files")

        for filename in json_files:
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    exam_data = json.load(f)
                    self._process_exam_data(exam_data, filename)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue

        logger.info(f"Successfully loaded {len(self.questions_data)} questions")

    def _process_exam_data(self, exam_data: Dict[str, Any], filename: str) -> None:
        """
        Process individual exam JSON data and extract questions.

        Args:
            exam_data: Parsed JSON exam data
            filename: Source filename for tracking
        """
        questions = exam_data.get('questions', [])

        for idx, question_obj in enumerate(questions):
            processed_question = {
                'id': f"{filename}_{idx}",
                'question': question_obj.get('question', '').strip(),
                'marks': question_obj.get('marks', '0'),
                'module': question_obj.get('module', 'Unknown'),
                'course_code': exam_data.get('courseCode', 'Unknown'),
                'course_name': exam_data.get('courseName', 'Unknown'),
                'year': exam_data.get('year', 'Unknown'),
                'month': exam_data.get('month', 'Unknown'),
                'scheme': exam_data.get('scheme', 'Unknown'),
                'source_file': filename
            }

            # Only add if question text exists
            if processed_question['question']:
                self.questions_data.append(processed_question)

    def build_embeddings(self) -> None:
        """
        Build vector embeddings for all questions using sentence transformers.
        """
        if not self.questions_data:
            raise ValueError("No questions loaded. Call load_json_files() first.")

        logger.info("Building embeddings for all questions...")

        # Extract question texts
        question_texts = [q['question'] for q in self.questions_data]

        # Generate embeddings
        self.embeddings = self.model.encode(
            question_texts,
            convert_to_tensor=False,
            show_progress_bar=True
        )

        self.is_fitted = True
        logger.info(f"Built embeddings for {len(question_texts)} questions")

    def semantic_search(self, query: str, similarity_threshold: float = 0.5,
                        top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Perform semantic search to find similar questions.

        Args:
            query: Search query text
            similarity_threshold: Minimum similarity score (0-1)
            top_k: Maximum number of results to return

        Returns:
            List of matching questions with similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_embeddings() first.")

        # Encode the query
        query_embedding = self.model.encode([query])

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get indices of questions above threshold
        valid_indices = np.where(similarities >= similarity_threshold)[0]

        # Sort by similarity (descending)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]

        # Limit results
        top_indices = sorted_indices[:top_k]

        # Prepare results
        results = []
        for idx in top_indices:
            question_data = self.questions_data[idx].copy()
            question_data['similarity_score'] = float(similarities[idx])
            question_data['similarity_percentage'] = float(round(similarities[idx] * 100, 2))
            results.append(question_data)

        return results

    def get_module_distribution(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze module distribution for search results.

        Args:
            search_results: Results from semantic_search()

        Returns:
            Module distribution statistics
        """
        if not search_results:
            return {'modules': {}, 'total_questions': 0}

        module_counts = Counter()
        module_similarities = defaultdict(list)

        for result in search_results:
            module = result['module']
            similarity = result['similarity_score']

            module_counts[module] += 1
            module_similarities[module].append(similarity)

        # Calculate statistics
        module_stats = {}
        total_questions = len(search_results)

        for module, count in module_counts.items():
            avg_similarity = float(np.mean(module_similarities[module]))
            module_stats[module] = {
                'count': count,
                'percentage': float(round((count / total_questions) * 100, 2)),
                'avg_similarity': float(round(avg_similarity, 3))
            }

        return {
            'modules': module_stats,
            'total_questions': total_questions
        }

    def get_marks_distribution(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze marks distribution for search results.

        Args:
            search_results: Results from semantic_search()

        Returns:
            Marks distribution statistics
        """
        if not search_results:
            return {'marks': {}, 'total_questions': 0}

        marks_counts = Counter()
        marks_similarities = defaultdict(list)

        for result in search_results:
            marks = result['marks']
            similarity = result['similarity_score']

            marks_counts[marks] += 1
            marks_similarities[marks].append(similarity)

        # Calculate statistics
        marks_stats = {}
        total_questions = len(search_results)

        for marks, count in marks_counts.items():
            avg_similarity = float(np.mean(marks_similarities[marks]))
            marks_stats[marks] = {
                'count': count,
                'percentage': float(round((count / total_questions) * 100, 2)),
                'avg_similarity': float(round(avg_similarity, 3))
            }

        return {
            'marks': marks_stats,
            'total_questions': total_questions
        }

    def detect_frequent_topics(self, similarity_threshold: float = 0.7,
                               min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Detect frequently appearing topics/questions across all exams.

        Args:
            similarity_threshold: Threshold for considering questions similar
            min_frequency: Minimum frequency to be considered repeated

        Returns:
            List of frequent topic clusters
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_embeddings() first.")

        logger.info("Detecting frequent topics...")

        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(self.embeddings)

        # Find clusters of similar questions
        visited = set()
        topic_clusters = []

        for i in range(len(self.questions_data)):
            if i in visited:
                continue

            # Find all questions similar to question i
            similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]

            if len(similar_indices) >= min_frequency:
                cluster = []
                for idx in similar_indices:
                    if idx not in visited:
                        cluster.append({
                            'question_data': self.questions_data[idx],
                            'similarity_to_representative': float(similarity_matrix[i][idx])
                        })
                        visited.add(idx)

                if cluster:
                    # Representative question (first one)
                    representative = cluster[0]['question_data']

                    topic_clusters.append({
                        'representative_question': representative['question'],
                        'frequency': len(cluster),
                        'questions': cluster,
                        'modules': list(set(q['question_data']['module'] for q in cluster)),
                        'courses': list(set(q['question_data']['course_code'] for q in cluster)),
                        'years': list(set(str(q['question_data']['year']) for q in cluster))
                    })

        # Sort by frequency (descending)
        topic_clusters.sort(key=lambda x: x['frequency'], reverse=True)

        logger.info(f"Found {len(topic_clusters)} frequent topic clusters")
        return topic_clusters

    def get_stats(self) -> Dict[str, Any]:
        """
        Get general statistics about the loaded dataset.

        Returns:
            Dataset statistics
        """
        if not self.questions_data:
            return {'error': 'No data loaded'}

        # Count by various dimensions
        courses = Counter(q['course_code'] for q in self.questions_data)
        modules = Counter(q['module'] for q in self.questions_data)
        marks = Counter(q['marks'] for q in self.questions_data)
        years = Counter(str(q['year']) for q in self.questions_data)

        return {
            'total_questions': len(self.questions_data),
            'total_courses': len(courses),
            'total_modules': len(modules),
            'courses': dict(courses.most_common()),
            'modules': dict(modules.most_common()),
            'marks_distribution': dict(marks.most_common()),
            'years': dict(years.most_common()),
            'is_fitted': self.is_fitted
        }