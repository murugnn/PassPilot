import json
import os
import pickle

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExamAnalyzer:
    """
    Intelligent exam analysis engine using pre-labeled topics and semantic search.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './cache'):
        """
        Initialize the analyzer with a sentence transformer model.
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.questions_data: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_fitted = False
        self.data_hash: Optional[str] = None

    def load_json_files(self, folder_path: str) -> None:
        """
        Load and process all JSON files from the specified folder.
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

        self.data_hash = self._generate_data_hash()
        logger.info(f"Successfully loaded {len(self.questions_data)} questions")

    def _process_exam_data(self, exam_data: Dict[str, Any], filename: str) -> None:
        """
        Process individual exam JSON data and extract questions.
        """
        questions = exam_data.get('questions', [])

        for idx, question_obj in enumerate(questions):
            processed_question = {
                'id': f"{filename}_{idx}",
                'question': question_obj.get('question', '').strip(),
                ## --- NEW: Added the 'topic' field ---
                'topic': question_obj.get('topic', 'Untagged').strip(),
                'marks': question_obj.get('marks', '0'),
                'module': question_obj.get('module', 'Unknown'),
                'course_code': exam_data.get('courseCode', 'Unknown'),
                'course_name': exam_data.get('courseName', 'Unknown'),
                'year': exam_data.get('year', 'Unknown'),
                'month': exam_data.get('month', 'Unknown'),
                'scheme': exam_data.get('scheme', 'Unknown'),
                'source_file': filename
            }

            if processed_question['question']:
                self.questions_data.append(processed_question)

    # --- (Caching and embedding functions remain largely the same) ---
    def _generate_data_hash(self) -> str:
        import hashlib
        data_str = json.dumps(
            [(q['question'], q['course_code'], q['year'], q['topic']) for q in self.questions_data],
            sort_keys=True
        )
        return hashlib.md5(data_str.encode()).hexdigest()

    def _save_cache(self) -> None:
        try:
            cache_file = os.path.join(self.cache_dir, f'embeddings_{self.model_name.replace("/", "_")}.pkl')
            cache_data = {
                'questions_data': self.questions_data,
                'embeddings': self.embeddings,
                'data_hash': self.data_hash,
                'model_name': self.model_name
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")

    def _load_cache(self) -> bool:
        try:
            cache_file = os.path.join(self.cache_dir, f'embeddings_{self.model_name.replace("/", "_")}.pkl')
            if not os.path.exists(cache_file): return False

            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            if (cache_data.get('model_name') == self.model_name and
                    cache_data.get('data_hash') == self.data_hash):
                self.questions_data = cache_data['questions_data']
                self.embeddings = cache_data['embeddings']
                self.is_fitted = True
                logger.info("Loaded valid cache")
                return True
            else:
                logger.info("Cache invalid (different data or model)")
                return False
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return False

    def build_embeddings(self) -> None:
        if not self.questions_data:
            raise ValueError("No questions loaded. Call load_json_files() first.")

        if self._load_cache():
            logger.info("Using cached embeddings")
            return

        logger.info("Building embeddings for all questions...")
        question_texts = [q['question'] for q in self.questions_data]
        self.embeddings = self.model.encode(question_texts, convert_to_tensor=False, show_progress_bar=True)
        self.is_fitted = True
        self._save_cache()
        logger.info(f"Built embeddings for {len(question_texts)} questions")

    def semantic_search(self, query: str, similarity_threshold: float = 0.5, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_embeddings() first.")

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        top_indices = sorted_indices[:top_k]

        results = []
        for idx in top_indices:
            question_data = self.questions_data[idx].copy()
            question_data['similarity_score'] = float(similarities[idx])
            results.append(question_data)

        return results

    # --- (Distribution functions remain the same) ---
    def get_module_distribution(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This function works as-is
        pass

    def get_marks_distribution(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This function works as-is
        pass

    ## --- REWRITTEN & SIMPLIFIED: Replaces detect_frequent_topics ---
    def get_topic_analysis(self, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Analyzes topics by grouping questions based on their pre-labeled 'topic' field.

        Args:
            min_frequency: Minimum number of questions a topic must have to be included.

        Returns:
            A list of topics, sorted by frequency, with detailed analysis.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_embeddings() first.")

        logger.info(f"Analyzing pre-labeled topics with min_frequency={min_frequency}...")

        # Group question indices by their topic name
        topic_groups = defaultdict(list)
        for idx, question in enumerate(self.questions_data):
            topic_groups[question['topic']].append(idx)

        analyzed_topics = []
        for topic_name, indices in topic_groups.items():
            if len(indices) < min_frequency:
                continue

            topic_questions = [self.questions_data[i] for i in indices]

            # Use our embeddings to calculate how semantically similar the questions in this topic are
            cohesion_score = self._calculate_cluster_cohesion(indices)

            # Create examples (first 2 questions, truncated)
            examples = []
            for q in topic_questions[:2]:
                question_text = q['question']
                examples.append({
                    'question': question_text[:97] + "..." if len(question_text) > 100 else question_text,
                    'module': q['module'],
                    'course_code': q['course_code'],
                    'marks': q['marks']
                })

            analyzed_topics.append({
                'topic_name': topic_name,
                'frequency': len(topic_questions),
                'cohesion_score': cohesion_score,
                'modules': list(set(q['module'] for q in topic_questions)),
                'courses': list(set(q['course_code'] for q in topic_questions)),
                'years': list(set(str(q['year']) for q in topic_questions)),
                'examples': examples
            })

        # Sort by frequency (most frequent first)
        analyzed_topics.sort(key=lambda x: x['frequency'], reverse=True)

        logger.info(f"Found {len(analyzed_topics)} topics meeting the frequency criteria.")
        return analyzed_topics

    def _calculate_cluster_cohesion(self, cluster_indices: List[int]) -> float:
        """
        Calculate intra-cluster cohesion using average pairwise similarity.
        (This function is still useful for quality metrics!)
        """
        if len(cluster_indices) < 2:
            return 1.0  # A single item cluster is perfectly cohesive

        cluster_embeddings = self.embeddings[cluster_indices]
        similarity_matrix = cosine_similarity(cluster_embeddings)

        # Get upper triangle (excluding diagonal) to calculate average similarity
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        if len(upper_triangle_indices[0]) == 0:
            return 0.0

        mean_similarity = np.mean(similarity_matrix[upper_triangle_indices])
        return float(mean_similarity)

    ## --- DELETED ---
    # _extract_distinctive_keywords, _deduplicate_clusters, and detect_frequent_topics
    # are no longer needed.

    def get_stats(self) -> Dict[str, Any]:
        if not self.questions_data: return {'error': 'No data loaded'}

        return {
            'total_questions': len(self.questions_data),
            'total_courses': len(Counter(q['course_code'] for q in self.questions_data)),
            ## --- NEW: Added stats for topics ---
            'total_topics': len(Counter(q['topic'] for q in self.questions_data)),
            'courses': dict(Counter(q['course_code'] for q in self.questions_data).most_common()),
            'topics': dict(Counter(q['topic'] for q in self.questions_data).most_common()),
            'is_fitted': self.is_fitted
        }

    # In exam_analyzer.py, add these methods inside the ExamAnalyzer class

    def _calculate_topic_weights(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes all questions to calculate the frequency and average marks for each topic.
        This is a crucial helper for the pass strategy feature.
        """
        topic_stats = defaultdict(lambda: {'marks': [], 'frequency': 0})

        for q in self.questions_data:
            topic = q.get('topic', 'Untagged')
            if topic == 'Untagged':
                continue

            # Accumulate marks, converting to float for calculation
            try:
                # We handle marks that might be strings or numbers
                marks_value = float(q.get('marks', 0))
                if marks_value > 0:
                    topic_stats[topic]['marks'].append(marks_value)
            except (ValueError, TypeError):
                # Ignore if marks is not a valid number
                pass

            topic_stats[topic]['frequency'] += 1

        # Now, calculate the final weight and average marks for each topic
        weighted_topics = {}
        for topic, data in topic_stats.items():
            if not data['marks']:
                continue  # Skip topics that never have marks assigned

            avg_marks = np.mean(data['marks'])
            frequency = data['frequency']

            # The "value" or "weight" of a topic is its average mark value multiplied by its frequency.
            # This prioritizes topics that are both high-value and reliable.
            weighted_topics[topic] = {
                'average_marks': round(avg_marks, 2),
                'frequency': frequency,
                'strategic_value': round(avg_marks * frequency, 2)
            }

        return weighted_topics

    def get_pass_strategy(self, studied_topics: List[str], min_pass_marks: int) -> Dict[str, Any]:
        """
        Calculates a study strategy to help a student reach the minimum pass marks.

        Args:
            studied_topics: A list of topic names the student has already covered.
            min_pass_marks: The target score the student needs in the external exam.

        Returns:
            A dictionary containing the student's current status and recommended topics.
        """
        if not self.is_fitted:
            raise ValueError("Analyzer is not fitted. Please load data first.")

        # 1. Get the strategic value of every topic in the dataset
        all_topic_weights = self._calculate_topic_weights()

        # 2. Calculate the student's current estimated score
        current_estimated_score = 0
        for topic in studied_topics:
            if topic in all_topic_weights:
                current_estimated_score += all_topic_weights[topic]['average_marks']

        current_estimated_score = round(current_estimated_score)

        # 3. Determine the score deficit
        score_deficit = min_pass_marks - current_estimated_score

        # If the student is already on track, we can tell them!
        if score_deficit <= 0:
            return {
                'current_estimated_score': current_estimated_score,
                'target_marks': min_pass_marks,
                'score_deficit': 0,
                'recommendations': [],
                'summary': "You're on track! Your current study topics already cover the estimated marks needed to pass."
            }

        # 4. Find the most efficient topics to study next
        candidate_topics = {
            topic: data for topic, data in all_topic_weights.items()
            if topic not in studied_topics
        }

        # Sort candidate topics by their strategic value (most valuable first)
        sorted_candidates = sorted(
            candidate_topics.items(),
            key=lambda item: item[1]['strategic_value'],
            reverse=True
        )

        # 5. Build the list of recommendations
        recommendations = []
        projected_score_gain = 0
        for topic, data in sorted_candidates:
            if projected_score_gain >= score_deficit:
                break  # We have enough recommendations

            recommendations.append({
                'topic_name': topic,
                'potential_marks': data['average_marks'],
                'strategic_value': data['strategic_value']
            })
            projected_score_gain += data['average_marks']

        projected_new_score = current_estimated_score + projected_score_gain

        return {
            'current_estimated_score': current_estimated_score,
            'target_marks': min_pass_marks,
            'score_deficit': round(score_deficit, 2),
            'recommendations': recommendations,
            'projected_new_score': round(projected_new_score),
            'summary': f"You need about {round(score_deficit)} more marks. Studying the recommended topics can get you to an estimated score of {round(projected_new_score)}."
        }

    # In exam_analyzer.py, add these methods inside the ExamAnalyzer class

    def _prepare_simulation_data(self) -> None:
        """
        [NEW] Pre-calculates the probabilities and possible marks for each topic.
        This is run once and cached to make simulations fast.
        """
        if hasattr(self, 'simulation_data') and self.simulation_data:
            return  # Already prepared

        logger.info("Preparing data for Monte Carlo simulation...")

        # First, find out the total number of unique exams (papers)
        # We use 'source_file' as a unique identifier for an exam paper.
        unique_exams = set(q['source_file'] for q in self.questions_data)
        total_papers = len(unique_exams)

        if total_papers == 0:
            self.simulation_data = {}
            return

        # Group all marks and appearances by topic
        topic_profiles = defaultdict(lambda: {'appearances': 0, 'marks_options': []})
        for q in self.questions_data:
            topic = q.get('topic', 'Untagged')
            if topic == 'Untagged':
                continue

            try:
                marks = float(q.get('marks', 0))
                if marks > 0:
                    topic_profiles[topic]['marks_options'].append(marks)
                    # We need a way to count unique papers a topic appears in
                    # Let's use a set for each topic to store paper filenames
                    if 'papers' not in topic_profiles[topic]:
                        topic_profiles[topic]['papers'] = set()
                    topic_profiles[topic]['papers'].add(q['source_file'])
            except (ValueError, TypeError):
                continue

        self.simulation_data = {}
        for topic, data in topic_profiles.items():
            num_appearances = len(data.get('papers', set()))
            self.simulation_data[topic] = {
                # Probability of this topic appearing on any given paper
                'probability': num_appearances / total_papers,
                # List of possible marks this topic could be worth if it appears
                'marks_options': data['marks_options']
            }

        logger.info(f"Prepared simulation data for {len(self.simulation_data)} topics.")

    def run_pass_simulation(self, studied_topics: List[str], target_marks: int, num_simulations: int = 100000) -> Dict[
        str, Any]:
        """
        [NEW] Runs a Monte Carlo simulation to estimate pass probability.
        """
        # Ensure the statistical data is ready
        self._prepare_simulation_data()

        studied_topics_set = set(studied_topics)
        simulation_scores = []

        # This is the main simulation loop
        for _ in range(num_simulations):
            current_exam_score = 0
            # Build one "random" exam paper based on historical probabilities
            for topic, data in self.simulation_data.items():
                # Does this topic appear in our simulated exam?
                if random.random() < data['probability']:
                    # Yes. Is this a topic the student has studied?
                    if topic in studied_topics_set:
                        # Yes. Assign a random historical mark value to it.
                        marks_for_this_topic = random.choice(data['marks_options'])
                        current_exam_score += marks_for_this_topic

            simulation_scores.append(current_exam_score)

        # Now, analyze the results from all 10,000 simulations
        scores_array = np.array(simulation_scores)

        successful_simulations = np.sum(scores_array >= target_marks)
        pass_probability = successful_simulations / num_simulations

        return {
            'pass_probability': round(pass_probability, 2),
            'target_marks': target_marks,
            'num_simulations': num_simulations,
            'projected_score': {
                'average': round(np.mean(scores_array), 2),
                'median': round(np.median(scores_array), 2),
            },
            # This shows the range of likely outcomes (worst-case vs. best-case)
            'score_distribution': {
                'likely_range_5_to_95_percentile': [
                    round(np.percentile(scores_array, 5), 2),
                    round(np.percentile(scores_array, 95), 2)
                ]
            }
        }