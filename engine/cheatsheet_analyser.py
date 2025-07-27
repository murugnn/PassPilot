# cheatsheet_analyser.py

import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

class CheatSheetAnalyzer:
    """
    A class to handle the generation of exam cheat sheets.
    It reads questions from a JSON file and uses a generative AI
    model to create concise answers for a selected module.
    """

    def __init__(self, model_name: str = 'gemini-1.5-flash-latest'):
        """
        Initializes the Cheat Sheet Analyzer.

        Args:
            model_name (str): The name of the Gemini model to use.
        """
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"CheatSheetAnalyzer initialized successfully with model: {model_name}")
        except Exception as e:
            logger.critical(f"Failed to initialize the GenerativeModel: {e}", exc_info=True)
            self.model = None

        # Define the core instruction for the AI model
        self.system_instruction = (
            "You are an expert academic assistant. Your goal is to provide a concise, "
            "accurate, and easy-to-understand answer to the following university-level exam question. "
            "Focus on the key points that would be most useful for a cheat sheet."
        )
        # Define the generation configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=300
        )

    def get_available_modules(self, json_file_path: str) -> dict:
        """
        Loads a JSON file and returns a list of available modules.

        Args:
            json_file_path (str): The full path to the questions JSON file.

        Returns:
            dict: A dictionary containing the list of modules or an error message.
        """
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            questions = data.get("questions")
            if not questions:
                return {"error": "JSON file does not contain a 'questions' list."}
            
            all_modules = sorted(list(set(q['module'] for q in questions if 'module' in q)))
            return {"modules": all_modules}

        except FileNotFoundError:
            logger.warning(f"File not found: {json_file_path}")
            return {"error": f"The file '{json_file_path}' was not found."}
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in file: {json_file_path}")
            return {"error": f"The file '{json_file_path}' is not a valid JSON file."}
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading modules: {e}", exc_info=True)
            return {"error": "An unexpected error occurred."}

    def generate_for_module(self, json_file_path: str, selected_module: str) -> dict:
        """
        Generates a cheat sheet for a specific module from a JSON file.

        Args:
            json_file_path (str): The full path to the questions JSON file.
            selected_module (str): The name of the module to generate the cheat sheet for.

        Returns:
            dict: A dictionary containing the generated cheat sheet or an error message.
        """
        if not self.model:
            return {"error": "The generative model is not initialized. Check server logs."}

        # First, check if the module is valid using the helper method
        module_data = self.get_available_modules(json_file_path)
        if "error" in module_data:
            return module_data # Return the error from the helper
        if selected_module not in module_data.get("modules", []):
            return {"error": f"Module '{selected_module}' not found. Available modules: {module_data['modules']}"}

        # Since we already loaded the file, we reload it here.
        # In a more complex app, you might cache the file content.
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        selected_qs = [q for q in data["questions"] if q['module'] == selected_module]
        course_name = data.get("courseName", "the course")

        # This dictionary will be our final JSON response
        cheatsheet_result = {
            "courseName": course_name,
            "module": selected_module,
            "cheatsheet_content": []
        }

        logger.info(f"Generating cheat sheet for module: '{selected_module}'")

        for qobj in selected_qs:
            question = qobj['question']
            marks = qobj.get('marks', 'N/A')
            prompt = f"{self.system_instruction}\n\nQuestion: {question}"

            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                answer = response.text.strip()
                cheatsheet_result["cheatsheet_content"].append({
                    "question": question,
                    "marks": marks,
                    "answer": answer
                })
            except Exception as e:
                logger.error(f"API call failed for question '{question[:50]}...': {e}")
                cheatsheet_result["cheatsheet_content"].append({
                    "question": question,
                    "marks": marks,
                    "answer": f"[Error: API call failed for this question. See server logs for details.]"
                })

        return cheatsheet_result