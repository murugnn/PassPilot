import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

def main():
    """
    Main function to run the interactive cheat sheet generator.
    """
    print("Exam Cheat Sheet Generator ")

    # --- Securely configure the API key from a .env file ---
    # Call load_dotenv() to load variables from the .env file
    load_dotenv() 

    try:
        # Get the API key from the environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Corrected the error message to match the variable name
            raise ValueError("GEMINI_API_KEY not found in your environment or .env file.")
        genai.configure(api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with GEMINI_API_KEY='your_key_here'.")
        return

    # --- Initialize the Gemini Model ---
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        print(f"Error initializing the model: {e}")
        return

    # --- Load JSON file ---
    try:
        filename = input("Enter the path to your JSON file: ").strip()
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not a valid JSON file.")
        return

    # --- Module Selection ---
    all_modules = sorted(set(q['module'] for q in data["questions"]))
    if not all_modules:
        print("No modules found in the JSON file.")
        return

    print("\nAvailable modules:")
    for idx, mod in enumerate(all_modules, 1):
        print(f"  [{idx}] {mod}")

    while True:
        try:
            choice_str = input("\nSelect a module number: ").strip()
            mod_choice = int(choice_str)
            if 1 <= mod_choice <= len(all_modules):
                selected_module = all_modules[mod_choice - 1]
                break
            else:
                print("Invalid number. Please choose from the list.")
        except (ValueError):
            print("Invalid input. Please enter a number.")

    # --- Filter questions and generate answers ---
    selected_qs = [q for q in data["questions"] if q['module'] == selected_module]
    course_name = data.get("courseName", "the course")
    
    print("\n" + "="*50)
    print(f"Generating Cheat Sheet for: {selected_module} - {course_name}")
    print("="*50 + "\n")

    # Define system instructions for the AI model
    system_instruction = (
        "You are an expert academic assistant. Your goal is to provide a concise, "
        "accurate, and easy-to-understand answer to the following university-level exam question."
    )

    for qobj in selected_qs:
        question = qobj['question']
        marks = qobj.get('marks', 'N/A')
        
        print(f" Q: {question} ({marks} marks)")
        print("   Generating answer...")

        try:
            # Combine the instruction and the question into a single prompt
            prompt = f"{system_instruction}\n\nQuestion: {question}"

            # Call the API without the unsupported argument
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=300
                )
            )
            
            answer = response.text.strip()
            print(f" A: {answer}\n")
            
        except Exception as e:
            print(f"   [Error: API call failed for this question. {e}]\n")

if __name__ == "__main__":
    main()