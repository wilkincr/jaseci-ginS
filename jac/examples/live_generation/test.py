import threading
import time
import os
import base64
from google import genai
from google.genai import types

# Initialize Gemini Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

running = True
current_level = 1
last_user_action = "The player started the game."


def clean_code_block(text):
    lines = text.strip().splitlines()
    return "\n".join(
        line for line in lines if not line.strip().startswith("```")
    )


def get_next_level_code(level, last_action):
    model = "gemini-2.0-flash" 

    prompt = f"""
    You're designing a text-based puzzle maze game in Python.
    Generate Python code for level {level}. Make it *slightly more difficult* than the previous level.

    Base it on this input from the player: "{last_action}"

    Requirements:
    - Ask the player to solve a small puzzle or maze (e.g., a choice, riddle, or logic gate).
    - Print results depending on the player’s input.
    - Keep the code short and self-contained.
    - Code must be valid and runnable with exec().
    - Return only code (no explanations, no markdown).
    """

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain"
    )

    full_code = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        full_code += chunk.text

    return clean_code_block(full_code)


def monitor_input():
    global running, current_level, last_user_action
    while running:
        user_input = input("\nType 'next' to go to the next level, or 'quit' to exit: ")
        if user_input.strip().lower() == 'next':
            current_level += 1
            print(f"\n--- LEVEL {current_level} ---")
            code = get_next_level_code(current_level, last_user_action)
            print("Injecting code from Gemini:\n")
            print(code)
            try:
                exec(code, globals())
            except Exception as e:
                print("Error in generated code:", e)
        elif user_input.strip().lower() == 'quit':
            running = False
        else:
            last_user_action = user_input


thread = threading.Thread(target=monitor_input, daemon=True)
thread.start()

print(f"--- LEVEL {current_level} ---")
initial_code = get_next_level_code(current_level, last_user_action)
print("Injecting initial level code:\n")
print(initial_code)
try:
    exec(initial_code, globals())
except Exception as e:
    print("Error in generated code:", e)

try:
    while running:
        time.sleep(1)
except KeyboardInterrupt:
    running = False