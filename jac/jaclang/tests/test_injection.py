from jaclang.runtimelib.gins.ghost import ShellGhost
from google import genai
from google.genai import types
ghost = ShellGhost()

client = genai.Client(
    api_key="AIzaSyDCb9X9QEPg12gPPcfvAu7zudYnF9Qowu0"
)
def clean_code_block(text):
    lines = text.strip().splitlines()
    return "\n".join(
        line for line in lines if not line.strip().startswith("```")
    )
code_to_inject = """
print("Hello from injected code!")
self.my_injected_variable = "I was injected!"
"""
model = "gemini-2.0-flash" 

prompt = f"""
You're currently a shell ghost running parallel to a python script, give random safe python code to insert into the script for testing purposes.
Requirements:
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

full_code = clean_code_block(full_code)
print(full_code)
result = ghost.inject_code(full_code, method_name="worker", position="before")

if result["success"]:
    print(f"Code successfully injected into method: {result['method']}")
else:
    print(f"Failed to inject code: {result['error']}")

ghost.start_ghost()