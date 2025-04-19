from lark import Lark
import os

# --- Adjust these paths ---
grammar_file_path = "jaclang/compiler/jac.lark"
test_jac_code = """
with entry {
    smart_assert 1 == 1;
    smart_assert 1 == 2, "failed";
}
"""
# --- ---

try:
    # Ensure the path is correct relative to where you run the script
    script_dir = os.path.dirname(__file__)
    abs_grammar_path = os.path.join(script_dir, grammar_file_path)

    with open(abs_grammar_path, 'r') as f:
        grammar = f.read()

    print("Attempting to load grammar...")
    parser = Lark(grammar, start='start', parser='lalr', lexer='contextual') # Use same options as your project
    print("Grammar loaded successfully.")

    print("\nAttempting to parse test code...")
    tree = parser.parse(test_jac_code)
    print("Parsing successful!")
    print(tree.pretty()) # Optional: view the parse tree
except Exception as e:
    print(f"\n--- Error during grammar test ---")
    print(e)
    # Optional: raise e # Re-raise to see full traceback