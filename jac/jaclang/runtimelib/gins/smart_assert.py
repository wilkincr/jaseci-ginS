from jaclang.runtimelib.gins.ghost import active_shell_ghost
import os
from datetime import datetime
import json

def smart_assert(condition: bool):
    """
    Acts like a regular assert, but prompts the LLM with annotated code if the assertion fails.
    """
    if condition:
        return

    if active_shell_ghost is None:
        raise AssertionError(f"[SmartAssert] Assertion failed (No active ghost available)")

    if active_shell_ghost:
        active_shell_ghost.worker_update_once()

    annotated_code = active_shell_ghost.annotate_source_code()

    file_path = getattr(active_shell_ghost, "mod_path", "<unknown>")

    prompt = f"""
    A smart assertion failed during program execution.
        
    Below is the program annotated with control flow, execution frequency, and memory/runtime info:

    {annotated_code}

    Taking into account the annotated code, please explain why this failure occurred and how it might be prevented or fixed.
    """
    print("[SmartAssert] Sending context and annotated code to LLM...")
    response = active_shell_ghost.model.generate(prompt)
    print(f"\n[SmartAssert LLM Analysis]:\n{response.strip()}\n")
    output = {
        "file" : file_path,
        "annotated_code": annotated_code,
        "llm_response": response,
    }

    # 4) Write it out
    out_dir = "examples/gins_scripts/smart_assert/smart_assert_reports"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"report_{datetime.utcnow():%Y%m%d_%H%M%S%f}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as fp:
        json.dump(output, fp, indent=2)
        
    raise AssertionError(f"[SmartAssert] Assertion failed")
