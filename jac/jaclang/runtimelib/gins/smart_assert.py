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

    versions = active_shell_ghost.annotate_source_code()
    versions = versions.get("complete")
    file_path = active_shell_ghost.source_file_path
    file_name = os.path.basename(file_path)

    prompt = (
        f"A smart assertion failed.\n\n"
        f"Here is the program annotated with control flow, instructions, and variable values:\n\n{versions}\n\n"
        "Please explain why this failure occurred and how to fix it."
    )
    response, token_info = active_shell_ghost.model.generate(prompt)
    response = response.strip()

    if token_info:
        print(f"Token usage for generating {token_info}")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "file": file_name,
        "prompt": prompt,
        "response": response,
        "token_info" : token_info
    }
    #out_dir = "examples/final_test_cases/test_suite/smart_assert_reports"
    #os.makedirs(file_path, exist_ok=True)
    source_dir = os.path.dirname(active_shell_ghost.source_file_path)

    fname = f"{file_name}_complete_report_{datetime.utcnow():%Y%m%d_%H%M%S%f}.json"
    full_path = os.path.join(source_dir, fname)

    with open(full_path, "w") as fp:
        json.dump(report, fp, indent=2)
    
    print(f"[SmartAssert] failed\n\n{response}")


    # prompt = f"""
    # A smart assertion failed during program execution.
        
    # Below is the program annotated with control flow, execution frequency, and memory/runtime info:

    # {annotated_code}

    # Taking into account the annotated code, please explain why this failure occurred and how it might be prevented or fixed.
    # """
    # print("[SmartAssert] Sending context and annotated code to LLM...")
    # response = active_shell_ghost.model.generate(prompt)
    # print(f"\n[SmartAssert LLM Analysis]:\n{response.strip()}\n")
    # output = {
    #     "file" : file_path,
    #     "annotated_code": annotated_code,
    #     "llm_response": response,
    # }

    # out_dir = "examples/gins_scripts/smart_assert/smart_assert_reports"
    # os.makedirs(out_dir, exist_ok=True)
    # fname = f"{file_name}_report_{datetime.utcnow():%Y%m%d_%H%M%S%f}.json"
    # path = os.path.join(out_dir, fname)
    # with open(path, "w") as fp:
    #     json.dump(output, fp, indent=2)
        
    raise AssertionError(f"[SmartAssert] failed")
