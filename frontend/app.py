import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odin.graph import build_graph


load_dotenv()
st.set_page_config(page_title="ODIN", page_icon="O", layout="centered")
st.title("ODIN")
st.caption("Agentic self-debugging demo")
task = st.text_area("Task description", placeholder="Describe the Python program to generate and debug.")

if st.button("Run", type="primary", disabled=not task.strip()):
    status = st.empty()
    output = st.container()
    initial_state = {
        "task": task,
        "code": "",
        "error": "",
        "error_category": "",
        "iteration": 0,
        "max_iterations": 3,
        "success": False,
        "history": [],
    }
    final_state = initial_state.copy()
    for update in build_graph().stream(initial_state, stream_mode="updates"):
        stage, values = next(iter(update.items()))
        final_state.update(values)
        status.write(f"Stage: {stage}")
        if stage == "generate":
            with output.container():
                with st.expander(f"Iteration {values['iteration']} code", expanded=True):
                    st.code(values["code"], language="python")
        elif stage == "analyze" and values.get("error_category"):
            st.write(f"Error category: {values['error_category']}")

    if final_state["success"]:
        st.success("Execution succeeded")
    else:
        st.error("Execution failed after the maximum iterations")
        if final_state["error"]:
            st.code(final_state["error"])