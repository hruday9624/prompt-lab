"""PromptLab: lint, rewrite and compare prompts with your own API key.

Run:  streamlit run app.py
The key you type lives in this browser session only. It is not saved.
"""

import random

import streamlit as st

from promptlab.lint import lint_prompt
from promptlab.providers import PROVIDERS, ProviderError, complete, list_models
from promptlab.rewrite import (
    GOALS,
    build_judge_request,
    build_rewrite_request,
    fill_placeholders,
    parse_judge,
    parse_rewrite,
    placeholders,
)

st.set_page_config(page_title="PromptLab", layout="wide")
ss = st.session_state

for k, v in {
    "models": [],
    "improved_text": "",
    "changes": [],
    "questions": [],
    "runs": None,
    "verdict": None,
}.items():
    ss.setdefault(k, v)

# ---------------- sidebar: bring your own key ----------------
with st.sidebar:
    st.header("Your model")
    provider = st.selectbox(
        "Provider", list(PROVIDERS), format_func=lambda p: PROVIDERS[p], key="provider"
    )
    api_key = st.text_input(
        "API key", type="password", key="api_key",
        help="Kept in this browser session only. Never saved or logged.",
    )
    if st.button("Load my models", disabled=not api_key):
        try:
            ss.models = list_models(provider, api_key)
            if not ss.models:
                st.warning("No text models returned for this key.")
        except (ProviderError, OSError) as e:
            ss.models = []
            st.error(str(e))
    if ss.models:
        model = st.selectbox("Model", ss.models, key="model_pick")
    else:
        model = st.text_input("Model id", key="model_text", help="Or press Load my models.")
    st.caption(
        "Calls go from this app straight to the provider with your key. "
        "You pay the provider for tokens used. Linting is free and local."
    )

st.title("PromptLab")
st.write("Check a prompt, rewrite it for the job it has to do, then test the old and new versions side by side.")

tab_improve, tab_compare = st.tabs(["Improve", "Compare"])

# ---------------- tab 1: lint and rewrite ----------------
with tab_improve:
    original = st.text_area("Your prompt", height=180, key="original",
                            placeholder="Paste the prompt you want to improve.")
    goal = st.selectbox("What is this prompt for?", list(GOALS),
                        format_func=lambda g: GOALS[g]["label"], key="goal")
    st.caption(GOALS[goal]["guidance"])

    result = lint_prompt(original)
    c1, c2 = st.columns([1, 3])
    c1.metric("Prompt check", f"{result.score}/100", help="Local checks, no model call.")
    with c2:
        if not original.strip():
            st.info("Paste a prompt to see the checks.")
        elif result.clean:
            st.success("No issues found by the local checks.")
        else:
            for f in result.findings:
                st.markdown(f"**{f.severity.upper()}**: {f.message}  \n_Fix:_ {f.fix}")

    if st.button("Rewrite with my model", type="primary",
                 disabled=not (original.strip() and api_key and model)):
        system, user = build_rewrite_request(original, goal, [f.message for f in result.findings])
        with st.spinner("Rewriting..."):
            try:
                reply = complete(provider, api_key, model, system, user)
                parsed = parse_rewrite(reply.text)
                ss.improved_text = parsed.improved
                ss.changes = parsed.changes
                ss.questions = parsed.questions
                ss.runs = None
                ss.verdict = None
                if not parsed.parsed_ok:
                    st.warning("The model did not follow the reply format, so the raw reply is shown.")
            except (ProviderError, OSError) as e:
                st.error(str(e))
    if not api_key:
        st.caption("Enter an API key in the sidebar to rewrite.")

    if ss.improved_text:
        st.subheader("Improved prompt")
        st.text_area("Edit freely", height=260, key="improved_text")
        after = lint_prompt(ss.improved_text)
        st.caption(f"Check score: {result.score} before, {after.score} after.")

        holes = placeholders(ss.improved_text)
        if holes:
            st.markdown("**Fill in the blanks** (the rewrite left placeholders rather than guess)")
            for h in holes:
                st.text_input(h, key=f"ph_{h}")

            def _apply():
                vals = {h: ss.get(f"ph_{h}", "") for h in placeholders(ss.improved_text)}
                ss.improved_text = fill_placeholders(ss.improved_text, vals)

            st.button("Apply answers", on_click=_apply)

        if ss.changes:
            with st.expander("What changed and why", expanded=True):
                for line in ss.changes:
                    st.markdown(f"- {line}")
        if ss.questions:
            with st.expander("Details the rewrite could not know"):
                for line in ss.questions:
                    st.markdown(f"- {line}")

# ---------------- tab 2: compare ----------------
with tab_compare:
    if not ss.improved_text:
        st.info("Rewrite a prompt on the Improve tab first.")
    else:
        as_system = st.checkbox(
            "Treat the prompts as system prompts and send a test message",
            value=(goal == "system"),
            help="Use this for app system prompts. Otherwise each prompt is sent as the user message.",
        )
        test_msg = ""
        if as_system:
            test_msg = st.text_area("Test message a user would send", key="test_msg", height=100)
        criteria = st.text_input("What makes an answer good? (optional, used by the judge)", key="criteria")

        ready = bool(api_key and model and original.strip() and (test_msg.strip() or not as_system))
        if st.button("Run both", type="primary", disabled=not ready):
            runs = {}
            with st.spinner("Running original and improved..."):
                try:
                    for name, p in (("original", original), ("improved", ss.improved_text)):
                        if as_system:
                            runs[name] = complete(provider, api_key, model, p, test_msg)
                        else:
                            runs[name] = complete(provider, api_key, model, "", p)
                    ss.runs = runs
                    ss.verdict = None
                    ss.judge_task = test_msg if as_system else original
                except (ProviderError, OSError) as e:
                    st.error(str(e))

        if ss.runs:
            left, right = st.columns(2)
            for col, name in ((left, "original"), (right, "improved")):
                r = ss.runs[name]
                col.subheader(name.capitalize())
                col.write(r.text or "(empty reply)")
                if r.input_tokens is not None:
                    col.caption(f"{r.input_tokens} tokens in, {r.output_tokens} tokens out")

            if st.button("Ask the model to judge"):
                order = ["original", "improved"]
                random.shuffle(order)  # hide which is which, so position cannot bias the verdict
                a, b = ss.runs[order[0]].text, ss.runs[order[1]].text
                system, user = build_judge_request(ss.get("judge_task", ""), criteria, a, b)
                with st.spinner("Judging..."):
                    try:
                        reply = complete(provider, api_key, model, system, user)
                        winner, reason = parse_judge(reply.text)
                        label = {"A": order[0], "B": order[1]}.get(winner, winner.lower())
                        ss.verdict = (label, reason)
                    except (ProviderError, OSError) as e:
                        st.error(str(e))
            if ss.verdict:
                st.success(f"Judge picked: {ss.verdict[0]}. {ss.verdict[1]}")
                st.caption(
                    "One judge run with one sample is a hint, not proof. "
                    "A model judging its own answers can favour its own style."
                )
