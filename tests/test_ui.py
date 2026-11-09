"""Headless UI smoke test with the network call replaced. Run: python tests/test_ui.py"""
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest
from promptlab.providers import Completion

REPLY = ("<improved_prompt>\nYou are a tutor. Explain {{topic}} to a beginner in 5 bullets.\n"
         "</improved_prompt><changes>\n- Added role\n</changes><questions>\n- Which topic?\n</questions>")

def fake_complete(provider, key, model, system, user, http=None):
    if "Goal type" in user:
        return Completion(REPLY, 10, 20)
    if "<answer_a>" in user:
        return Completion("WINNER: A\nREASON: clearer.", 5, 5)
    return Completion("an answer", 7, 3)

with mock.patch("promptlab.providers.complete", fake_complete):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception, at.exception

    at.text_area(key="original").set_value("explain recursion").run()
    assert any("Prompt check" in m.label for m in at.metric), "metric missing"

    at.text_input(key="model_text").set_value("m1")
    at.text_input(key="api_key").set_value("sk-fake-key-0000000000").run()
    btn = [b for b in at.button if b.label == "Rewrite with my model"][0]
    assert not btn.disabled
    btn.click().run()
    assert not at.exception, at.exception
    assert "tutor" in at.session_state["improved_text"], at.session_state["improved_text"]

    at.text_input(key="ph_topic").set_value("recursion").run()
    [b for b in at.button if b.label == "Apply answers"][0].click().run()
    assert "{{" not in at.session_state["improved_text"], at.session_state["improved_text"]

    [b for b in at.button if b.label == "Run both"][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["runs"] is not None
    [b for b in at.button if b.label == "Ask the model to judge"][0].click().run()
    assert at.session_state["verdict"] is not None, "no verdict"
    assert at.session_state["verdict"][0] in ("original", "improved"), at.session_state["verdict"]
print("UI smoke test passed; verdict =", at.session_state["verdict"][0])
