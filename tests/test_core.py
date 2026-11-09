"""Credential-free tests. Run: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promptlab.lint import lint_prompt
from promptlab.providers import ProviderError, complete, list_models
from promptlab.rewrite import (
    GOALS,
    build_judge_request,
    build_rewrite_request,
    fill_placeholders,
    parse_judge,
    parse_rewrite,
    placeholders,
)


def ids(result):
    return {f.id for f in result.findings}


class LintTests(unittest.TestCase):
    def test_empty(self):
        r = lint_prompt("   ")
        self.assertEqual(r.score, 0)
        self.assertIn("empty", ids(r))

    def test_too_short(self):
        self.assertIn("too_short", ids(lint_prompt("write a blog")))

    def test_good_prompt_scores_high(self):
        good = (
            "You are a support lead writing for new customers. Summarise the ticket below "
            "in three bullet points so that an agent can reply in under a minute. "
            "Use only the ticket text; if something is missing, say \"not found\".\n"
            "<ticket>\nMy order 4412 arrived broken.\n</ticket>"
        )
        r = lint_prompt(good)
        self.assertGreaterEqual(r.score, 90, r.findings)

    def test_no_format_and_goal_and_audience(self):
        r = lint_prompt("Tell me about solar panels and how they are installed on roofs today")
        self.assertTrue({"no_format", "no_goal", "no_audience"} <= ids(r))

    def test_audience_recognised_from_for_phrase(self):
        r = lint_prompt("Write a LinkedIn post about my AI audit service for college admissions teams under 150 words")
        self.assertNotIn("no_audience", ids(r))

    def test_mechanical_task_skips_audience_and_goal(self):
        r = lint_prompt("Extract name, email and company from the email below and return JSON with those keys")
        self.assertNotIn("no_audience", ids(r))
        self.assertNotIn("no_goal", ids(r))

    def test_conflicting_length(self):
        r = lint_prompt("Give a brief but comprehensive overview of our onboarding for the sales team")
        self.assertIn("conflicting_length", ids(r))

    def test_fact_request_without_fallback(self):
        r = lint_prompt("Give me statistics on remote work adoption for my newsletter to executives")
        self.assertIn("no_fallback", ids(r))

    def test_fact_request_with_fallback_passes(self):
        r = lint_prompt(
            "Give me statistics on remote work for my newsletter. Only use the report below "
            "and say not found if a figure is missing."
        )
        self.assertNotIn("no_fallback", ids(r))

    def test_unfenced_long_input(self):
        r = lint_prompt("Summarise this for the team: " + ("lorem ipsum dolor sit amet " * 40))
        self.assertIn("unfenced_input", ids(r))

    def test_fenced_long_input_ok(self):
        r = lint_prompt("Summarise this for the team:\n<doc>\n" + ("lorem ipsum dolor " * 50) + "\n</doc>")
        self.assertNotIn("unfenced_input", ids(r))

    def test_shouting(self):
        r = lint_prompt("You MUST NEVER ever ignore this IMPORTANT rule when you write the report for me")
        self.assertIn("shouting", ids(r))

    def test_score_bounds(self):
        r = lint_prompt("x")
        self.assertTrue(0 <= r.score <= 100)


class RewriteTests(unittest.TestCase):
    REPLY = (
        "<improved_prompt>\nYou are a tutor. Explain {{topic}} to a beginner.\n</improved_prompt>\n"
        "<changes>\n- Added a role\n- Added a placeholder\n</changes>\n"
        "<questions>\n- What topic?\n</questions>"
    )

    def test_parse(self):
        r = parse_rewrite(self.REPLY)
        self.assertTrue(r.parsed_ok)
        self.assertIn("You are a tutor", r.improved)
        self.assertEqual(r.changes, ["Added a role", "Added a placeholder"])
        self.assertEqual(r.questions, ["What topic?"])

    def test_parse_none_questions(self):
        r = parse_rewrite("<improved_prompt>x</improved_prompt><questions>\nnone\n</questions>")
        self.assertEqual(r.questions, [])

    def test_parse_fallback_on_missing_tags(self):
        r = parse_rewrite("Just a plain answer")
        self.assertFalse(r.parsed_ok)
        self.assertEqual(r.improved, "Just a plain answer")

    def test_placeholders(self):
        text = "Hi {{name}}, about {{ topic }} and {{name}} again"
        self.assertEqual(placeholders(text), ["name", "topic"])

    def test_fill(self):
        out = fill_placeholders("Hi {{name}} re {{topic}}", {"name": "Sam", "topic": ""})
        self.assertEqual(out, "Hi Sam re {{topic}}")

    def test_every_goal_builds(self):
        for g in GOALS:
            system, user = build_rewrite_request("write a post about dogs", g, ["no format"])
            self.assertIn("improved_prompt", system)
            self.assertIn("write a post about dogs", user)
            self.assertIn("no format", user)

    def test_unknown_goal(self):
        with self.assertRaises(ValueError):
            build_rewrite_request("x", "nope")

    def test_judge_parse(self):
        self.assertEqual(parse_judge("WINNER: B\nREASON: clearer."), ("B", "clearer."))
        self.assertEqual(parse_judge("winner: tie\nreason: same")[0], "TIE")
        self.assertEqual(parse_judge("no idea")[0], "UNKNOWN")

    def test_judge_request_has_both_answers(self):
        _, user = build_judge_request("task", "", "ans one", "ans two")
        self.assertIn("ans one", user)
        self.assertIn("ans two", user)


class FakeResp:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


class FakeHttp:
    """Records the last call and returns a canned response."""

    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.resp

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.resp


class ProviderTests(unittest.TestCase):
    KEY = "sk-test-SECRET-1234567890"

    def test_anthropic_complete(self):
        http = FakeHttp(FakeResp(200, {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }))
        c = complete("anthropic", self.KEY, "m", "sys", "hi", http=http)
        self.assertEqual((c.text, c.input_tokens, c.output_tokens), ("hello", 5, 2))
        method, url, kw = http.calls[0]
        self.assertEqual(kw["json"]["system"], "sys")
        self.assertEqual(kw["headers"]["x-api-key"], self.KEY)
        self.assertNotIn(self.KEY, url)

    def test_openai_complete(self):
        http = FakeHttp(FakeResp(200, {
            "choices": [{"message": {"content": "yo"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }))
        c = complete("openai", self.KEY, "m", "sys", "hi", http=http)
        self.assertEqual((c.text, c.input_tokens), ("yo", 3))
        self.assertEqual(http.calls[0][2]["json"]["messages"][0]["role"], "system")

    def test_gemini_complete_key_in_header_not_url(self):
        http = FakeHttp(FakeResp(200, {
            "candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        }))
        c = complete("gemini", "AIzaFAKEKEY1234567", "gem", "sys", "hi", http=http)
        self.assertEqual(c.text, "ab")
        _, url, kw = http.calls[0]
        self.assertNotIn("AIza", url)
        self.assertEqual(kw["headers"]["x-goog-api-key"], "AIzaFAKEKEY1234567")
        self.assertIn("systemInstruction", kw["json"])

    def test_error_hides_key(self):
        http = FakeHttp(FakeResp(401, {"error": {"message": f"bad key {self.KEY}"}}))
        with self.assertRaises(ProviderError) as ctx:
            complete("openai", self.KEY, "m", "", "hi", http=http)
        msg = str(ctx.exception)
        self.assertNotIn("SECRET", msg)
        self.assertIn("rejected", msg)

    def test_missing_key_or_model(self):
        with self.assertRaises(ProviderError):
            complete("openai", "", "m", "", "hi", http=FakeHttp(FakeResp(200, {})))
        with self.assertRaises(ProviderError):
            complete("openai", "k", "", "", "hi", http=FakeHttp(FakeResp(200, {})))

    def test_list_models_filters(self):
        http = FakeHttp(FakeResp(200, {"data": [
            {"id": "gpt-x"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"},
        ]}))
        self.assertEqual(list_models("openai", "k", http=http), ["gpt-x"])

    def test_list_models_gemini_only_generate(self):
        http = FakeHttp(FakeResp(200, {"models": [
            {"name": "models/g1", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/e1", "supportedGenerationMethods": ["embedContent"]},
        ]}))
        self.assertEqual(list_models("gemini", "k", http=http), ["g1"])

    def test_list_models_anthropic(self):
        http = FakeHttp(FakeResp(200, {"data": [{"id": "a1"}, {"id": "a2"}]}))
        self.assertEqual(list_models("anthropic", "k", http=http), ["a1", "a2"])


if __name__ == "__main__":
    unittest.main()
