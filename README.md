# PromptLab

Check a prompt, rewrite it for the job it has to do, then run the old and new versions side by side on your own model.

PromptLab started in March 2025 as a small Hugging Face Space, [prompt-lab](https://huggingface.co/spaces/hruday96/prompt-lab), that rewrote a prompt with one Gemini key. This version is a rewrite: you bring your own key, it works with three providers, and it tests the result instead of only producing it.

## What it does

1. **Check.** Local rules score the prompt from 0 to 100 and say what is missing: audience, purpose, output format, a rule for missing facts, delimiters around pasted text, conflicting length requests, and a few more. No model call, no cost, same answer every time.
2. **Rewrite.** Pick what the prompt is for (a general task, an agent or tool-using task, an app system prompt, structured extraction, or writing). Your model rewrites the prompt to fit that job. It does not invent details: anything it cannot know becomes a `{{placeholder}}` that you fill in.
3. **Compare.** Run the original and the rewrite on the same model and read the answers side by side, with token counts. An optional judge picks a winner without knowing which answer came from which prompt.

## Run it

```
git clone https://github.com/hruday9624/prompt-lab
cd prompt-lab
python -m venv .venv
.venv/Scripts/activate        # on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Choose a provider in the sidebar, paste your key, press **Load my models**, and pick a model from the list your key can use. Python 3.10 or newer.

## Your key

- The key is held in your browser session by Streamlit. It is not written to disk, logged, or put in a URL.
- Calls go from the app straight to Anthropic, OpenAI or Google. There is no server of mine in between.
- Error messages strip the key before they are shown.
- You pay your provider for the tokens used. The check step costs nothing.

If you host this for other people, each visitor enters their own key. Do not put yours in the app.

## Providers

| Provider | Models listed from | Notes |
|---|---|---|
| Anthropic | `/v1/models` | Messages API |
| OpenAI | `/v1/models` | Chat Completions; embedding, audio and image models are filtered out of the list |
| Google | `/v1beta/models` | Only models that support `generateContent` |

Model names change often, so the app asks the provider what your key can use instead of keeping a list that goes stale. The calls use plain HTTP through `requests`, so there are no vendor SDKs to install.

## What the checks look for

Audience, purpose, output format, a rule for missing information when facts are requested, vague wording, long pasted text without delimiters, short and detailed requests in the same prompt, prompts made mostly of prohibitions, all-caps emphasis, and a strict format with no example. Extraction and conversion prompts skip the audience and purpose checks. The checks are heuristics. Each finding says why it fired so you can overrule it.

## Limits

- A rewrite is a draft. Read it before you rely on it.
- The judge is one model giving one opinion on one sample. Treat it as a hint. A model judging answers can favour its own style.
- A better-looking prompt does not always produce a better answer. That is why the Compare tab exists: run both and look.
- The linter is English-only.

## Tests

```
python -m unittest discover -s tests -v
python tests/test_ui.py
```

The first command runs the checks, the prompt building and parsing, and the provider adapters against a fake HTTP layer. The second drives the full interface with the network call replaced. Neither needs an API key.

## Licence

MIT. See `LICENSE`.
