"""LLM synthetic data generation entry point."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load_prompt(): return (ROOT/'prompts'/'generation_prompt.txt').read_text()
def call_llm_to_generate(client, attributes):
    prompt=load_prompt().format(**attributes)
    # response = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0.9)
    # return json.loads(response.choices[0].message.content)
    raise NotImplementedError('Connect your LLM provider here.')
