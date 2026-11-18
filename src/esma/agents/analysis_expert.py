"""Battle-summary agent."""

import time

from openai import OpenAI


class AnalyzationExpert:
    def __init__(self, api_base, api_key, model):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.input_tokens = 0
        self.ouput_tokens = 0
        self.inference_time = 0

    def system_prompt(self):
        return (
            "Analyze the StarCraft II battle. Summarize effective tactics, key issues, "
            "and one improvement in no more than 100 words. Do not add a heading."
        )

    def user_prompt(self, obs):
        return f"Battle data: {obs}\nProvide a concise summary."

    def _get_summary(self, system_prompt, user_prompt):
        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        started_at = time.time()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        self.inference_time = time.time() - started_at
        return completion.choices[0].message.content
