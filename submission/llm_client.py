"""
DeepSeek API client for prompt enhancement.
Converts casual user ideas into high-quality SDXL prompts via LLM.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from openai import OpenAI

SYSTEM_PROMPT = """You are a world-class visual art director and prompt engineer specializing in Stable Diffusion SDXL.

Your task: transform a user's rough poster/cover idea into a SINGLE high-quality English prompt optimized for SDXL generation.

RULES:
1. Output ONLY the final prompt — no explanations, no prefixes, no labels.
2. Describe the visual scene in rich detail: composition, lighting, color palette, art style, textures, mood.
3. Include specific artistic terminology: "rule of thirds", "chiaroscuro lighting", "depth of field", "bokeh", "film grain", "cinematic lighting", "matte painting", "octane render".
4. Specify shot type when relevant: close-up, wide shot, bird's eye view, dutch angle.
5. The prompt must be 40-80 words, highly detailed but concise.
6. Always end with quality boosters: "masterpiece, best quality, highly detailed, sharp focus".
7. NEVER include NSFW, gore, or harmful content.
8. If the user input is in Chinese, still output the prompt in English.

FORMAT: Just the prompt string, nothing else."""


class PromptEnhancer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY not set. Export it as an environment variable "
                    "or pass api_key= to PromptEnhancer()."
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def enhance(self, user_idea: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_idea},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^[\"'`]+|[\"'`]+$", "", raw)
        return raw
