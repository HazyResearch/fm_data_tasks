"""OpenAI client with cache."""
import logging
import os

import openai

from fm_data_tasks.utils.cache import Cache

openai.api_key = os.environ.get("OPENAI_API_KEY")

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class Client(object):
    """OpenAI Client."""

    def __init__(self, cache_path: str):
        """Init."""
        self.cache = Cache(cache_path)

    def query(
        self,
        engine: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: int,
        frequency_penalty: int,
        presence_penalty: int,
        n: int,
        overwrite_cache: bool,
    ) -> str:
        """Query OpenAI with cache."""
        request_params = {
            "engine": engine,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
        }
        cache_key = request_params.copy()
        try:

            def _run_completion():
                return openai.Completion.create(**request_params)

            response, cached = self.cache.get(
                cache_key, overwrite_cache, _run_completion
            )
        except openai.error.OpenAIError as e:
            logger.error(e)
            raise e
        return response
