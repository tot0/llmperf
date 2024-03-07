"""RequestGenerator which sweeps over prompt/generation token ratios generating requests for each."""

from typing import Any, Iterator, List

from llmperf.models import RequestConfig
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    sample_random_positive_int,
)


class RatioSweepRequestGenerator:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        model: str,
        max_num_requests: int,
        total_tokens_mean: int,
        total_tokens_stddev: int,
        prompt_percentage: float,
    ):
        self.model = model
        self.max_num_requests = max_num_requests
        self.total_tokens_mean = total_tokens_mean
        self.total_tokens_stddev = total_tokens_stddev
        self.prompt_percentage = prompt_percentage

    def __iter__(self) -> Iterator[RequestConfig]:
        i = 0
        if self.max_num_requests == -1:

            def condition(i):
                return True
        else:

            def condition(i):
                return i < self.max_num_requests

        while condition(i):
            total_tokens = sample_random_positive_int(
                self.total_tokens_mean, self.total_tokens_stddev
            )
            prompt_tokens = int(total_tokens * self.prompt_percentage)
            generation_tokens = total_tokens - prompt_tokens
            prompt = randomly_sample_sonnet_lines_prompt(
                prompt_tokens, 0, generation_tokens
            )
            yield RequestConfig(
                model=self.model,
                prompt=prompt,
                sampling_params={"max_tokens": generation_tokens},
            )
            i += 1

    def __len__(self) -> int:
        return self.max_num_requests

    def metadata(self) -> Any:
        return {
            "total_tokens_mean": self.total_tokens_mean,
            "total_tokens_stddev": self.total_tokens_stddev,
            "prompt_percentage": self.prompt_percentage,
        }
