"""RequestGenerator which sweeps over prompt/generation token ratios generating requests for each."""

from typing import Any, Iterator, List

from llmperf.models import RequestConfig
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    sample_random_positive_int,
)


class InputOutputMeanStdDevRequestGenerator:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        model: str,
        max_num_requests: int,
        mean_input_tokens: int,
        stddev_input_tokens: int,
        mean_output_tokens: int,
        stddev_output_tokens: int,
    ):
        self.model = model
        self.max_num_requessts = max_num_requests
        self.mean_input_tokens = mean_input_tokens
        self.stddev_input_tokens = stddev_input_tokens
        self.mean_output_tokens = mean_output_tokens
        self.stddev_output_tokens = stddev_output_tokens

    def __iter__(self) -> Iterator[RequestConfig]:
        i = 0
        if self.max_num_requests == -1:

            def condition(i):
                return True
        else:

            def condition(i):
                return i < self.max_num_requests

        while condition(i):
            num_output_tokens = sample_random_positive_int(
                self.mean_output_tokens, self.stddev_output_tokens
            )

            prompt = randomly_sample_sonnet_lines_prompt(
                prompt_tokens_mean=self.mean_input_tokens,
                prompt_tokens_stddev=self.stddev_input_tokens,
                expect_output_tokens=num_output_tokens,
            )

            yield RequestConfig(
                model=self.model,
                prompt=prompt,
                sampling_params={"max_tokens": num_output_tokens},
            )
            i += 1

    def __len__(self) -> int:
        return self.max_num_requests

    def metadata(self) -> Any:
        return {
            "mean_input_tokens": self.mean_input_tokens,
            "stddev_input_tokens": self.stddev_input_tokens,
            "mean_output_tokens": self.mean_output_tokens,
            "stddev_output_tokens": self.stddev_output_tokens,
        }
