import json
import os
import re
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray
from opentelemetry import trace
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients
from llmperf.models import RequestConfig
from llmperf.otel import _root_logger
from llmperf.request_generators.sonnet.input_output import (
    InputOutputMeanStdDevRequestGenerator,
)
from llmperf.request_generators.sonnet.ratio_sweep import RatioSweepRequestGenerator
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import LLMPerfResults, RetryConfig


def get_token_throughput_latencies(
    model: str,
    request_generator: Iterable[RequestConfig],
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
    user_metadata={},
    metrics_meter=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    clients = construct_clients(
        llm_api=llm_api,
        num_clients=num_concurrent_requests,
        retry_config=RetryConfig(max_retries=10, delay_s=3, backoff=1),
    )
    req_launcher = RequestsLauncher(clients)
    completed_requests = []
    num_completed_requests = 0
    start_time = time.monotonic()
    iteration = 0
    pbar = tqdm(total=max_num_completed_requests)

    request_generator_iter = iter(request_generator)
    while (
        time.monotonic() - start_time < test_timeout_s
        and len(completed_requests) < max_num_completed_requests
    ):
        iteration += 1

        request_config = next(request_generator_iter)

        request_uuid_id = f"{str(uuid.uuid4().hex)}"
        request_id = ""
        if "experiment_name" in user_metadata:
            request_id = f"{user_metadata.get('experiment_name', '')}-"
        request_id += f"{user_metadata.get('session_id', '')}-{user_metadata.get('benchmark_config_name', '')}-{user_metadata.get('num_concurrent_requests', '')}-{user_metadata.get('platform', '')}-{user_metadata.get('model', '')}-{request_uuid_id}"

        request_config.sampling_params.update(additional_sampling_params)
        request_config.llm_api = llm_api
        request_config.metadata["request_id"] = request_id

        req_launcher.launch_requests(request_config)
        # Retrieving results less frequently allows for more concurrent requests
        # to be launched. This will overall reduce the amount of time it takes
        # for the test to run.
        if not (iteration % num_concurrent_requests):
            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, request_config = out
                num_output_tokens = get_token_length(gen_text)
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
                    request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                )

                message = "Completed Request"
                if request_metrics.get(common_metrics.ERROR_CODE, None) is not None:
                    message = f"Failed Request: {request_metrics.get(common_metrics.ERROR_CODE)}"
                _root_logger.info(
                    message,
                    extra={
                        "request_id": request_config.metadata.get("request_id", "none"),
                        "num_input_tokens": request_metrics[
                            common_metrics.NUM_INPUT_TOKENS
                        ],
                        "num_output_tokens": request_metrics[
                            common_metrics.NUM_OUTPUT_TOKENS
                        ],
                        "num_total_tokens": request_metrics[
                            common_metrics.NUM_TOTAL_TOKENS
                        ],
                        "inter_token_lat": request_metrics[
                            common_metrics.INTER_TOKEN_LAT
                        ],
                        "ttft": request_metrics[common_metrics.TTFT],
                        "e2e_lat": request_metrics[common_metrics.E2E_LAT],
                        "req_output_throughput": request_metrics[
                            common_metrics.REQ_OUTPUT_THROUGHPUT
                        ],
                        "error_code": request_metrics.get(common_metrics.ERROR_CODE, "")
                        or "",
                        "error_msg": request_metrics.get(common_metrics.ERROR_MSG, ""),
                    },
                )
                if metrics_meter:
                    # TODO: Record metrics on failure?
                    metrics_meter.record_request_metrics(
                        request_config, request_metrics, user_metadata
                    )

                all_metrics.append(request_metrics)
            completed_requests.extend(all_metrics)
        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        if num_output_tokens:
            request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
        else:
            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
        )

        all_metrics.append(request_metrics)
    completed_requests.extend(all_metrics)

    print(
        f"\nResults for token benchmark for {model} queried with the {llm_api} api.\n"
    )
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        **request_generator.metadata(),
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput (per/s): {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")
    print(f"Completed Requests (per/s): {num_completed_requests_per_min / 60}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_ratio_sweep_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    total_tokens_mean: int,
    total_tokens_stddev: int,
    prompt_input_percentage: List[float],
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    metrics_meter,
):
    """Run the prompt sweep throughput benchmark."""

    sweep_results = []
    for prompt_percentage in prompt_input_percentage:
        user_metadata["prompt_input_percentage"] = prompt_percentage
        with tracer.start_as_current_span(
            "prompt_ratio_start", attributes={**user_metadata}
        ):
            _root_logger.info(
                f"prompt_ratio_start: {total_tokens_mean = } {total_tokens_stddev = } {prompt_percentage = }",
                extra=user_metadata,
            )
            request_generator = RatioSweepRequestGenerator(
                model=model,
                max_num_requests=-1,
                total_tokens_mean=total_tokens_mean,
                total_tokens_stddev=total_tokens_stddev,
                prompt_percentage=prompt_percentage,
            )

            summary, individual_responses = get_token_throughput_latencies(
                model=model,
                llm_api=llm_api,
                test_timeout_s=test_timeout_s,
                max_num_completed_requests=max_num_completed_requests,
                request_generator=request_generator,
                num_concurrent_requests=num_concurrent_requests,
                additional_sampling_params=json.loads(additional_sampling_params),
                user_metadata=user_metadata,
                metrics_meter=metrics_meter,
            )

            filename = f"{model}_{total_tokens_mean}total_{total_tokens_stddev}stdev_{int(prompt_percentage*100)}percent__{num_concurrent_requests}clients"
            filename = re.sub(r"[^\w\d-]+", "-", filename)
            filename = re.sub(r"-{2,}", "-", filename)
            summary_filename = f"{filename}_summary"
            individual_responses_filename = f"{filename}_individual_responses"

            # Update to metadata.
            summary.update(user_metadata)

            results = LLMPerfResults(name=summary_filename, metadata=summary)

            if results_dir:
                results_dir = Path(results_dir)
                if not results_dir.exists():
                    results_dir.mkdir(parents=True)
                elif not results_dir.is_dir():
                    raise ValueError(f"{results_dir} is not a directory")

                try:
                    with open(results_dir / f"{summary_filename}.json", "w") as f:
                        json.dump(results.to_dict(), f, indent=4, default=str)
                except Exception as e:
                    print(results.to_dict())
                    raise e

                try:
                    with open(
                        results_dir / f"{individual_responses_filename}.json", "w"
                    ) as f:
                        json.dump(individual_responses, f, indent=4)
                except Exception as e:
                    print(individual_responses)
                    raise e

            sweep_results.append(results)

    return sweep_results


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    metrics_meter,
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    request_generator = InputOutputMeanStdDevRequestGenerator(
        model=model,
        max_num_requests=-1,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
    )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        request_generator=request_generator,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
        user_metadata=user_metadata,
        metrics_meter=metrics_meter,
    )

    filename = (
        f"{model}_{mean_input_tokens}_{mean_output_tokens}_{num_concurrent_requests}"
    )
    filename = re.sub(r"[^\w\d-]+", "-", filename)
    filename = re.sub(r"-{2,}", "-", filename)
    summary_filename = f"{filename}_summary"
    individual_responses_filename = f"{filename}_individual_responses"

    # Update to metadata.
    summary.update(user_metadata)

    results = LLMPerfResults(name=summary_filename, metadata=summary)

    if results_dir:
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e

    return results


class MetricsMeter:
    def __init__(self, name: str):
        from opentelemetry import metrics

        self.metrics_meter = metrics.get_meter_provider().get_meter(name)

        self.num_input_tokens = self.metrics_meter.create_histogram("num_input_tokens")
        self.num_output_tokens = self.metrics_meter.create_histogram(
            "num_output_tokens"
        )
        self.num_total_tokens = self.metrics_meter.create_histogram("num_total_tokens")
        self.inter_token_lat = self.metrics_meter.create_histogram("inter_token_lat")
        self.ttft = self.metrics_meter.create_histogram("ttft")
        self.e2e_lat = self.metrics_meter.create_histogram("e2e_lat")
        self.req_output_throughput = self.metrics_meter.create_histogram(
            "req_output_throughput"
        )

    def record_request_metrics(
        self, request_config: RequestConfig, request_metrics: dict, user_metadata: dict
    ):
        custom_dimensions = {
            "request_id": request_config.metadata.get("request_id", "none"),
            **user_metadata,
        }
        self.num_input_tokens.record(
            request_metrics[common_metrics.NUM_INPUT_TOKENS], custom_dimensions
        )
        self.num_output_tokens.record(
            request_metrics[common_metrics.NUM_OUTPUT_TOKENS], custom_dimensions
        )
        self.num_total_tokens.record(
            request_metrics[common_metrics.NUM_TOTAL_TOKENS], custom_dimensions
        )
        self.inter_token_lat.record(
            request_metrics[common_metrics.INTER_TOKEN_LAT], custom_dimensions
        )
        self.ttft.record(request_metrics[common_metrics.TTFT], custom_dimensions)
        self.e2e_lat.record(request_metrics[common_metrics.E2E_LAT], custom_dimensions)
        self.req_output_throughput.record(
            request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT], custom_dimensions
        )


if __name__ == "__main__":
    import argparse

    from llmperf.common import SUPPORTED_APIS
    from llmperf.otel import init_telemetry

    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--model", type=str, required=True, help="The model to use for this load test."
    )
    args.add_argument(
        "--platform",
        type=str,
        required=False,
        help="The platform target model is running on, e.g 'vllm-cu12.0.1-4xA100'",
    )
    args.add_argument(
        "--endpoint",
        type=str,
        default="http://0.0.0.0:5001/v1",
        help="The endpoint to use.",
    )
    args.add_argument(
        "--llm-api",
        type=str,
        default="openai",
        help=(
            f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
            " (default: %(default)s)"
        ),
    )
    args.add_argument(
        "--additional-sampling-params",
        type=str,
        default="{}",
        help=(
            "Additional sampling params to send with the each request to the LLM API. "
            "(default: %(default)s) No additional sampling params are sent."
        ),
    )
    args.add_argument(
        "--azure-monitor-connection-string",
        type=str,
        required=True,
        help="Connection string for Azure Monitor to export otel logs/spans/metrics to.",
    )
    args.add_argument(
        "--benchmark-configs",
        metavar="CONFIG",
        nargs="+",
        type=str,
        required=True,
        help="The paths to the benchmark configs.",
    )
    args.add_argument(
        "--benchmark-key-override",
        type=str,
        nargs="+",
        help=(
            "Override the benchmark key in the benchmark config with the given value."
            " This is useful for running the same benchmark with different parameters."
            "e.g. --benchmark-key-override '{\"num_concurrent_requests\": [1]}' '{\"prompt_input_percentage\": [0.8]}'"
        ),
    )
    args.add_argument(
        "--results-dir",
        type=str,
        default="results_dir",
        help=(
            "The directory to save the results to. "
            "(`default: %(default)s`) No results are saved)"
        ),
    )
    args.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help=("Unique name for experiment being run."),
    )

    args = args.parse_args()

    os.environ["OPENAI_API_BASE"] = args.endpoint
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "memes")

    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})

    experiment_name = args.experiment_name
    session_id = uuid.uuid4().hex

    init_telemetry(args.azure_monitor_connection_string, experiment_name, session_id)

    results_dir = str(Path(args.results_dir) / args.model / args.platform)

    start = time.time()
    _root_logger.info(
        f"llmperf_start: session_id={session_id}",
        extra={
            "session_id": session_id,
            "experiment_name": experiment_name,
            "model": args.model,
            "platform": args.platform,
            "llm_api": args.llm_api,
            "additional_sampling_params": args.additional_sampling_params,
            "benchmark_configs": args.benchmark_configs,
            "results_dir": results_dir,
        },
    )

    tracer = trace.get_tracer(__name__)

    session_results = []

    with tracer.start_as_current_span(
        "llmperf_start", attributes={"experiment_name": experiment_name}
    ):
        metrics_meter = MetricsMeter("llmperf_metrics")
        for benchmark_config_file in args.benchmark_configs:
            with open(benchmark_config_file) as f:
                benchmark_config = json.load(f)

            if args.benchmark_key_override:
                for override in args.benchmark_key_override:
                    override = json.loads(override)
                    for key, value in override.items():
                        _root_logger.info(f"Overriding {key}={value}")
                        benchmark_config[key] = value

            for concurrent_requests in benchmark_config["num_concurrent_requests"]:
                benchmark_config_name = Path(benchmark_config_file).name

                user_metadata = {
                    "model": args.model,
                    "platform": args.platform,
                    "experiment_name": experiment_name,
                    "session_id": session_id,
                    "benchmark_config_name": benchmark_config_name,
                    "num_concurrent_requests": concurrent_requests,
                    "max_num_completed_requests": benchmark_config[
                        "max_num_completed_requests"
                    ],
                    "timeout": benchmark_config["timeout"],
                }
                with tracer.start_as_current_span(
                    "run_benchmark", attributes={**user_metadata}
                ):
                    _root_logger.info(
                        f"benchmark_start: {benchmark_config_name = } {concurrent_requests = }",
                        extra=user_metadata,
                    )

                    # TODO: Parse benchmark_config into BenchRunner and RequestGenerator instead of this hack.
                    if "prompt_input_percentage" in benchmark_config:
                        user_metadata["total_tokens_mean"] = benchmark_config[
                            "total_tokens"
                        ]["mean"]
                        user_metadata["total_tokens_stddev"] = benchmark_config[
                            "total_tokens"
                        ]["stddev"]
                        results = run_ratio_sweep_benchmark(
                            llm_api=args.llm_api,
                            model=args.model,
                            test_timeout_s=benchmark_config["timeout"],
                            max_num_completed_requests=benchmark_config[
                                "max_num_completed_requests"
                            ],
                            total_tokens_mean=benchmark_config["total_tokens"]["mean"],
                            total_tokens_stddev=benchmark_config["total_tokens"][
                                "stddev"
                            ],
                            prompt_input_percentage=benchmark_config[
                                "prompt_input_percentage"
                            ],
                            num_concurrent_requests=concurrent_requests,
                            additional_sampling_params=args.additional_sampling_params,
                            results_dir=results_dir,
                            user_metadata=user_metadata,
                            metrics_meter=metrics_meter,
                        )
                    else:
                        user_metadata["mean_input_tokens"] = benchmark_config[
                            "mean_input_tokens"
                        ]
                        user_metadata["stddev_input_tokens"] = benchmark_config[
                            "stddev_input_tokens"
                        ]
                        user_metadata["mean_output_tokens"] = benchmark_config[
                            "mean_output_tokens"
                        ]
                        user_metadata["stddev_output_tokens"] = benchmark_config[
                            "stddev_output_tokens"
                        ]
                        results = run_token_benchmark(
                            llm_api=args.llm_api,
                            model=args.model,
                            test_timeout_s=benchmark_config["timeout"],
                            max_num_completed_requests=benchmark_config[
                                "max_num_completed_requests"
                            ],
                            mean_input_tokens=benchmark_config["mean_input_tokens"],
                            stddev_input_tokens=benchmark_config["stddev_input_tokens"],
                            mean_output_tokens=benchmark_config["mean_output_tokens"],
                            stddev_output_tokens=benchmark_config[
                                "stddev_output_tokens"
                            ],
                            num_concurrent_requests=concurrent_requests,
                            additional_sampling_params=args.additional_sampling_params,
                            results_dir=results_dir,
                            user_metadata=user_metadata,
                            metrics_meter=metrics_meter,
                        )

                    if isinstance(results, list):
                        session_results.extend(results)
                    else:
                        session_results.append(results)

    duration = time.time() - start
    _root_logger.info(
        f"llmperf_end: session_id={session_id}, duration_s={duration}",
        extra={
            "duration": duration,
            "experiment_name": experiment_name,
            "session_id": session_id,
            "model": args.model,
            "platform": args.platform,
            "llm_api": args.llm_api,
            "additional_sampling_params": args.additional_sampling_params,
            "benchmark_configs": args.benchmark_configs,
            "results_dir": results_dir,
        },
    )
