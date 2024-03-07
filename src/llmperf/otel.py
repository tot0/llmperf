import logging
import sys
import uuid

_root_logger = logging.getLogger(__name__)
_root_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.flush = sys.stdout.flush
_FORMAT = "%(levelname)s %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%YYYY-%m-%dT%H:%M:%S"
_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
_root_logger.addHandler(_handler)


def _setup_otel_tracer(connection_string: str, resource):
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _tracer_provider = TracerProvider(resource=resource)

    trace.set_tracer_provider(_tracer_provider)
    exporter = AzureMonitorTraceExporter.from_connection_string(
        connection_string,
        instrumentation_collection=True,
    )
    span_processor = BatchSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)


def _setup_otel_logs(connection_string: str, resource, logging_level=logging.INFO):
    from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
    from opentelemetry._logs import (
        get_logger_provider,
        set_logger_provider,
    )
    from opentelemetry.sdk._logs import (
        LoggerProvider,
        LoggingHandler,
    )
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    set_logger_provider(LoggerProvider(resource=resource))

    exporter = AzureMonitorLogExporter.from_connection_string(connection_string)
    get_logger_provider().add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler()
    handler.setLevel(logging_level)
    _root_logger.addHandler(handler)


def _setup_otel_metrics(connection_string: str, resource):
    """Set up otel metrics exporter."""
    from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    exporter = AzureMonitorMetricExporter.from_connection_string(connection_string)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=1000)
    metrics.set_meter_provider(
        MeterProvider(metric_readers=[reader], resource=resource)
    )


def init_telemetry(connection_string: str, experiment_name: str, session_id: str):
    import platform
    import socket

    from opentelemetry.sdk.resources import (
        ProcessResourceDetector,
        get_aggregated_resources,
    )
    from opentelemetry.sdk.trace import Resource
    from opentelemetry.semconv.resource import ResourceAttributes

    _instance_id = f"{uuid.uuid4().hex}-{socket.gethostname()}-{platform.platform()}"

    _otel_resource = get_aggregated_resources(
        detectors=[ProcessResourceDetector()],
        initial_resource=Resource.create(
            attributes={
                # Check `opentelemetry/sdk/resources/__init__.py` for others.
                ResourceAttributes.SERVICE_NAME: "llmperf_benchmark",
                ResourceAttributes.SERVICE_INSTANCE_ID: _instance_id,
                "experiment_name": experiment_name,
                "session_id": session_id,
            }
        ),
    )

    # Spans
    _setup_otel_tracer(connection_string, _otel_resource)

    # Logs
    _setup_otel_logs(connection_string, _otel_resource)

    # Metrics
    _setup_otel_metrics(connection_string, _otel_resource)
