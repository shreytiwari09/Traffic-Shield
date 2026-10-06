"""Hermetic test settings.

Environment variables win over .env in pydantic-settings, and this file is
imported before any test module (and therefore before services.shared.settings
is instantiated). A developer's local .env may turn on tracing or a prompt
canary for a demo; tests must behave identically on every machine and in CI.
"""

import os

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
os.environ["CANARY_PERCENT"] = "0"
