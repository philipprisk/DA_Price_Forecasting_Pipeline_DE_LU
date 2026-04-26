from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from ..config import EnergyArenaSubmissionConfig, load_config
from ..integrations.energy_arena.client import EnergyArenaClient
from ..integrations.energy_arena.formatters import (
    build_candidate_payload,
    build_submission_context,
    extract_submission_predictions,
    load_payload_template,
    render_payload_template,
)
from ..integrations.energy_arena.sources import (
    EnergyArenaForecastResult,
    infer_source_name,
    load_or_run_forecast_source,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Energy Arena submissions."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a forecast to Energy Arena."
    )
    parser.add_argument("--config", required=True, help="Path to the Energy Arena submission YAML config.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the payload. Without this flag, the command runs in dry-run mode.",
    )
    return parser


def _require_api_key(env_var: str) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Environment variable '{env_var}' is not set.")
    return api_key


def _resolve_submit_url(config: EnergyArenaSubmissionConfig) -> str | None:
    if not config.submit_url:
        return None
    if config.submit_url.startswith("http://") or config.submit_url.startswith("https://"):
        return config.submit_url
    return urljoin(config.api_base_url.rstrip("/") + "/", config.submit_url.lstrip("/"))


def _build_payload(
    submission_config: EnergyArenaSubmissionConfig,
    forecast_result: EnergyArenaForecastResult,
) -> tuple[dict, pd.DataFrame]:
    predictions = extract_submission_predictions(
        forecast_df=forecast_result.forecast,
        forecast_date=submission_config.forecast_date,
        objective=submission_config.objective.value,
        target_tz=getattr(forecast_result.model_config, "target_tz", submission_config.target_tz),
        value_column=submission_config.value_column or forecast_result.default_value_column,
        quantile_columns=submission_config.quantile_columns or forecast_result.default_quantile_columns,
    )

    if submission_config.payload_template_path is None:
        payload = build_candidate_payload(
            predictions=predictions,
            submission_config=submission_config,
            model_config=forecast_result.model_config,
            source_metadata=forecast_result.metadata,
        )
        return payload, predictions

    template = load_payload_template(submission_config.payload_template_path)
    context = build_submission_context(
        predictions=predictions,
        submission_config=submission_config,
        model_config=forecast_result.model_config,
        source_metadata=forecast_result.metadata,
    )
    payload = render_payload_template(template, context)
    return payload, predictions


def run_energy_arena_submission(
    submission_config: EnergyArenaSubmissionConfig,
    submit: bool = False,
) -> None:
    """Generate an Energy Arena payload and optionally submit it via the API."""
    load_dotenv(submission_config.repo_root / ".env")

    source_name = infer_source_name(submission_config)
    artifact_dir = (
        submission_config.artifacts_dir
        / f"challenge_{submission_config.challenge_id}"
        / submission_config.forecast_date.isoformat()
        / source_name
    )
    forecast_dir = artifact_dir / "forecast_run"

    forecast_result = load_or_run_forecast_source(
        submission_config=submission_config,
        forecast_dir=forecast_dir,
    )

    payload, predictions = _build_payload(
        submission_config=submission_config,
        forecast_result=forecast_result,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifact_dir / "predictions_for_submission.csv", index=False)
    payload_path = artifact_dir / "test_payload.json"
    with open(payload_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(artifact_dir / "submission_payload.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(artifact_dir / "submission_config.json", "w", encoding="utf-8") as handle:
        json.dump(submission_config.model_dump(mode="json"), handle, indent=2)
    if forecast_result.model_config is not None and hasattr(forecast_result.model_config, "model_dump"):
        with open(artifact_dir / "source_config.json", "w", encoding="utf-8") as handle:
            json.dump(forecast_result.model_config.model_dump(mode="json"), handle, indent=2)

    if not submit:
        print(
            "Dry run complete. Saved payload to "
            f"{artifact_dir / 'test_payload.json'}"
        )
        return

    api_key = _require_api_key(submission_config.api_key_env)
    client = EnergyArenaClient(
        api_key=api_key,
        api_base_url=submission_config.api_base_url,
        submit_url=_resolve_submit_url(submission_config),
        api_key_header_name=submission_config.api_key_header_name,
        api_key_prefix=submission_config.api_key_prefix,
        timeout_seconds=submission_config.request_timeout_seconds,
    )
    response_payload = client.submit_json(payload)
    with open(artifact_dir / "submission_response.json", "w", encoding="utf-8") as handle:
        json.dump(response_payload, handle, indent=2)

    print(f"Submission complete. Saved response to {artifact_dir / 'submission_response.json'}")


def main(argv: list[str] | None = None) -> None:
    """Generate and optionally submit an Energy Arena forecast."""
    args = build_parser().parse_args(argv)
    submission_config = load_config(Path(args.config), EnergyArenaSubmissionConfig)
    run_energy_arena_submission(submission_config=submission_config, submit=args.submit)


if __name__ == "__main__":
    main()
