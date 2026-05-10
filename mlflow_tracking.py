"""Shared MLflow integration helpers for StoneVolMain training scripts.

This module is intentionally fail-safe: all logging calls are wrapped so
training can continue even when MLflow is not installed or tracking is down.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional


def parse_mlflow_tags(tags_text: Optional[str]) -> Dict[str, str]:
    """Parse comma-separated tags in the form 'k=v,k2=v2'."""
    tags: Dict[str, str] = {}
    if not tags_text:
        return tags

    for token in str(tags_text).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                tags[key] = value
        else:
            tags[token] = "true"
    return tags


class MLflowTracker:
    """Minimal wrapper around MLflow with defensive error handling."""

    def __init__(
        self,
        enabled: bool = False,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = {str(k): str(v) for k, v in (tags or {}).items()}

        self._mlflow = None
        self._active = False
        self._warned = False

        if not self.enabled:
            return

        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            if self.tracking_uri:
                self._mlflow.set_tracking_uri(self.tracking_uri)
            if self.experiment_name:
                self._mlflow.set_experiment(self.experiment_name)
        except Exception as exc:  # pragma: no cover - depends on environment
            self._warn(f"MLflow disabled: {exc}")
            self.enabled = False
            self._mlflow = None

    @property
    def is_active(self) -> bool:
        return bool(self.enabled and self._active and self._mlflow is not None)

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        print(f"[mlflow] {message}")
        self._warned = True

    def start_run(self) -> None:
        if not self.enabled or self._mlflow is None or self._active:
            return
        try:
            kwargs: Dict[str, Any] = {}
            if self.run_name:
                kwargs["run_name"] = self.run_name
            self._mlflow.start_run(**kwargs)
            self._active = True
            if self.tags:
                self.set_tags(self.tags)
        except Exception as exc:
            self._warn(f"Failed to start run: {exc}")
            self.enabled = False

    def end_run(self, status: str = "FINISHED") -> None:
        if not self._active or self._mlflow is None:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:
            self._warn(f"Failed to end run: {exc}")
        finally:
            self._active = False

    def _sanitize_param_value(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
        elif isinstance(value, (list, tuple, dict)):
            try:
                text = json.dumps(value)
            except Exception:
                text = str(value)
        else:
            text = str(value)

        # Some backends limit param length; keep this conservative.
        if len(text) > 500:
            return text[:497] + "..."
        return text

    def _flatten(self, mapping: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in mapping.items():
            k = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(self._flatten(value, prefix=k))
            else:
                out[k] = value
        return out

    def log_params(self, params: Dict[str, Any], prefix: str = "") -> None:
        if not self.is_active:
            return
        try:
            flat = self._flatten(params, prefix=prefix) if prefix else self._flatten(params)
            prepared = {k: self._sanitize_param_value(v) for k, v in flat.items()}
            if not prepared:
                return

            # Log in chunks to avoid backend payload issues.
            items = list(prepared.items())
            chunk_size = 100
            for i in range(0, len(items), chunk_size):
                chunk = dict(items[i : i + chunk_size])
                self._mlflow.log_params(chunk)
        except Exception as exc:
            self._warn(f"Failed to log params: {exc}")

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            metric = float(value)
            if metric != metric:  # NaN check
                return None
            if metric in (float("inf"), float("-inf")):
                return None
            return metric
        except Exception:
            return None

    def _normalize_metric_key(self, key: Any) -> str:
        """Normalize metric names to avoid file-store path collisions.

        MLflow file tracking stores metrics as files on disk. If we log both
        keys like "loss" and "loss/0", some stores can create a directory/file
        collision under the same path. We flatten inner separators to keep a
        readable but single-level metric namespace.
        """
        name = str(key)
        # Keep '/' reserved for the optional external prefix (e.g. "train/").
        name = name.replace("/", "_")
        # Avoid whitespace in metric names for safer backend compatibility.
        name = "_".join(name.split())
        return name

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
        if not self.is_active:
            return
        try:
            prepared: Dict[str, float] = {}
            for key, value in metrics.items():
                metric = self._to_float(value)
                if metric is None:
                    continue
                normalized_key = self._normalize_metric_key(key)
                name = f"{prefix}/{normalized_key}" if prefix else normalized_key
                prepared[name] = metric

            if not prepared:
                return

            if step is None:
                self._mlflow.log_metrics(prepared)
            else:
                for name, metric in prepared.items():
                    self._mlflow.log_metric(name, metric, step=int(step))
        except Exception as exc:
            self._warn(f"Failed to log metrics: {exc}")

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if not self.is_active:
            return
        try:
            sanitized = {str(k): self._sanitize_param_value(v) for k, v in tags.items()}
            self._mlflow.set_tags(sanitized)
        except Exception as exc:
            self._warn(f"Failed to set tags: {exc}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self.is_active:
            return
        try:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as exc:
            self._warn(f"Failed to log artifact '{path}': {exc}")

    def log_artifacts(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self.is_active:
            return
        try:
            self._mlflow.log_artifacts(path, artifact_path=artifact_path)
        except Exception as exc:
            self._warn(f"Failed to log artifacts from '{path}': {exc}")
