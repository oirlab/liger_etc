import io
from datetime import datetime, timezone

import numpy as np
import streamlit as st


def _to_asdf_safe(obj):
	"""Convert nested objects into ASDF-friendly Python/numpy types."""
	if isinstance(obj, dict):
		return {str(k): _to_asdf_safe(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [_to_asdf_safe(v) for v in obj]
	if isinstance(obj, np.generic):
		return obj.item()
	if isinstance(obj, np.ndarray):
		return obj
	if isinstance(obj, (str, int, float, bool)) or obj is None:
		return obj
	return str(obj)


def build_results_asdf_bytes(payload: dict) -> bytes | None:
	"""Serialize results payload to ASDF bytes."""
	try:
		import asdf
	except Exception:
		return None

	tree = _to_asdf_safe(payload)
	buffer = io.BytesIO()
	af = asdf.AsdfFile(tree)
	af.write_to(buffer)
	return buffer.getvalue()


def download_results(payload: dict, filename_prefix: str = "liger_etc_results"):
	"""Render a download button for results packaged as an ASDF file."""
	asdf_bytes = build_results_asdf_bytes(payload)
	if asdf_bytes is None:
		st.warning("Install the 'asdf' Python package to enable ASDF downloads.")
		return

	stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	st.download_button(
		label="Download Results (.asdf)",
		data=asdf_bytes,
		file_name=f"{filename_prefix}_{stamp}.asdf",
		mime="application/octet-stream",
		key="download_results_asdf",
	)