"""
Reflexive Policy Layer
======================
Evaluates the ExtractionBundle produced by ParallelExtractor and:

  1. Selects or fuses field-level outputs using confidence scores,
     field coverage, and source-specific strengths.
  2. Applies cross-field consistency rules (date ordering, party
     name cross-reference, etc.).
  3. Normalises values — strips domain jargon, standardises date
     formats, removes filler phrases like "as defined herein".

Policy weights
--------------
* Donut  → stronger for semantic / free-form fields (party names,
           governing law, confidentiality period).
* LayoutLM → stronger for spatially-anchored fields (effective date,
             signatures, amounts tied to layout position).

Output
------
PolicyResult
    fields          : fused {field_name: value}
    confidences     : {field_name: 0.0–1.0}
    coverage        : fraction of non-None fields
    consistency_ok  : bool
    issues          : list of policy-level warnings
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from extraction.parallel_extractor import ExtractionBundle, ExtractorResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jargon / filler phrases to strip from string values
# ---------------------------------------------------------------------------
_JARGON_PATTERNS = [
    re.compile(r"\b(as defined herein|as set forth herein|hereinafter referred to as)\b", re.I),
    re.compile(r"\b(the (company|party|entity|undersigned))\b", re.I),
    re.compile(r'["""]', re.I),
    re.compile(r"\(the\s+\"[\w\s]+\"\)", re.I),  # (the "Company")
]

# Fields where LayoutLMv3 spatial precision is preferred over Donut semantics
_SPATIAL_PREFERRED = {"effective_date", "start_date", "salary", "amount",
                      "signing_date", "expiry_date"}

# Minimum confidence to use a field's value at all (below = treat as None)
_CONF_FLOOR = 0.15

# A value with more words than this is a clause or a text dump, not a field value.
_MAX_VALUE_WORDS = 10


def _key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _as_date(value: Any) -> Optional[str]:
    """Canonical ISO date for any parseable date string with an explicit year, else None."""
    text = str(value).strip()
    if not re.search(r"\d{4}", text):
        return None
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(text, default=datetime(1900, 1, 1)).date().isoformat()
    except (ValueError, OverflowError):
        return None


def _type_valid(value: Any, field_type: str) -> bool:
    if field_type == "number":
        return _as_number(value) is not None
    if field_type == "date":
        return _as_date(value) is not None
    return True


def _canonical(value: Any, field_type: str) -> str:
    """Comparison form: equal dates and equal numbers compare equal whatever their format."""
    if field_type == "number" and _as_number(value) is not None:
        return repr(float(_as_number(value)))
    if field_type == "date" and _as_date(value) is not None:
        return _as_date(value)
    return _key(value)


def _reused_values(result: Optional[ExtractorResult]) -> set:
    """Values one extractor returned for two or more different fields. A model that gives the
    same answer to different questions was not answering them."""
    if result is None or not result.fields:
        return set()
    seen: Dict[str, int] = {}
    for value in result.fields.values():
        if value is not None:
            seen[_key(value)] = seen.get(_key(value), 0) + 1
    return {k for k, n in seen.items() if n > 1}


def _implausible(value: Any, field_type: str, reused: set) -> Optional[str]:
    """Why a value cannot be trusted regardless of its confidence, or None if it looks usable."""
    if not _type_valid(value, field_type):
        return f"not a valid {field_type}"
    if _key(value) in reused:
        return "same answer given to several fields"
    if field_type == "string" and len(str(value).split()) > _MAX_VALUE_WORDS:
        return "clause, not a value"
    return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    fields:         Dict[str, Any]
    confidences:    Dict[str, float]
    spatial_meta:   Dict[str, Any]
    coverage:       float               # filled / total
    consistency_ok: bool
    issues:         List[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "PolicyResult":
        return cls(
            fields={}, confidences={}, spatial_meta={},
            coverage=0.0, consistency_ok=True
        )


# ---------------------------------------------------------------------------
# Policy layer
# ---------------------------------------------------------------------------

class ReflexivePolicyLayer:
    """
    Fuses two ExtractorResult objects from an ExtractionBundle into a single
    coherent, normalised field set according to reflexive policy rules.
    """

    def __init__(
        self,
        conf_floor: float = _CONF_FLOOR,
        spatial_preferred: Optional[set] = None,
        normalise: bool = True,
        donut_fill: str = "non_numeric",
    ):
        # Whether Donut may fill a field LayoutLM left empty: "all", "non_numeric" (strings, dates,
        # booleans; a number on a page is easy to ground but hard to attribute to the right field)
        # or "none" (Donut only cross-checks and replaces unusable LayoutLM values).
        if donut_fill not in ("all", "non_numeric", "none"):
            raise ValueError(f"donut_fill must be all, non_numeric or none, not {donut_fill!r}")
        self.donut_fill = donut_fill
        self.conf_floor        = conf_floor
        # An explicit empty set disables the spatial rule (ablation).
        self.spatial_preferred = _SPATIAL_PREFERRED if spatial_preferred is None else spatial_preferred
        self.normalise         = normalise

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def fuse(self, bundle: ExtractionBundle, schema: Dict) -> PolicyResult:
        """Apply all policy rules and return a PolicyResult."""
        if bundle.donut is None and bundle.layoutlm is None:
            logger.warning("[Policy] Both extractors returned None")
            return PolicyResult.empty()

        issues: List[str] = []
        fused_fields:  Dict[str, Any]   = {}
        fused_conf:    Dict[str, float] = {}
        fused_spatial: Dict[str, Any]   = {}

        all_fields = bundle.all_field_names
        reused = {"donut": _reused_values(bundle.donut), "layoutlm": _reused_values(bundle.layoutlm)}

        for fname in all_fields:
            ftype = schema.get("fields", {}).get(fname, {}).get("type", "string")
            val, conf, spatial = self._select_field(fname, bundle, issues, ftype, reused)
            if self.normalise:
                val = self._normalise(val, schema.get("fields", {}).get(fname, {}))
            fused_fields[fname]  = val
            fused_conf[fname]    = conf
            fused_spatial[fname] = spatial

        # Cross-field consistency checks
        consistent = self._check_consistency(fused_fields, schema, issues)

        filled   = sum(1 for v in fused_fields.values() if v is not None)
        total    = len(fused_fields)
        coverage = filled / total if total else 0.0

        if issues:
            for issue in issues:
                logger.warning(f"[Policy] {issue}")

        logger.info(
            f"[Policy] Fusion complete: {filled}/{total} fields "
            f"(coverage={coverage:.2f}, consistent={consistent})"
        )
        return PolicyResult(
            fields=fused_fields,
            confidences=fused_conf,
            spatial_meta=fused_spatial,
            coverage=coverage,
            consistency_ok=consistent,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Per-field selection
    # ------------------------------------------------------------------

    def _select_field(
        self,
        fname: str,
        bundle: ExtractionBundle,
        issues: List[str],
        ftype: str = "string",
        reused: Optional[Dict[str, set]] = None,
    ) -> Tuple[Any, float, Any]:
        """
        Decide which extractor's value to use for this field.

        Strategy
        --------
        1. If both agree (normalised equality) → use LayoutLM value (has spatial).
        2. If only one has a value → use that one, flag low-coverage if weak.
        3. If both disagree, LayoutLM wins (it carries spatial evidence, and Donut's confidences are
           uncalibrated). Donut wins only when LayoutLM's value is demonstrably unusable (wrong type,
           the same answer given to several fields, or a clause rather than a value) while Donut's
           is usable, and the field is not spatially anchored. Confidence is never the deciding signal.
        4. Apply confidence floor: values below the floor are discarded (→ None).
        """
        d_val  = self._get(bundle.donut,    fname)
        l_val  = self._get(bundle.layoutlm, fname)
        d_conf = self._get_conf(bundle.donut,    fname)
        l_conf = self._get_conf(bundle.layoutlm, fname)
        l_spat = (bundle.layoutlm.spatial_meta or {}).get(fname) if bundle.layoutlm else None

        # Both None
        if d_val is None and l_val is None:
            return None, 0.0, None

        reused = reused or {}
        d_bad = _implausible(d_val, ftype, reused.get("donut", set())) if d_val is not None else None
        l_bad = _implausible(l_val, ftype, reused.get("layoutlm", set())) if l_val is not None else None

        # Only one has value
        if d_val is None:
            if l_conf < self.conf_floor:
                issues.append(f"'{fname}': LayoutLM value {l_val!r} below conf floor ({l_conf:.2f})")
                return None, l_conf, l_spat
            return l_val, l_conf, l_spat

        if l_val is None:
            if d_conf < self.conf_floor:
                issues.append(f"'{fname}': Donut value {d_val!r} below conf floor ({d_conf:.2f})")
                return None, d_conf, None
            if d_bad:
                issues.append(f"'{fname}': Donut-only value {d_val!r} discarded ({d_bad})")
                return None, 0.0, None
            if self.donut_fill == "none" or (self.donut_fill == "non_numeric" and ftype == "number"):
                issues.append(f"'{fname}': Donut-only {ftype} value {d_val!r} not used (donut_fill={self.donut_fill})")
                return None, 0.0, None
            return d_val, d_conf, None

        # Both have values — check agreement
        if self._values_agree(d_val, l_val, ftype):
            # Prefer LayoutLM (has spatial), use max confidence
            return l_val, max(d_conf, l_conf), l_spat

        # Disagreement. LayoutLM is the default; Donut takes over only on evidence that LayoutLM's
        # value is unusable and Donut's is not, never on confidence.
        if fname in self.spatial_preferred:
            chosen_val, chosen_conf, chosen_spat = l_val, l_conf, l_spat
            reason = "spatial field → LayoutLM preferred"
        elif l_bad and not d_bad:
            chosen_val, chosen_conf, chosen_spat = d_val, d_conf, None
            reason = f"LayoutLM value unusable ({l_bad}) → Donut"
        else:
            chosen_val, chosen_conf, chosen_spat = l_val, l_conf, l_spat
            reason = "LayoutLM default" + (f" (Donut value unusable: {d_bad})" if d_bad else "")

        issues.append(
            f"'{fname}': disagreement (donut={d_val!r} conf={d_conf:.2f} vs "
            f"layoutlm={l_val!r} conf={l_conf:.2f}) → {reason}"
        )
        return chosen_val, chosen_conf, chosen_spat

    # ------------------------------------------------------------------
    # Consistency rules
    # ------------------------------------------------------------------

    def _check_consistency(
        self,
        fields: Dict[str, Any],
        schema: Dict,
        issues: List[str],
    ) -> bool:
        ok = True

        # Rule 1: effective_date should not be after expiry/end dates
        date_order_pairs = [
            ("effective_date", "expiry_date"),
            ("effective_date", "termination_date"),
            ("start_date",     "end_date"),
        ]
        for early_f, late_f in date_order_pairs:
            early = fields.get(early_f)
            late  = fields.get(late_f)
            if early and late and isinstance(early, str) and isinstance(late, str):
                if early > late:   # ISO date strings compare correctly
                    issues.append(
                        f"Date order inconsistency: {early_f}={early} > {late_f}={late}"
                    )
                    ok = False

        # Rule 2: disclosing_party ≠ receiving_party
        dp = fields.get("disclosing_party")
        rp = fields.get("receiving_party")
        if dp and rp and str(dp).strip().lower() == str(rp).strip().lower():
            issues.append(
                "disclosing_party and receiving_party are identical — likely extraction error"
            )
            ok = False

        # Rule 3: required fields present (warn, don't hard-fail)
        for fname, meta in schema.get("fields", {}).items():
            if meta.get("required") and fields.get(fname) is None:
                issues.append(f"Required field '{fname}' is None after fusion")

        return ok

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(value: Any, field_meta: Dict) -> Any:
        """Strip domain jargon and standardise formats."""
        if not isinstance(value, str):
            return value
        v = value.strip()
        # Remove jargon patterns
        for pat in _JARGON_PATTERNS:
            v = pat.sub("", v).strip()
        # Collapse internal whitespace
        v = re.sub(r"\s{2,}", " ", v).strip()
        # Remove trailing punctuation noise (commas, semicolons)
        v = v.rstrip(",:;")
        return v or None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get(result: Optional[ExtractorResult], fname: str) -> Any:
        if result is None or result.fields is None:
            return None
        return result.fields.get(fname)

    @staticmethod
    def _get_conf(result: Optional[ExtractorResult], fname: str) -> float:
        if result is None or result.confidences is None:
            return 0.0
        return result.confidences.get(fname, 0.0)

    @staticmethod
    def _values_agree(a: Any, b: Any, field_type: str = "string") -> bool:
        """Loose equality: both non-None; dates and numbers compare by value, not by format."""
        if a is None or b is None:
            return False
        return _canonical(a, field_type) == _canonical(b, field_type)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def fuse_extractions(bundle: ExtractionBundle, schema: Dict) -> PolicyResult:
    return ReflexivePolicyLayer().fuse(bundle, schema)