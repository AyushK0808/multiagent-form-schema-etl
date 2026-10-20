"""
Evaluation framework for contract extraction system.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

def normalize_text(value) -> str:
    """Lowercase, drop articles and punctuation (keeping decimal points), collapse spaces."""
    text = str(value).lower().strip()
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # thousands separators
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred, gold) -> float:
    """SQuAD-style token-overlap F1 between two values."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class ExtractionMetrics:
    """
    Field-level metrics with token-overlap credit.

    A field only earns credit in proportion to how much of its value matches
    the gold value. A non-null prediction against a non-null gold value that
    shares no tokens scores 0. ``fill_rate`` keeps the earlier "was the field
    filled at all" measure, which is not a correctness metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.score_sum = 0.0       # sum of per-field token F1 over fields with a gold value
        self.n_pred = 0            # non-null predictions
        self.n_gold = 0            # non-null gold values
        self.exact_matches = 0
        self.filled = 0            # gold present and prediction non-null
        self.hallucinated = 0      # prediction non-null where gold is null
        self.missed = 0            # gold present, prediction null
        self.null_correct = 0      # both null
        self.total_fields = 0

    def update(self, predicted: Dict, ground_truth: Dict) -> None:
        """Accumulate one document into the running counters."""
        for field in set(predicted) | set(ground_truth):
            pred_val = predicted.get(field)
            true_val = ground_truth.get(field)
            self.total_fields += 1

            if pred_val is None and true_val is None:
                self.null_correct += 1
                continue
            if pred_val is not None:
                self.n_pred += 1
            if true_val is not None:
                self.n_gold += 1

            if pred_val is not None and true_val is not None:
                self.filled += 1
                self.score_sum += token_f1(pred_val, true_val)
                if normalize_text(pred_val) == normalize_text(true_val):
                    self.exact_matches += 1
            elif pred_val is not None:
                self.hallucinated += 1
            else:
                self.missed += 1

    def compute_field_level(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """Metrics for a single document."""
        self.reset()
        self.update(predicted, ground_truth)
        return self.get_metrics()

    def get_metrics(self) -> Dict:
        """Soft precision/recall/F1 over the accumulated documents."""
        precision = self.score_sum / self.n_pred if self.n_pred else 0.0
        recall = self.score_sum / self.n_gold if self.n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scored = self.n_gold + self.hallucinated
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "exact_accuracy": round(self.exact_matches / self.n_gold, 3) if self.n_gold else 0.0,
            "fill_rate": round(self.filled / self.n_gold, 3) if self.n_gold else 0.0,
            "hallucination_rate": round(self.hallucinated / self.n_pred, 3) if self.n_pred else 0.0,
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
            "exact_matches": self.exact_matches,
            "hallucinated": self.hallucinated,
            "missed": self.missed,
        }


class BaselineExtractor:
    """Simple baseline extractor for comparison."""
    
    def extract(self, text: str, field_name: str, field_type: str) -> any:
        """
        Naive extraction using regex patterns.
        
        Args:
            text: Full document text
            field_name: Field to extract
            field_type: Field type
            
        Returns:
            Extracted value or None
        """
        text_lower = text.lower()
        
        # Simple pattern matching based on field name
        patterns = {
            "effective_date": r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b',
            "governing_law": r'governed by.*?laws of ([\w\s]+)',
            "termination": r'terminate.*?(\d+\s+days)',
        }
        
        # Try to find pattern
        import re
        for key, pattern in patterns.items():
            if key in field_name.lower():
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(1)
        
        return None


class Evaluator:
    """Main evaluation class."""
    
    def __init__(self, test_data_dir: Path = None):
        self.test_data_dir = test_data_dir
        self.metrics = ExtractionMetrics()
        self.baseline = BaselineExtractor()
    
    def load_test_set(self) -> List[Dict]:
        """Load test cases from directory."""
        if not self.test_data_dir or not self.test_data_dir.exists():
            logger.warning("Test data directory not found")
            return []
        
        test_cases = []
        for test_file in self.test_data_dir.glob("*.json"):
            try:
                with open(test_file, 'r') as f:
                    test_case = json.load(f)
                    test_cases.append(test_case)
            except Exception as e:
                logger.error(f"Failed to load test case {test_file}: {e}")
        
        return test_cases
    
    def evaluate_extraction(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """Evaluate a single extraction."""
        return self.metrics.compute_field_level(predicted, ground_truth)
    
    def compare_with_baseline(self, test_case: Dict, extraction_fn) -> Dict:
        """
        Compare system extraction with baseline.
        
        Args:
            test_case: Test case with document and ground truth
            extraction_fn: Function that performs extraction
            
        Returns:
            Comparison results
        """
        document_text = test_case.get("document_text", "")
        ground_truth = test_case.get("ground_truth", {})
        
        # System extraction
        system_result = extraction_fn(test_case)
        system_metrics = self.evaluate_extraction(system_result, ground_truth)
        
        # Baseline extraction
        baseline_result = {}
        for field in ground_truth.keys():
            field_type = test_case.get("schema", {}).get("fields", {}).get(field, {}).get("type", "string")
            baseline_result[field] = self.baseline.extract(document_text, field, field_type)
        
        baseline_metrics = self.evaluate_extraction(baseline_result, ground_truth)
        
        return {
            "system": system_metrics,
            "baseline": baseline_metrics,
            "improvement": {
                "f1": round(system_metrics["f1"] - baseline_metrics["f1"], 3),
                "accuracy": round(system_metrics["exact_accuracy"] - baseline_metrics["exact_accuracy"], 3)
            }
        }
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate evaluation report."""
        report_lines = ["=" * 60]
        report_lines.append("EXTRACTION EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Aggregate results
        system_f1_avg = sum(r["system"]["f1"] for r in results) / len(results)
        baseline_f1_avg = sum(r["baseline"]["f1"] for r in results) / len(results)
        
        system_acc_avg = sum(r["system"]["exact_accuracy"] for r in results) / len(results)
        baseline_acc_avg = sum(r["baseline"]["exact_accuracy"] for r in results) / len(results)
        
        report_lines.append(f"\nTest Cases: {len(results)}")
        report_lines.append("\nSYSTEM PERFORMANCE:")
        report_lines.append(f"  Average F1: {system_f1_avg:.3f}")
        report_lines.append(f"  Average Accuracy: {system_acc_avg:.3f}")
        
        report_lines.append("\nBASELINE PERFORMANCE:")
        report_lines.append(f"  Average F1: {baseline_f1_avg:.3f}")
        report_lines.append(f"  Average Accuracy: {baseline_acc_avg:.3f}")
        
        report_lines.append("\nIMPROVEMENT:")
        report_lines.append(f"  F1 Improvement: {(system_f1_avg - baseline_f1_avg):.3f}")
        report_lines.append(f"  Accuracy Improvement: {(system_acc_avg - baseline_acc_avg):.3f}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def create_test_case(pdf_path: str, ground_truth: Dict, schema: Dict) -> Dict:
    """
    Create a test case for evaluation.
    
    Args:
        pdf_path: Path to PDF file
        ground_truth: Dictionary of correct field values
        schema: Schema definition
        
    Returns:
        Test case dictionary
    """
    return {
        "pdf_path": pdf_path,
        "ground_truth": ground_truth,
        "schema": schema,
        "metadata": {
            "created": "2024-01-01",
            "annotator": "human"
        }
    }