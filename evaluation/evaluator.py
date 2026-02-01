"""
Evaluation framework for contract extraction system.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExtractionMetrics:
    """Computes evaluation metrics for extraction results."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        self.exact_matches = 0
        self.partial_matches = 0
        self.total_fields = 0
    
    def compute_field_level(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """
        Compute field-level metrics.
        
        Args:
            predicted: Predicted field values
            ground_truth: Ground truth field values
            
        Returns:
            Dictionary of metrics
        """
        self.reset()
        
        all_fields = set(predicted.keys()) | set(ground_truth.keys())
        self.total_fields = len(all_fields)
        
        for field in all_fields:
            pred_val = predicted.get(field)
            true_val = ground_truth.get(field)
            
            if pred_val is None and true_val is None:
                continue  # Both null, skip
            elif pred_val is not None and true_val is not None:
                # Both have values
                self.tp += 1
                if self._normalize(pred_val) == self._normalize(true_val):
                    self.exact_matches += 1
                elif self._partial_match(pred_val, true_val):
                    self.partial_matches += 1
            elif pred_val is not None and true_val is None:
                self.fp += 1
            else:  # pred_val is None and true_val is not None
                self.fn += 1
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict:
        """Calculate precision, recall, F1."""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        exact_accuracy = self.exact_matches / self.total_fields if self.total_fields > 0 else 0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "exact_accuracy": round(exact_accuracy, 3),
            "partial_matches": self.partial_matches,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn
        }
    
    def _normalize(self, value) -> str:
        """Normalize value for comparison."""
        return str(value).lower().strip()
    
    def _partial_match(self, pred, true) -> bool:
        """Check for partial match (substring or similarity)."""
        pred_norm = self._normalize(pred)
        true_norm = self._normalize(true)
        
        # Check substring match
        return pred_norm in true_norm or true_norm in pred_norm


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