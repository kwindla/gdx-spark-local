"""Word Error Rate (WER) and Character Error Rate (CER) calculation.

Uses the jiwer library for standard WER/CER computation without
any text normalization (normalization is handled by the LLM judge).
"""

from datetime import datetime
from typing import Optional

import jiwer
from loguru import logger

from asr_eval.models import EvaluationMetrics, ServiceName
from asr_eval.storage.database import Database


class WERCalculator:
    """Calculates WER and CER metrics for transcription results."""

    def __init__(self):
        self.db = Database()

    def calculate_wer(
        self,
        reference: str,
        hypothesis: str,
    ) -> dict:
        """Calculate WER and related metrics.

        Args:
            reference: Ground truth transcription
            hypothesis: STT transcription to evaluate

        Returns:
            Dictionary with WER, substitutions, deletions, insertions, reference_words
        """
        # Handle empty strings
        if not reference.strip():
            if not hypothesis.strip():
                return {
                    "wer": 0.0,
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                    "reference_words": 0,
                }
            else:
                # All words are insertions
                words = len(hypothesis.split())
                return {
                    "wer": float("inf"),
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": words,
                    "reference_words": 0,
                }

        if not hypothesis.strip():
            # All words are deletions
            words = len(reference.split())
            return {
                "wer": 1.0,
                "substitutions": 0,
                "deletions": words,
                "insertions": 0,
                "reference_words": words,
            }

        # Calculate WER using jiwer
        output = jiwer.process_words(reference, hypothesis)

        return {
            "wer": output.wer,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "reference_words": len(reference.split()),
        }

    def calculate_cer(
        self,
        reference: str,
        hypothesis: str,
    ) -> float:
        """Calculate Character Error Rate.

        Args:
            reference: Ground truth transcription
            hypothesis: STT transcription to evaluate

        Returns:
            CER as a float (0.0 = perfect, 1.0 = all errors)
        """
        if not reference.strip():
            if not hypothesis.strip():
                return 0.0
            else:
                return float("inf")

        if not hypothesis.strip():
            return 1.0

        return jiwer.cer(reference, hypothesis)

    def evaluate(
        self,
        sample_id: str,
        service_name: ServiceName,
        reference: str,
        hypothesis: str,
    ) -> EvaluationMetrics:
        """Evaluate a transcription against ground truth.

        Args:
            sample_id: Sample identifier
            service_name: STT service that produced the hypothesis
            reference: Ground truth transcription
            hypothesis: STT transcription

        Returns:
            EvaluationMetrics with WER, CER, and error counts
        """
        wer_results = self.calculate_wer(reference, hypothesis)
        cer = self.calculate_cer(reference, hypothesis)

        return EvaluationMetrics(
            sample_id=sample_id,
            service_name=service_name,
            wer=wer_results["wer"],
            cer=cer,
            substitutions=wer_results["substitutions"],
            deletions=wer_results["deletions"],
            insertions=wer_results["insertions"],
            reference_words=wer_results["reference_words"],
            timestamp=datetime.utcnow(),
        )

    async def evaluate_all(
        self,
        service_name: ServiceName,
        progress_callback: Optional[callable] = None,
    ) -> list[EvaluationMetrics]:
        """Evaluate all transcriptions for a service.

        Fetches ground truth and transcriptions from the database,
        computes WER/CER, and stores the results.

        Args:
            service_name: Service to evaluate
            progress_callback: Optional callback(current, total, sample_id)

        Returns:
            List of EvaluationMetrics
        """
        await self.db.initialize()

        # Get all samples
        samples = await self.db.get_all_samples()
        results = []

        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, len(samples), sample.sample_id)

            # Get ground truth
            gt = await self.db.get_ground_truth(sample.sample_id)
            if not gt:
                logger.warning(f"No ground truth for sample {sample.sample_id}")
                continue

            # Get transcription
            transcription = await self.db.get_transcription(
                sample.sample_id, service_name
            )
            if not transcription:
                logger.warning(
                    f"No transcription for sample {sample.sample_id} "
                    f"from {service_name.value}"
                )
                continue

            # Calculate metrics
            metrics = self.evaluate(
                sample_id=sample.sample_id,
                service_name=service_name,
                reference=gt.text,
                hypothesis=transcription.transcribed_text,
            )

            # Store metrics
            await self.db.insert_metrics(metrics)
            results.append(metrics)

            logger.debug(
                f"[{i+1}/{len(samples)}] {sample.sample_id}: "
                f"WER={metrics.wer:.2%}, CER={metrics.cer:.2%}"
            )

        return results


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Convenience function to calculate WER.

    Args:
        reference: Ground truth text
        hypothesis: Transcription to evaluate

    Returns:
        WER as a float
    """
    calculator = WERCalculator()
    result = calculator.calculate_wer(reference, hypothesis)
    return result["wer"]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Convenience function to calculate CER.

    Args:
        reference: Ground truth text
        hypothesis: Transcription to evaluate

    Returns:
        CER as a float
    """
    calculator = WERCalculator()
    return calculator.calculate_cer(reference, hypothesis)
