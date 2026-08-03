"""
Batch Processing Module

Multi-file analysis pipeline for processing entire datasets:
- Parallel processing support
- Progress tracking
- Aggregate statistics
- Quality filtering
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_dir: str
    output_dir: str = "batch_output"
    max_workers: int = 4
    use_multiprocessing: bool = False
    min_frames: int = 10
    min_fps: float = 20.0
    export_formats: List[str] = field(default_factory=lambda: ['csv'])
    generate_plots: bool = True
    verbose: bool = False


@dataclass
class BatchResult:
    """Result of processing a single file."""
    filename: str
    success: bool
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class BatchSummary:
    """Summary of batch processing job."""
    total_files: int
    successful: int
    failed: int
    skipped: int
    total_processing_time: float
    results: List[BatchResult]
    aggregate_stats: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """
    Multi-file batch processing pipeline.

    Features:
    - Parallel processing (threading/multiprocessing)
    - Progress tracking
    - Quality filtering
    - Aggregate statistics
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig(input_dir="motion_data")
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.processed_count = 0
        self.total_count = 0
        self.start_time = None

    def process_all(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> BatchSummary:
        """
        Process all motion files in input directory.

        Args:
            progress_callback: Callback function(processed, total, filename)

        Returns:
            BatchSummary with results
        """
        # Find all JSON files
        input_dir = Path(self.config.input_dir)
        json_files = list(input_dir.glob("*.json"))

        # Filter out manifest
        json_files = [f for f in json_files if f.name != "recording_manifest.json"]

        self.total_count = len(json_files)
        self.processed_count = 0
        self.start_time = time.time()

        logger.info("Found %d files to process", self.total_count)

        results = []

        # Process files
        if self.config.max_workers > 1:
            results = self._process_parallel(json_files, progress_callback)
        else:
            results = self._process_sequential(json_files, progress_callback)

        # Calculate summary
        total_time = time.time() - self.start_time
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        skipped = self.total_count - len(results)

        # Calculate aggregate statistics
        aggregate = self._calculate_aggregate_stats(results)

        summary = BatchSummary(
            total_files=self.total_count,
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_processing_time=total_time,
            results=results,
            aggregate_stats=aggregate
        )

        # Save summary
        self._save_summary(summary)

        return summary

    def _process_sequential(
        self,
        files: List[Path],
        progress_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """Process files sequentially."""
        results = []

        for filepath in files:
            result = self._process_single_file(filepath)
            results.append(result)

            self.processed_count += 1
            if progress_callback:
                progress_callback(self.processed_count, self.total_count, filepath.name)

        return results

    def _process_parallel(
        self,
        files: List[Path],
        progress_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """Process files in parallel."""
        results = []

        executor_class = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor

        with executor_class(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, filepath): filepath
                for filepath in files
            }

            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(BatchResult(
                        filename=filepath.name,
                        success=False,
                        error=str(e)
                    ))

                self.processed_count += 1
                if progress_callback:
                    progress_callback(self.processed_count, self.total_count, filepath.name)

        return results

    def _process_single_file(self, filepath: Path) -> BatchResult:
        """Process a single motion file."""
        start_time = time.time()

        try:
            # Load data
            with open(filepath, 'r') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            frames = data.get('frames', [])

            # Quality filtering
            if len(frames) < self.config.min_frames:
                return BatchResult(
                    filename=filepath.name,
                    success=False,
                    error=f"Too few frames: {len(frames)} < {self.config.min_frames}"
                )

            fps = metadata.get('average_fps', 0)
            if fps < self.config.min_fps:
                return BatchResult(
                    filename=filepath.name,
                    success=False,
                    error=f"Low FPS: {fps} < {self.config.min_fps}"
                )

            # Analyze
            stats = self._analyze_file(data)

            # Export formats
            output_files = {}
            if self.config.export_formats:
                output_files = self._export_file(filepath, data)

            # Generate plots if requested
            if self.config.generate_plots:
                self._generate_plots(filepath, data)

            processing_time = time.time() - start_time

            return BatchResult(
                filename=filepath.name,
                success=True,
                stats=stats,
                output_files=output_files,
                processing_time=processing_time
            )

        except Exception as e:
            return BatchResult(
                filename=filepath.name,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

    def _analyze_file(self, data: Dict) -> Dict[str, Any]:
        """Analyze a single file and return statistics."""
        metadata = data.get('metadata', {})
        frames = data.get('frames', [])

        # Primitive counts
        primitives = [f.get('primitive', 'unknown') for f in frames]
        primitive_counts = {}
        for p in primitives:
            primitive_counts[p] = primitive_counts.get(p, 0) + 1

        # Velocity stats
        velocities = [
            f.get('velocity', {}).get('magnitude', 0)
            for f in frames
            if f.get('velocity')
        ]

        # Openness stats
        openness = [
            f.get('features', {}).get('hand_openness', 0)
            for f in frames
        ]

        return {
            'gesture': metadata.get('gesture_name', 'unknown'),
            'frames': len(frames),
            'duration': metadata.get('duration_seconds', 0),
            'fps': metadata.get('average_fps', 0),
            'primitive_counts': primitive_counts,
            'unique_primitives': len(primitive_counts),
            'avg_velocity': sum(velocities) / len(velocities) if velocities else 0,
            'max_velocity': max(velocities) if velocities else 0,
            'avg_openness': sum(openness) / len(openness) if openness else 0
        }

    def _export_file(self, filepath: Path, data: Dict) -> Dict[str, str]:
        """Export file to configured formats."""
        output_files = {}

        try:
            from hand_motion.export import MotionExporter
            exporter = MotionExporter(str(self.output_dir / "exports"))

            for fmt in self.config.export_formats:
                if fmt == 'csv':
                    output_files['csv'] = exporter.export_csv(
                        str(filepath),
                        output_name=filepath.stem
                    )
                elif fmt == 'parquet':
                    try:
                        output_files['parquet'] = exporter.export_parquet(
                            str(filepath),
                            output_name=filepath.stem
                        )
                    except ImportError:
                        logger.info("Parquet export skipped for %s", filepath.name)
                elif fmt == 'coco':
                    output_files['coco'] = exporter.export_coco(
                        str(filepath),
                        output_name=filepath.stem
                    )
        except Exception as e:
            logger.error("Export failed for %s: %s", filepath.name, e)

        return output_files

    def _generate_plots(self, filepath: Path, data: Dict) -> None:
        """Generate analysis plots for a file."""
        try:
            from hand_motion.analyzer import MotionAnalyzer
            analyzer = MotionAnalyzer(str(filepath))

            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Generate trajectory plot
            analyzer.plot_trajectory(
                save_path=str(plots_dir / f"{filepath.stem}_trajectory.png")
            )
            import matplotlib.pyplot as plt
            plt.close('all')

        except Exception as e:
            logger.error("Plot generation failed for %s: %s", filepath.name, e)

    def _calculate_aggregate_stats(self, results: List[BatchResult]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all processed files."""
        successful = [r for r in results if r.success and r.stats]

        if not successful:
            return {}

        # Aggregate by gesture
        gesture_stats = {}
        for result in successful:
            gesture = result.stats['gesture']
            if gesture not in gesture_stats:
                gesture_stats[gesture] = {
                    'count': 0,
                    'total_frames': 0,
                    'total_duration': 0,
                    'avg_fps': []
                }

            gesture_stats[gesture]['count'] += 1
            gesture_stats[gesture]['total_frames'] += result.stats['frames']
            gesture_stats[gesture]['total_duration'] += result.stats['duration']
            gesture_stats[gesture]['avg_fps'].append(result.stats['fps'])

        # Calculate averages
        for gesture in gesture_stats:
            stats = gesture_stats[gesture]
            fps_list = stats.pop('avg_fps')
            stats['avg_fps'] = sum(fps_list) / len(fps_list) if fps_list else 0

        # Overall stats
        all_frames = [r.stats['frames'] for r in successful]
        all_durations = [r.stats['duration'] for r in successful]

        return {
            'total_files': len(successful),
            'total_frames': sum(all_frames),
            'total_duration': sum(all_durations),
            'avg_frames_per_file': sum(all_frames) / len(all_frames),
            'avg_duration': sum(all_durations) / len(all_durations),
            'gesture_breakdown': gesture_stats
        }

    def _save_summary(self, summary: BatchSummary) -> None:
        """Save batch processing summary."""
        summary_data = {
            'total_files': summary.total_files,
            'successful': summary.successful,
            'failed': summary.failed,
            'skipped': summary.skipped,
            'total_processing_time': summary.total_processing_time,
            'aggregate_stats': summary.aggregate_stats,
            'results': [
                {
                    'filename': r.filename,
                    'success': r.success,
                    'error': r.error,
                    'processing_time': r.processing_time
                }
                for r in summary.results
            ]
        }

        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        logger.info("Summary saved to: %s", summary_path)

    def filter_by_quality(
        self,
        min_score: float = 0.5
    ) -> List[str]:
        """
        Filter files by quality score.

        Args:
            min_score: Minimum quality score (0-1)

        Returns:
            List of filenames that pass the filter
        """
        from hand_motion.validation import validate_motion_file

        input_dir = Path(self.config.input_dir)
        json_files = list(input_dir.glob("*.json"))

        passing_files = []
        for filepath in json_files:
            if filepath.name == "recording_manifest.json":
                continue

            result = validate_motion_file(str(filepath))
            if result.score >= min_score:
                passing_files.append(filepath.name)

        return passing_files


def run_batch_processing(
    input_dir: str = "motion_data",
    output_dir: str = "batch_output",
    max_workers: int = 4,
    export_formats: Optional[List[str]] = None,
    generate_plots: bool = True
) -> BatchSummary:
    """
    Convenience function for batch processing.

    Args:
        input_dir: Input directory with motion files
        output_dir: Output directory for results
        max_workers: Number of parallel workers
        export_formats: Formats to export ('csv', 'parquet', 'coco')
        generate_plots: Whether to generate plots

    Returns:
        BatchSummary with results
    """
    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        export_formats=export_formats or ['csv'],
        generate_plots=generate_plots
    )

    processor = BatchProcessor(config)
    return processor.process_all()
