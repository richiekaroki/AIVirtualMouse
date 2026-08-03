"""
Dataset analysis - generates quality reports and visualizations.

Usage:
    python -m hand_motion.apps.analyze_dataset
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

from hand_motion.analyzer import MotionAnalyzer


def load_manifest():
    manifest_path = 'motion_data/recording_manifest.json'
    if not os.path.exists(manifest_path):
        print("No manifest found. Run batch_record.py first.")
        return None
    with open(manifest_path, 'r') as f:
        return json.load(f)


def analyze_all_recordings(manifest):
    print("\n" + "=" * 70)
    print("Dataset Analysis")
    print("=" * 70)

    recordings = manifest['recordings']
    print(f"\nAnalyzing {len(recordings)} recordings...")

    gesture_groups = {}
    for filepath in recordings:
        filename = Path(filepath).stem
        gesture_name = '_'.join(filename.split('_')[:-2])
        if gesture_name not in gesture_groups:
            gesture_groups[gesture_name] = []
        gesture_groups[gesture_name].append(filepath)

    results = {}
    for gesture_name, files in gesture_groups.items():
        gesture_results = []
        for filepath in files:
            try:
                analyzer = MotionAnalyzer(filepath)
                stats = analyzer.data['metadata']
                fps_score = min(stats['average_fps'] / 30.0, 1.0)
                frame_score = min(stats['total_frames'] / 90.0, 1.0)
                quality_score = (fps_score + frame_score) / 2

                gesture_results.append({
                    'filepath': filepath,
                    'duration': stats['duration_seconds'],
                    'frames': stats['total_frames'],
                    'fps': stats['average_fps'],
                    'quality_score': quality_score,
                    'primitives': stats['primitives_used']
                })
            except Exception as e:
                print(f"  Error analyzing {filepath}: {e}")

        gesture_results.sort(key=lambda x: x['quality_score'], reverse=True)
        results[gesture_name] = gesture_results

    return results


def generate_summary_report(results, manifest):
    report = ["# Dataset Summary Report\n"]
    report.append(f"**Generated:** {manifest['session_date']}\n")
    report.append(f"**Total Recordings:** {manifest['total_recordings']}\n\n")

    report.append("| Gesture | Attempts | Best FPS | Best Frames | Avg Quality |\n")
    report.append("|---------|----------|----------|-------------|-------------|\n")

    for gesture_name, attempts in sorted(results.items()):
        if not attempts:
            continue
        best = attempts[0]
        avg_quality = sum(a['quality_score'] for a in attempts) / len(attempts)
        report.append(f"| {gesture_name} | {len(attempts)} | "
                      f"{best['fps']:.1f} | {best['frames']} | {avg_quality:.2f} |\n")

    return ''.join(report)


def main():
    manifest = load_manifest()
    if not manifest:
        return

    results = analyze_all_recordings(manifest)
    summary = generate_summary_report(results, manifest)

    with open('dataset_summary.md', 'w') as f:
        f.write(summary)

    print(f"\nSummary saved to dataset_summary.md")

    generate_plots = input("\nGenerate plots? (y/n): ").lower()
    if generate_plots == 'y':
        output_dir = 'analysis_plots'
        os.makedirs(output_dir, exist_ok=True)

        for gesture_name, attempts in results.items():
            if not attempts:
                continue
            try:
                analyzer = MotionAnalyzer(attempts[0]['filepath'])
                base_name = f"{gesture_name}_best"
                analyzer.plot_trajectory(save_path=f"{output_dir}/{base_name}_trajectory.png")
                plt.close('all')
                analyzer.plot_primitives_timeline(save_path=f"{output_dir}/{base_name}_primitives.png")
                plt.close('all')
                print(f"  Plotted {gesture_name}")
            except Exception as e:
                print(f"  Error: {e}")

        print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
