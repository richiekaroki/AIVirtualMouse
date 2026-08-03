"""
Motion Analyzer - Offline Analysis & Visualization Tools

Analyzes recorded motion sequences from JSON files and generates:
- Trajectory plots
- Primitive transition diagrams
- Velocity profiles
- Statistical summaries
- Gesture comparisons
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import argparse
import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')


class MotionAnalyzer:
    """Comprehensive analysis toolkit for recorded motion sequences."""

    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = self._load_data()

        if not self.data:
            raise ValueError(f"Failed to load data from {json_file}")

        self.gesture_name = self.data['metadata']['gesture_name']
        self.frames = self.data['frames']
        self.duration = self.data['metadata']['duration_seconds']
        self.fps = self.data['metadata']['average_fps']

    def _load_data(self) -> Optional[Dict]:
        """Load and validate JSON data."""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            if 'metadata' not in data or 'frames' not in data:
                logger.error("Invalid JSON structure in %s", self.json_file)
                return None
            return data
        except FileNotFoundError:
            logger.error("File not found: %s", self.json_file)
            return None
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            return None

    def print_summary(self):
        """Print statistical summary of the gesture."""
        print("\n" + "=" * 70)
        print(f"Motion Analysis: {self.gesture_name}")
        print("=" * 70)

        metadata = self.data['metadata']
        print(f"\nMetadata:")
        print(f"  Recorded: {metadata['recorded_at']}")
        print(f"  Duration: {metadata['duration_seconds']:.2f} seconds")
        print(f"  Frames: {metadata['total_frames']}")
        print(f"  Average FPS: {metadata['average_fps']:.1f}")

        primitives = [f['primitive'] for f in self.frames]
        unique_primitives = set(primitives)
        print(f"\nPrimitive Analysis:")
        print(f"  Unique primitives: {len(unique_primitives)}")
        for primitive in sorted(unique_primitives):
            count = primitives.count(primitive)
            percentage = (count / len(primitives)) * 100
            print(f"    {primitive:20s}: {count:4d} frames ({percentage:5.1f}%)")

        velocities = [f['velocity']['magnitude'] for f in self.frames
                      if f['velocity'] is not None]
        if velocities:
            print(f"\nVelocity Analysis:")
            print(f"  Mean velocity: {np.mean(velocities):.2f} px/s")
            print(f"  Max velocity: {np.max(velocities):.2f} px/s")
            print(f"  Min velocity: {np.min(velocities):.2f} px/s")

        openness_values = [f['features']['hand_openness'] for f in self.frames]
        print(f"\nHand Openness:")
        print(f"  Mean: {np.mean(openness_values):.2f}")
        print(f"  Range: {np.min(openness_values):.2f} - {np.max(openness_values):.2f}")
        print("\n" + "=" * 70 + "\n")

    def plot_trajectory(self, landmark: str = 'index_tip', save_path: Optional[str] = None):
        """Plot 2D trajectory of a specific landmark over time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if landmark == 'palm_center':
            x_coords = [f['features']['palm_center']['x'] for f in self.frames]
            y_coords = [f['features']['palm_center']['y'] for f in self.frames]
        else:
            x_coords = [f['landmarks'][landmark]['x'] for f in self.frames]
            y_coords = [f['landmarks'][landmark]['y'] for f in self.frames]

        ax1.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2)
        ax1.scatter(x_coords[0], y_coords[0], c='green', s=200,
                    marker='o', label='Start', zorder=5, edgecolors='darkgreen', linewidths=2)
        ax1.scatter(x_coords[-1], y_coords[-1], c='red', s=200,
                    marker='X', label='End', zorder=5, edgecolors='darkred', linewidths=2)

        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title(f'2D Trajectory: {self.gesture_name} ({landmark})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        timestamps = [f['relative_time'] for f in self.frames]
        ax2.plot(timestamps, x_coords, 'r-', label='X position', linewidth=2)
        ax2.plot(timestamps, y_coords, 'b-', label='Y position', linewidth=2)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Position (pixels)')
        ax2.set_title('Position Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_primitives_timeline(self, save_path: Optional[str] = None):
        """Visualize how primitives change over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        primitives = [f['primitive'] for f in self.frames]
        timestamps = [f['relative_time'] for f in self.frames]

        unique_primitives = sorted(set(primitives))
        primitive_map = {prim: i for i, prim in enumerate(unique_primitives)}
        primitive_values = [primitive_map[p] for p in primitives]

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_primitives)))
        color_map = {prim: colors[i] for i, prim in enumerate(unique_primitives)}

        for i in range(len(timestamps) - 1):
            ax.plot(timestamps[i:i+2], primitive_values[i:i+2],
                    color=color_map[primitives[i]], linewidth=3)

        for prim in unique_primitives:
            mask = [p == prim for p in primitives]
            t_filtered = [t for t, m in zip(timestamps, mask) if m]
            v_filtered = [v for v, m in zip(primitive_values, mask) if m]
            ax.scatter(t_filtered, v_filtered, c=[color_map[prim]],
                       s=50, label=prim, edgecolors='black', linewidths=0.5)

        ax.set_yticks(range(len(unique_primitives)))
        ax.set_yticklabels(unique_primitives)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Motion Primitive')
        ax.set_title(f'Primitive Timeline: {self.gesture_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_velocity_profile(self, save_path: Optional[str] = None):
        """Plot velocity over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        timestamps, velocities, vx_list, vy_list = [], [], [], []
        for frame in self.frames:
            if frame['velocity'] is not None:
                timestamps.append(frame['relative_time'])
                velocities.append(frame['velocity']['magnitude'])
                vx_list.append(frame['velocity']['vx'])
                vy_list.append(frame['velocity']['vy'])

        if not velocities:
            logger.warning("No velocity data available")
            return

        ax1.plot(timestamps, velocities, 'b-', linewidth=2, label='Velocity magnitude')
        ax1.fill_between(timestamps, velocities, alpha=0.3)
        mean_vel = np.mean(velocities)
        ax1.axhline(y=mean_vel, color='r', linestyle='--', linewidth=2,
                     label=f'Mean: {mean_vel:.1f} px/s')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Velocity (px/s)')
        ax1.set_title(f'Velocity Profile: {self.gesture_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(timestamps, vx_list, 'r-', linewidth=2, label='X velocity', alpha=0.7)
        ax2.plot(timestamps, vy_list, 'g-', linewidth=2, label='Y velocity', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Velocity Component (px/s)')
        ax2.set_title('Velocity Components (X and Y)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_hand_openness(self, save_path: Optional[str] = None):
        """Plot hand openness over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        timestamps = [f['relative_time'] for f in self.frames]
        openness = [f['features']['hand_openness'] for f in self.frames]

        ax.plot(timestamps, openness, 'purple', linewidth=2.5)
        ax.fill_between(timestamps, openness, alpha=0.3, color='purple')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Fully open')
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Closed')
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Half open')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Hand Openness (0=closed, 1=open)')
        ax.set_title(f'Hand Openness Over Time: {self.gesture_name}')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_primitive_distribution(self, save_path: Optional[str] = None):
        """Bar chart showing distribution of primitives."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        primitives = [f['primitive'] for f in self.frames]
        unique_primitives = sorted(set(primitives))
        counts = [primitives.count(p) for p in unique_primitives]
        percentages = [(c / len(primitives)) * 100 for c in counts]

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_primitives)))
        bars = ax1.bar(range(len(unique_primitives)), counts, color=colors,
                       edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(unique_primitives)))
        ax1.set_xticklabels(unique_primitives, rotation=45, ha='right')
        ax1.set_xlabel('Primitive')
        ax1.set_ylabel('Frame Count')
        ax1.set_title(f'Primitive Distribution: {self.gesture_name}')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{count}', ha='center', va='bottom', fontsize=9)

        ax2.pie(percentages, labels=unique_primitives, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Percentage Distribution')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_all_plots(self, output_dir: Optional[str] = None):
        """Generate all available plots for this gesture."""
        print(f"\nGenerating all plots for: {self.gesture_name}")
        print("=" * 70)

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            base_name = Path(self.json_file).stem

        self.print_summary()

        plots = [
            ("trajectory", self.plot_trajectory),
            ("primitives", self.plot_primitives_timeline),
            ("velocity", self.plot_velocity_profile),
            ("openness", self.plot_hand_openness),
            ("distribution", self.plot_primitive_distribution),
        ]

        for name, plot_func in plots:
            print(f"Generating {name} plot...")
            save_path = str(Path(output_dir) / f"{base_name}_{name}.png") if output_dir else None
            plot_func(save_path=save_path)

        print("\nAll plots generated successfully!")
        print("=" * 70 + "\n")


class GestureComparator:
    """Compare two gesture sequences side-by-side."""

    def __init__(self, json_file1: str, json_file2: str):
        self.analyzer1 = MotionAnalyzer(json_file1)
        self.analyzer2 = MotionAnalyzer(json_file2)

    def compare_trajectories(self, landmark: str = 'index_tip', save_path: Optional[str] = None):
        """Compare trajectories of two gestures."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for ax, analyzer, color, label in [
            (ax1, self.analyzer1, 'blue', self.analyzer1.gesture_name),
            (ax2, self.analyzer2, 'red', self.analyzer2.gesture_name)
        ]:
            frames = analyzer.frames
            if landmark == 'palm_center':
                x_coords = [f['features']['palm_center']['x'] for f in frames]
                y_coords = [f['features']['palm_center']['y'] for f in frames]
            else:
                x_coords = [f['landmarks'][landmark]['x'] for f in frames]
                y_coords = [f['landmarks'][landmark]['y'] for f in frames]

            ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=2, label=label)
            ax.scatter(x_coords[0], y_coords[0], c='green', s=150, marker='o', label='Start', zorder=5)
            ax.scatter(x_coords[-1], y_coords[-1], c='red', s=150, marker='X', label='End', zorder=5)
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()

        plt.suptitle(f'Trajectory Comparison: {landmark}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def compare_statistics(self):
        """Print side-by-side statistics."""
        print("\n" + "=" * 70)
        print("Gesture Comparison")
        print("=" * 70)

        for analyzer in [self.analyzer1, self.analyzer2]:
            print(f"\nGesture: {analyzer.gesture_name}")
            print(f"  Duration: {analyzer.duration:.2f}s")
            print(f"  Frames: {len(analyzer.frames)}")
            print(f"  FPS: {analyzer.fps:.1f}")

        print("\n" + "=" * 70 + "\n")


def main():
    """Command-line interface for MotionAnalyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize motion capture data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hand_motion.analyzer motion_data/wave_123.json
  python -m hand_motion.analyzer motion_data/wave_123.json --output plots/
  python -m hand_motion.analyzer wave1.json wave2.json --compare
  python -m hand_motion.analyzer motion_data/wave_123.json --plot trajectory
        """
    )

    parser.add_argument('files', nargs='+', help='JSON file(s) to analyze')
    parser.add_argument('--output', '-o', help='Output directory for saving plots')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare two gestures')
    parser.add_argument('--plot', choices=['trajectory', 'primitives', 'velocity', 'openness', 'distribution'],
                        help='Generate specific plot only')

    args = parser.parse_args()

    if args.compare:
        if len(args.files) != 2:
            print("Comparison requires exactly 2 JSON files")
            sys.exit(1)
        comparator = GestureComparator(args.files[0], args.files[1])
        comparator.compare_statistics()
        comparator.compare_trajectories(save_path=str(Path(args.output) / "comparison.png") if args.output else None)
        return

    if len(args.files) > 1:
        print("Multiple files provided but --compare not specified")
        sys.exit(1)

    analyzer = MotionAnalyzer(args.files[0])

    if args.plot:
        plot_methods = {
            'trajectory': analyzer.plot_trajectory,
            'primitives': analyzer.plot_primitives_timeline,
            'velocity': analyzer.plot_velocity_profile,
            'openness': analyzer.plot_hand_openness,
            'distribution': analyzer.plot_primitive_distribution,
        }
        save_path = str(Path(args.output) / f"{args.plot}.png") if args.output else None
        plot_methods[args.plot](save_path=save_path)
    else:
        analyzer.generate_all_plots(output_dir=args.output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nMotion Analyzer - Analysis & Visualization Tools")
        print("=" * 70)
        print("\nUsage: python -m hand_motion.analyzer <json_file> [options]")
        print("\nFor help: python -m hand_motion.analyzer --help")
        print("=" * 70 + "\n")
    else:
        main()
