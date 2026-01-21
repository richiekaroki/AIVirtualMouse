"""
Motion Analyzer - Offline Analysis & Visualization Tools

Analyzes recorded motion sequences from JSON files and generates:
- Trajectory plots (hand path over time)
- Primitive transition diagrams
- Velocity profiles
- Statistical summaries
- Comparison between gestures

This is critical for:
- Understanding gesture patterns
- Debugging motion capture
- Validating semantic accuracy
- Communicating findings to non-technical stakeholders
- Building documentation

Usage:
    python MotionAnalyzer.py motion_data/gesture.json
    python MotionAnalyzer.py motion_data/gesture1.json motion_data/gesture2.json --compare

Author: Richard Kabue Karoki
Created: January 2026
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import argparse

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


class MotionAnalyzer:
    """
    Comprehensive analysis toolkit for recorded motion sequences.
    
    Loads JSON files created by MotionDescriptor and provides
    various visualization and analysis methods.
    """
    
    def __init__(self, json_file: str):
        """
        Initialize analyzer with a motion data JSON file.
        
        Args:
            json_file: Path to JSON file containing motion sequence
        """
        self.json_file = json_file
        self.data = self._load_data()
        
        if not self.data:
            raise ValueError(f"Failed to load data from {json_file}")
        
        self.gesture_name = self.data['metadata']['gesture_name']
        self.frames = self.data['frames']
        self.duration = self.data['metadata']['duration_seconds']
        self.fps = self.data['metadata']['average_fps']
    
    def _load_data(self) -> Optional[Dict]:
        """Load and validate JSON data"""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if 'metadata' not in data or 'frames' not in data:
                print(f"✗ Invalid JSON structure in {self.json_file}")
                return None
            
            return data
        except FileNotFoundError:
            print(f"✗ File not found: {self.json_file}")
            return None
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            return None
    
    def print_summary(self):
        """Print statistical summary of the gesture"""
        print("\n" + "="*70)
        print(f"Motion Analysis: {self.gesture_name}")
        print("="*70)
        
        metadata = self.data['metadata']
        
        print(f"\nMetadata:")
        print(f"  Recorded: {metadata['recorded_at']}")
        print(f"  Duration: {metadata['duration_seconds']:.2f} seconds")
        print(f"  Frames: {metadata['total_frames']}")
        print(f"  Average FPS: {metadata['average_fps']:.1f}")
        
        # Primitive analysis
        primitives = [f['primitive'] for f in self.frames]
        unique_primitives = set(primitives)
        
        print(f"\nPrimitive Analysis:")
        print(f"  Unique primitives: {len(unique_primitives)}")
        
        for primitive in sorted(unique_primitives):
            count = primitives.count(primitive)
            percentage = (count / len(primitives)) * 100
            print(f"    {primitive:20s}: {count:4d} frames ({percentage:5.1f}%)")
        
        # Velocity analysis
        velocities = [f['velocity']['magnitude'] for f in self.frames 
                     if f['velocity'] is not None]
        
        if velocities:
            print(f"\nVelocity Analysis:")
            print(f"  Mean velocity: {np.mean(velocities):.2f} px/s")
            print(f"  Max velocity: {np.max(velocities):.2f} px/s")
            print(f"  Min velocity: {np.min(velocities):.2f} px/s")
            print(f"  Std deviation: {np.std(velocities):.2f} px/s")
        
        # Hand openness analysis
        openness_values = [f['features']['hand_openness'] for f in self.frames]
        
        print(f"\nHand Openness:")
        print(f"  Mean: {np.mean(openness_values):.2f}")
        print(f"  Range: {np.min(openness_values):.2f} - {np.max(openness_values):.2f}")
        
        print("\n" + "="*70 + "\n")
    
    def plot_trajectory(self, landmark: str = 'index_tip', save_path: Optional[str] = None):
        """
        Plot 2D trajectory of a specific landmark over time.
        
        Shows the path the hand took during the gesture, which is critical
        for understanding the Movement parameter of sign language.
        
        Args:
            landmark: Which landmark to track ('index_tip', 'palm_center', etc.)
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract coordinates
        if landmark == 'palm_center':
            x_coords = [f['features']['palm_center']['x'] for f in self.frames]
            y_coords = [f['features']['palm_center']['y'] for f in self.frames]
        else:
            x_coords = [f['landmarks'][landmark]['x'] for f in self.frames]
            y_coords = [f['landmarks'][landmark]['y'] for f in self.frames]
        
        # Left plot: 2D trajectory
        ax1.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2)
        ax1.scatter(x_coords[0], y_coords[0], c='green', s=200, 
                   marker='o', label='Start', zorder=5, edgecolors='darkgreen', linewidths=2)
        ax1.scatter(x_coords[-1], y_coords[-1], c='red', s=200, 
                   marker='X', label='End', zorder=5, edgecolors='darkred', linewidths=2)
        
        # Add direction arrows
        arrow_step = max(1, len(x_coords) // 10)
        for i in range(0, len(x_coords) - arrow_step, arrow_step):
            dx = x_coords[i + arrow_step] - x_coords[i]
            dy = y_coords[i + arrow_step] - y_coords[i]
            ax1.arrow(x_coords[i], y_coords[i], dx, dy, 
                     head_width=10, head_length=15, fc='blue', ec='blue', alpha=0.4)
        
        ax1.set_xlabel('X Position (pixels)', fontsize=12)
        ax1.set_ylabel('Y Position (pixels)', fontsize=12)
        ax1.set_title(f'2D Trajectory: {self.gesture_name} ({landmark})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Match screen coordinates
        
        # Right plot: Position over time
        timestamps = [f['relative_time'] for f in self.frames]
        
        ax2.plot(timestamps, x_coords, 'r-', label='X position', linewidth=2)
        ax2.plot(timestamps, y_coords, 'b-', label='Y position', linewidth=2)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Position (pixels)', fontsize=12)
        ax2.set_title('Position Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved trajectory plot to {save_path}")
        
        plt.show()
    
    def plot_primitives_timeline(self, save_path: Optional[str] = None):
        """
        Visualize how primitives change over time.
        
        Critical for sign language: shows the temporal structure of gestures.
        Transitions between primitives often carry semantic meaning.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Extract primitives and timestamps
        primitives = [f['primitive'] for f in self.frames]
        timestamps = [f['relative_time'] for f in self.frames]
        
        # Create numeric mapping for primitives
        unique_primitives = sorted(set(primitives))
        primitive_map = {prim: i for i, prim in enumerate(unique_primitives)}
        primitive_values = [primitive_map[p] for p in primitives]
        
        # Color mapping
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_primitives)))
        color_map = {prim: colors[i] for i, prim in enumerate(unique_primitives)}
        
        # Plot with color segments
        for i in range(len(timestamps) - 1):
            ax.plot(timestamps[i:i+2], primitive_values[i:i+2], 
                   color=color_map[primitives[i]], linewidth=3)
        
        # Scatter points
        for prim in unique_primitives:
            mask = [p == prim for p in primitives]
            t_filtered = [t for t, m in zip(timestamps, mask) if m]
            v_filtered = [v for v, m in zip(primitive_values, mask) if m]
            ax.scatter(t_filtered, v_filtered, c=[color_map[prim]], 
                      s=50, label=prim, edgecolors='black', linewidths=0.5)
        
        ax.set_yticks(range(len(unique_primitives)))
        ax.set_yticklabels(unique_primitives)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Motion Primitive', fontsize=12)
        ax.set_title(f'Primitive Timeline: {self.gesture_name}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved primitive timeline to {save_path}")
        
        plt.show()
    
    def plot_velocity_profile(self, save_path: Optional[str] = None):
        """
        Plot velocity over time.
        
        Movement dynamics are crucial in sign language. Fast vs. slow
        movement can change meaning. This visualization helps analyze
        motion speed patterns.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Extract velocity data
        timestamps = []
        velocities = []
        vx_list = []
        vy_list = []
        
        for frame in self.frames:
            if frame['velocity'] is not None:
                timestamps.append(frame['relative_time'])
                velocities.append(frame['velocity']['magnitude'])
                vx_list.append(frame['velocity']['vx'])
                vy_list.append(frame['velocity']['vy'])
        
        if not velocities:
            print("✗ No velocity data available")
            return
        
        # Top plot: Velocity magnitude
        ax1.plot(timestamps, velocities, 'b-', linewidth=2, label='Velocity magnitude')
        ax1.fill_between(timestamps, velocities, alpha=0.3)
        
        # Add mean line
        mean_vel = np.mean(velocities)
        ax1.axhline(y=mean_vel, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_vel:.1f} px/s')
        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Velocity (px/s)', fontsize=12)
        ax1.set_title(f'Velocity Profile: {self.gesture_name}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Velocity components
        ax2.plot(timestamps, vx_list, 'r-', linewidth=2, label='X velocity', alpha=0.7)
        ax2.plot(timestamps, vy_list, 'g-', linewidth=2, label='Y velocity', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Velocity Component (px/s)', fontsize=12)
        ax2.set_title('Velocity Components (X and Y)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved velocity profile to {save_path}")
        
        plt.show()
    
    def plot_hand_openness(self, save_path: Optional[str] = None):
        """
        Plot hand openness over time.
        
        Shows how the hand transitions between open and closed states,
        which is part of the Handshape parameter in sign language.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        timestamps = [f['relative_time'] for f in self.frames]
        openness = [f['features']['hand_openness'] for f in self.frames]
        
        # Plot with filled area
        ax.plot(timestamps, openness, 'purple', linewidth=2.5)
        ax.fill_between(timestamps, openness, alpha=0.3, color='purple')
        
        # Add reference lines
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Fully open (1.0)')
        ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Closed (0.0)')
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Half open (0.5)')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Hand Openness (0=closed, 1=open)', fontsize=12)
        ax.set_title(f'Hand Openness Over Time: {self.gesture_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved hand openness plot to {save_path}")
        
        plt.show()
    
    def plot_primitive_distribution(self, save_path: Optional[str] = None):
        """
        Bar chart showing distribution of primitives.
        
        Helps understand which primitives dominate the gesture and
        how much time is spent in each state.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count primitives
        primitives = [f['primitive'] for f in self.frames]
        unique_primitives = sorted(set(primitives))
        counts = [primitives.count(p) for p in unique_primitives]
        percentages = [(c / len(primitives)) * 100 for c in counts]
        
        # Bar chart
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_primitives)))
        bars = ax1.bar(range(len(unique_primitives)), counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(unique_primitives)))
        ax1.set_xticklabels(unique_primitives, rotation=45, ha='right')
        ax1.set_xlabel('Primitive', fontsize=12)
        ax1.set_ylabel('Frame Count', fontsize=12)
        ax1.set_title(f'Primitive Distribution: {self.gesture_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Pie chart
        ax2.pie(percentages, labels=unique_primitives, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved primitive distribution to {save_path}")
        
        plt.show()
    
    def generate_all_plots(self, output_dir: Optional[str] = None):
        """
        Generate all available plots for this gesture.
        
        Useful for comprehensive analysis and documentation.
        
        Args:
            output_dir: Directory to save plots (if None, just displays)
        """
        print(f"\nGenerating all plots for: {self.gesture_name}")
        print("="*70)
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            base_name = Path(self.json_file).stem
        
        # Print summary
        self.print_summary()
        
        # Generate plots
        print("Generating trajectory plot...")
        save_path = f"{output_dir}/{base_name}_trajectory.png" if output_dir else None
        self.plot_trajectory(save_path=save_path)
        
        print("Generating primitive timeline...")
        save_path = f"{output_dir}/{base_name}_primitives.png" if output_dir else None
        self.plot_primitives_timeline(save_path=save_path)
        
        print("Generating velocity profile...")
        save_path = f"{output_dir}/{base_name}_velocity.png" if output_dir else None
        self.plot_velocity_profile(save_path=save_path)
        
        print("Generating hand openness plot...")
        save_path = f"{output_dir}/{base_name}_openness.png" if output_dir else None
        self.plot_hand_openness(save_path=save_path)
        
        print("Generating primitive distribution...")
        save_path = f"{output_dir}/{base_name}_distribution.png" if output_dir else None
        self.plot_primitive_distribution(save_path=save_path)
        
        print("\n✓ All plots generated successfully!")
        print("="*70 + "\n")


class GestureComparator:
    """
    Compare two gesture sequences side-by-side.
    
    Useful for:
    - Comparing different attempts at the same sign
    - Analyzing variation between signers
    - Validating consistency
    """
    
    def __init__(self, json_file1: str, json_file2: str):
        self.analyzer1 = MotionAnalyzer(json_file1)
        self.analyzer2 = MotionAnalyzer(json_file2)
    
    def compare_trajectories(self, landmark: str = 'index_tip', save_path: Optional[str] = None):
        """Compare trajectories of two gestures"""
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
            
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
        
        plt.suptitle(f'Trajectory Comparison: {landmark}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison to {save_path}")
        
        plt.show()
    
    def compare_statistics(self):
        """Print side-by-side statistics"""
        print("\n" + "="*70)
        print("Gesture Comparison")
        print("="*70)
        
        print(f"\nGesture 1: {self.analyzer1.gesture_name}")
        print(f"  Duration: {self.analyzer1.duration:.2f}s")
        print(f"  Frames: {len(self.analyzer1.frames)}")
        print(f"  FPS: {self.analyzer1.fps:.1f}")
        
        print(f"\nGesture 2: {self.analyzer2.gesture_name}")
        print(f"  Duration: {self.analyzer2.duration:.2f}s")
        print(f"  Frames: {len(self.analyzer2.frames)}")
        print(f"  FPS: {self.analyzer2.fps:.1f}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Command-line interface for MotionAnalyzer"""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize motion capture data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single gesture
  python MotionAnalyzer.py motion_data/wave_123.json
  
  # Generate all plots and save
  python MotionAnalyzer.py motion_data/wave_123.json --output plots/
  
  # Compare two gestures
  python MotionAnalyzer.py motion_data/wave1.json motion_data/wave2.json --compare
  
  # Show specific plot only
  python MotionAnalyzer.py motion_data/wave_123.json --plot trajectory
        """
    )
    
    parser.add_argument('files', nargs='+', help='JSON file(s) to analyze')
    parser.add_argument('--output', '-o', help='Output directory for saving plots')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare two gestures')
    parser.add_argument('--plot', choices=['trajectory', 'primitives', 'velocity', 'openness', 'distribution'],
                       help='Generate specific plot only')
    
    args = parser.parse_args()
    
    # Comparison mode
    if args.compare:
        if len(args.files) != 2:
            print("✗ Comparison requires exactly 2 JSON files")
            sys.exit(1)
        
        comparator = GestureComparator(args.files[0], args.files[1])
        comparator.compare_statistics()
        comparator.compare_trajectories(save_path=f"{args.output}/comparison.png" if args.output else None)
        return
    
    # Single file analysis
    if len(args.files) > 1:
        print("✗ Multiple files provided but --compare not specified")
        sys.exit(1)
    
    analyzer = MotionAnalyzer(args.files[0])
    
    if args.plot:
        # Generate specific plot
        if args.plot == 'trajectory':
            analyzer.plot_trajectory(save_path=f"{args.output}/trajectory.png" if args.output else None)
        elif args.plot == 'primitives':
            analyzer.plot_primitives_timeline(save_path=f"{args.output}/primitives.png" if args.output else None)
        elif args.plot == 'velocity':
            analyzer.plot_velocity_profile(save_path=f"{args.output}/velocity.png" if args.output else None)
        elif args.plot == 'openness':
            analyzer.plot_hand_openness(save_path=f"{args.output}/openness.png" if args.output else None)
        elif args.plot == 'distribution':
            analyzer.plot_primitive_distribution(save_path=f"{args.output}/distribution.png" if args.output else None)
    else:
        # Generate all plots
        analyzer.generate_all_plots(output_dir=args.output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nMotion Analyzer - Analysis & Visualization Tools")
        print("="*70)
        print("\nUsage: python MotionAnalyzer.py <json_file> [options]")
        print("\nFor help: python MotionAnalyzer.py --help")
        print("="*70 + "\n")
    else:
        main()