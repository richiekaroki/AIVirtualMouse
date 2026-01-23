"""
Dataset Analysis Script

Analyzes all recorded gestures and generates:
- Quality report
- Best attempts selection
- Visualization plots
- Dataset summary

Usage:
    python analyze_dataset.py
"""

import json
import os
from pathlib import Path
from MotionAnalyzer import MotionAnalyzer
import matplotlib.pyplot as plt

def load_manifest():
    """Load recording manifest"""
    manifest_path = 'motion_data/recording_manifest.json'
    if not os.path.exists(manifest_path):
        print("✗ No manifest found. Run batch_record.py first.")
        return None
    
    with open(manifest_path, 'r') as f:
        return json.load(f)

def analyze_all_recordings(manifest):
    """Analyze all recordings and generate report"""
    print("\n" + "="*70)
    print("Dataset Analysis")
    print("="*70)
    
    recordings = manifest['recordings']
    
    print(f"\nAnalyzing {len(recordings)} recordings...")
    
    # Group by gesture
    gesture_groups = {}
    for filepath in recordings:
        filename = Path(filepath).stem
        gesture_name = '_'.join(filename.split('_')[:-2])  # Remove attempt and timestamp
        
        if gesture_name not in gesture_groups:
            gesture_groups[gesture_name] = []
        gesture_groups[gesture_name].append(filepath)
    
    # Analyze each group
    results = {}
    
    for gesture_name, files in gesture_groups.items():
        print(f"\n  Analyzing '{gesture_name}' ({len(files)} attempts)...")
        
        gesture_results = []
        
        for filepath in files:
            try:
                analyzer = MotionAnalyzer(filepath)
                stats = analyzer.data['metadata']
                
                # Quality score (simple heuristic)
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
                print(f"    ✗ Error analyzing {filepath}: {e}")
        
        # Sort by quality
        gesture_results.sort(key=lambda x: x['quality_score'], reverse=True)
        results[gesture_name] = gesture_results
    
    return results

def generate_summary_report(results, manifest):
    """Generate markdown summary report"""
    
    report = []
    report.append("# Dataset Summary Report\n")
    report.append(f"**Generated:** {manifest['session_date']}\n")
    report.append(f"**Total Recordings:** {manifest['total_recordings']}\n\n")
    
    report.append("## Overview\n")
    report.append("| Gesture | Attempts | Best FPS | Best Frames | Avg Quality |\n")
    report.append("|---------|----------|----------|-------------|-------------|\n")
    
    for gesture_name, attempts in sorted(results.items()):
        if not attempts:
            continue
        
        best = attempts[0]
        avg_quality = sum(a['quality_score'] for a in attempts) / len(attempts)
        
        report.append(f"| {gesture_name} | {len(attempts)} | "
                     f"{best['fps']:.1f} | {best['frames']} | {avg_quality:.2f} |\n")
    
    report.append("\n## Detailed Analysis\n\n")
    
    for gesture_name, attempts in sorted(results.items()):
        if not attempts:
            continue
        
        report.append(f"### {gesture_name}\n\n")
        report.append(f"**Best Attempt:** `{Path(attempts[0]['filepath']).name}`\n\n")
        report.append(f"- Duration: {attempts[0]['duration']:.2f}s\n")
        report.append(f"- Frames: {attempts[0]['frames']}\n")
        report.append(f"- FPS: {attempts[0]['fps']:.1f}\n")
        report.append(f"- Primitives: {', '.join(attempts[0]['primitives'])}\n")
        report.append(f"- Quality Score: {attempts[0]['quality_score']:.2f}/1.00\n\n")
        
        if len(attempts) > 1:
            report.append("**All Attempts:**\n\n")
            for i, attempt in enumerate(attempts, 1):
                report.append(f"{i}. `{Path(attempt['filepath']).name}` - "
                            f"Quality: {attempt['quality_score']:.2f}, "
                            f"Frames: {attempt['frames']}, "
                            f"FPS: {attempt['fps']:.1f}\n")
            report.append("\n")
    
    report.append("## Quality Assessment\n\n")
    
    # Calculate statistics
    all_attempts = [attempt for attempts in results.values() for attempt in attempts]
    
    if all_attempts:
        avg_fps = sum(a['fps'] for a in all_attempts) / len(all_attempts)
        avg_frames = sum(a['frames'] for a in all_attempts) / len(all_attempts)
        avg_quality = sum(a['quality_score'] for a in all_attempts) / len(all_attempts)
        
        report.append(f"- **Average FPS:** {avg_fps:.1f}\n")
        report.append(f"- **Average Frames:** {avg_frames:.0f}\n")
        report.append(f"- **Average Quality:** {avg_quality:.2f}/1.00\n")
        report.append(f"- **High Quality (>0.9):** {sum(1 for a in all_attempts if a['quality_score'] > 0.9)} recordings\n")
        report.append(f"- **Medium Quality (0.7-0.9):** {sum(1 for a in all_attempts if 0.7 <= a['quality_score'] <= 0.9)} recordings\n")
        report.append(f"- **Low Quality (<0.7):** {sum(1 for a in all_attempts if a['quality_score'] < 0.7)} recordings\n\n")
    
    report.append("## Recommendations\n\n")
    report.append("1. Use **best attempts** (highlighted above) for documentation and demos\n")
    report.append("2. Consider re-recording any gestures with quality < 0.7\n")
    report.append("3. Generate visualizations for top 5 gestures\n")
    report.append("4. Use dataset for training ML models or animation targets\n\n")
    
    report.append("## Next Steps\n\n")
    report.append("```bash\n")
    report.append("# Generate plots for best attempts\n")
    report.append("python MotionAnalyzer.py motion_data/GESTURE_best.json --output analysis_plots/\n\n")
    report.append("# Or use analyze_dataset.py to auto-generate for all best attempts\n")
    report.append("```\n")
    
    return ''.join(report)

def generate_best_plots(results, output_dir='analysis_plots'):
    """Generate plots for best attempt of each gesture"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots for best attempts...")
    
    for gesture_name, attempts in results.items():
        if not attempts:
            continue
        
        best_attempt = attempts[0]
        filepath = best_attempt['filepath']
        
        print(f"  Plotting {gesture_name}...")
        
        try:
            analyzer = MotionAnalyzer(filepath)
            
            # Generate key plots
            base_name = f"{gesture_name}_best"
            
            analyzer.plot_trajectory(
                save_path=f"{output_dir}/{base_name}_trajectory.png"
            )
            plt.close('all')
            
            analyzer.plot_primitives_timeline(
                save_path=f"{output_dir}/{base_name}_primitives.png"
            )
            plt.close('all')
            
            print(f"    ✓ Saved plots to {output_dir}/")
        
        except Exception as e:
            print(f"    ✗ Error: {e}")

def main():
    print("\n" + "="*70)
    print("Dataset Analysis Tool")
    print("="*70)
    
    # Load manifest
    manifest = load_manifest()
    if not manifest:
        return
    
    # Analyze all recordings
    results = analyze_all_recordings(manifest)
    
    # Generate summary
    summary = generate_summary_report(results, manifest)
    
    with open('dataset_summary.md', 'w') as f:
        f.write(summary)
    
    print(f"\n✓ Summary report saved to dataset_summary.md")
    
    # Ask about plotting
    generate_plots = input("\nGenerate visualization plots for best attempts? (y/n): ").lower()
    
    if generate_plots == 'y':
        generate_best_plots(results)
        print(f"\n✓ Plots saved to analysis_plots/")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nFiles generated:")
    print("  • dataset_summary.md")
    if generate_plots == 'y':
        print("  • analysis_plots/*.png")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()