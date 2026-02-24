"""Plotting helpers for sample size optimization.

Extracted from optimize_sample_size.py to keep file under 300 lines.
"""
from wisent.core.constants import VIZ_DPI


def save_optimization_plots(args, results, optimal_size):
    """Save performance and training time plots for sample size optimization.

    Args:
        args: Command arguments with task, model, etc.
        results: List of result dicts with sample_size, accuracy, f1_score, time
        optimal_size: The determined optimal sample size
    """
    import matplotlib.pyplot as plt

    try:
        from wisent_plots import LineChart
    except ImportError:
        LineChart = None

    if LineChart is None:
        print("   wisent_plots not available, skipping plot generation")
        return

    plot_path_svg = f"sample_size_optimization_{args.task}_{args.model.replace('/', '_')}.svg"
    plot_path_png = f"sample_size_optimization_{args.task}_{args.model.replace('/', '_')}.png"

    # Extract data for plotting
    x_data = [r['sample_size'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    times = [r['time'] for r in results]

    # Create performance plot (Accuracy and F1)
    chart1 = LineChart(style=1, figsize=(10, 6), show_markers=True)
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    chart1.plot_multiple(
        x=x_data,
        y_series=[accuracies, f1_scores],
        labels=['Accuracy', 'F1 Score'],
        title=f'Performance vs Sample Size\n{args.model} on {args.task}',
        xlabel='Training Sample Size',
        ylabel='Score',
        fig=fig1,
        ax=ax1,
        output_format='png'
    )

    # Add vertical line for optimal size
    ax1.axvline(x=optimal_size, color='#2ecc71', linestyle='--', linewidth=2,
               label=f'Optimal: {optimal_size}', alpha=0.7)
    ax1.legend()

    # Save performance plot
    fig1.savefig(plot_path_svg.replace('.svg', '_performance.svg'),
                format='svg', bbox_inches='tight')
    fig1.savefig(plot_path_png.replace('.png', '_performance.png'),
                dpi=VIZ_DPI, bbox_inches='tight')
    plt.close(fig1)

    # Create training time plot
    chart2 = LineChart(style=1, figsize=(10, 6), show_markers=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    chart2.plot_multiple(
        x=x_data,
        y_series=[times],
        labels=['Training Time'],
        colors=['#27ae60'],
        title=f'Training Time vs Sample Size\n{args.model} on {args.task}',
        xlabel='Training Sample Size',
        ylabel='Time (seconds)',
        fig=fig2,
        ax=ax2,
        output_format='png'
    )

    # Save time plot
    fig2.savefig(plot_path_svg.replace('.svg', '_time.svg'),
                format='svg', bbox_inches='tight')
    fig2.savefig(plot_path_png.replace('.png', '_time.png'),
                dpi=VIZ_DPI, bbox_inches='tight')
    plt.close(fig2)

    print(f"   Performance plot saved to:")
    print(f"   SVG: {plot_path_svg.replace('.svg', '_performance.svg')}")
    print(f"   PNG: {plot_path_png.replace('.png', '_performance.png')}")
    print(f"   Training time plot saved to:")
    print(f"   SVG: {plot_path_svg.replace('.svg', '_time.svg')}")
    print(f"   PNG: {plot_path_png.replace('.png', '_time.png')}\n")
