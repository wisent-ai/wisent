import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Create results directory
results_dir = Path("/workspace/results")
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating results in: {results_dir}")

# Generate some data and create a visualization
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.exp(-x/10)

# Create a nice plot
plt.figure(figsize=(12, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2, color='#2E86AB')
plt.plot(x, y2, label='cos(x)', linewidth=2, color='#A23B72')
plt.plot(x, y3, label='sin(x) * exp(-x/10)', linewidth=2, color='#F18F01')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Sample Mathematical Functions', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)

# Save the plot
plot_path = results_dir / "sample_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to: {plot_path}")

# Create JSON metadata
metadata = {
    "timestamp": datetime.now().isoformat(),
    "plot_info": {
        "filename": "sample_plot.png",
        "description": "Visualization of sin(x), cos(x), and damped sine wave",
        "x_range": [float(x.min()), float(x.max())],
        "num_points": len(x),
    },
    "statistics": {
        "sin_mean": float(np.mean(y1)),
        "sin_std": float(np.std(y1)),
        "cos_mean": float(np.mean(y2)),
        "cos_std": float(np.std(y2)),
        "damped_sin_mean": float(np.mean(y3)),
        "damped_sin_std": float(np.std(y3)),
    },
    "environment": {
        "results_directory": str(results_dir),
        "script_location": "wisent-guard/tests/test_runpod/test1.py",
    }
}

# Save JSON metadata
json_path = results_dir / "metadata.json"
with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved metadata to: {json_path}")
print("\nAll results saved successfully!")