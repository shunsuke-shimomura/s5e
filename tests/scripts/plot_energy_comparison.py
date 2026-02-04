#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    output_base = Path("out")

    modes = ["pure_damping", "lyapunov", "conditional", "scheduled", "crossproduct"]
    labels = ["Pure Damping", "Lyapunov Pointing", "Conditional", "Gain-Scheduled", "Cross-Product"]
    colors = ["blue", "red", "green", "purple", "orange"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for mode, label, color in zip(modes, labels, colors):
        csv_path = output_base / f"sp_mtq_{mode}" / "energy_tracking.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path, header=None, names=[
            "time", "angle_error_deg", "kinetic_energy", "potential_energy",
            "total_energy", "omega_norm", "H_x", "H_y", "H_z"
        ])

        # Angle error
        axes[0].plot(df["time"], df["angle_error_deg"], linewidth=1.5, alpha=0.8, label=label, color=color)

        # Total energy
        axes[1].plot(df["time"], df["total_energy"], linewidth=1.5, alpha=0.8, label=label, color=color)

        # Energy change (log scale)
        energy_change = df["total_energy"] - df["total_energy"].iloc[0]
        axes[2].plot(df["time"], energy_change, linewidth=1.5, alpha=0.8, label=label, color=color)

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Angle Error [deg]")
    axes[0].set_title("Sun Pointing Error Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Lyapunov Energy V")
    axes[1].set_title("Lyapunov Function V = (1/2)ω^T I ω + k_p(1-s·z)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Energy Change ΔV")
    axes[2].set_title("Energy Change from Initial")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle("MTQ Control Mode Comparison: Energy Dissipation", fontsize=14, y=0.995)
    plt.tight_layout()

    output_dir = output_base / "sp_mtq_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "energy_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
