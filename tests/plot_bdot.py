#!/usr/bin/env python3
"""
Plot angular velocity from bdot convergence test.
Reads tests/out/bdot/angular_velocity.csv and generates plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Read the CSV file
    csv_path = Path("out/bdot/angular_velocity.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Read CSV data (no header)
    # Columns: time, omega_x, omega_y, omega_z, omega_norm
    df = pd.read_csv(csv_path, header=None, names=["time", "omega_x", "omega_y", "omega_z", "omega_norm"])

    print(f"CSV shape: {df.shape}")
    print(f"Time range: {df['time'].min():.2f} - {df['time'].max():.2f} sec")

    # Create output directory
    output_dir = Path("out/bdot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create combined plot with all 3 components
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    component_names = ["omega_x", "omega_y", "omega_z"]
    labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    for i, (comp_name, label) in enumerate(zip(component_names, labels)):
        ax = axes[i]
        ax.plot(df["time"], df[comp_name], linewidth=1.5, alpha=0.8, color='blue')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [rad/s]")
        ax.set_title(f"Angular Velocity: {label}")
        ax.grid(True, alpha=0.3)

    # Norm plot
    ax = axes[3]
    ax.plot(df["time"], df["omega_norm"], linewidth=2, alpha=0.8, color='red')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$|\omega|$ [rad/s]")
    ax.set_title("Angular Velocity Norm")
    ax.grid(True, alpha=0.3)

    plt.suptitle("B-dot Controller: Angular Velocity Convergence", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_velocity.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nFinal angular velocity:")
    print(f"  omega_x: {df['omega_x'].iloc[-1]:.6f} rad/s")
    print(f"  omega_y: {df['omega_y'].iloc[-1]:.6f} rad/s")
    print(f"  omega_z: {df['omega_z'].iloc[-1]:.6f} rad/s")
    print(f"  norm:    {df['omega_norm'].iloc[-1]:.6f} rad/s")


if __name__ == "__main__":
    main()
