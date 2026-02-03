#!/usr/bin/env python3
"""
Plot sun direction error from sun pointing ideal convergence test.
Reads tests/out/sp_ideal/sun_direction_error.csv and generates plots.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot sun pointing ideal convergence test results")
    parser.add_argument("--dir", default="out/sp_ideal", help="Output directory containing CSV files")
    args = parser.parse_args()

    # Read the CSV file
    csv_path = Path(args.dir) / "sun_direction_error.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Read CSV data (no header)
    # Columns: time, angle_error_deg, sat_z_x, sat_z_y, sat_z_z, sun_x, sun_y, sun_z,
    #          omega_x, omega_y, omega_z, omega_norm, ang_mom_x, ang_mom_y, ang_mom_z, ang_mom_norm
    df = pd.read_csv(csv_path, header=None, names=[
        "time", "angle_error_deg",
        "sat_z_x", "sat_z_y", "sat_z_z",
        "sun_x", "sun_y", "sun_z",
        "omega_x", "omega_y", "omega_z", "omega_norm",
        "ang_mom_x", "ang_mom_y", "ang_mom_z", "ang_mom_norm"
    ])

    print(f"CSV shape: {df.shape}")
    print(f"Time range: {df['time'].min():.2f} - {df['time'].max():.2f} sec")

    # Create output directory
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create main plot: Sun direction error
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Angle error plot (linear scale)
    ax = axes[0]
    ax.plot(df["time"], df["angle_error_deg"], linewidth=2, alpha=0.8, color='red')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target: 10 deg')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle Error [deg]")
    ax.set_title("Sun Pointing Error (angle between satellite Z-axis and Sun direction)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Angle error in log scale
    ax = axes[1]
    ax.semilogy(df["time"], df["angle_error_deg"], linewidth=2, alpha=0.8, color='red')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target: 10 deg')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle Error [deg]")
    ax.set_title("Sun Pointing Error (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle("Sun Pointing Ideal Controller: Convergence", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "sun_direction_error.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create vector comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    component_names = ["x", "y", "z"]

    for i, comp in enumerate(component_names):
        ax = axes[i]
        ax.plot(df["time"], df[f"sat_z_{comp}"], linewidth=1.5, alpha=0.8, label=f"Satellite Z-axis ({comp})")
        ax.plot(df["time"], df[f"sun_{comp}"], linewidth=1.5, alpha=0.8, linestyle='--', label=f"Sun direction ({comp})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{comp.upper()} component")
        ax.set_title(f"Vector Component: {comp.upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Sun Pointing Ideal: Direction Vectors (ECI frame)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "direction_vectors.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create angular velocity plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    omega_components = ["omega_x", "omega_y", "omega_z"]
    omega_labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    for i, (comp_name, label) in enumerate(zip(omega_components, omega_labels)):
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

    plt.suptitle("Sun Pointing Ideal Controller: Angular Velocity", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_velocity.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create angular momentum plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Angular momentum components
    ax = axes[0]
    ax.plot(df["time"], df["ang_mom_x"], linewidth=1.5, alpha=0.8, label="H_x")
    ax.plot(df["time"], df["ang_mom_y"], linewidth=1.5, alpha=0.8, label="H_y")
    ax.plot(df["time"], df["ang_mom_z"], linewidth=1.5, alpha=0.8, label="H_z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Momentum [kg·m²/s]")
    ax.set_title("Angular Momentum Components (ECI frame)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Angular momentum magnitude
    ax = axes[1]
    ax.plot(df["time"], df["ang_mom_norm"], linewidth=2, alpha=0.8, color='purple')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Momentum Magnitude [kg·m²/s]")
    ax.set_title("Angular Momentum Magnitude")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Sun Pointing Ideal Controller: Angular Momentum", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_momentum.png"
    plt.savefig(output_path, dpi=150)
    print(f"Angular momentum plot saved to: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nInitial sun pointing error: {df['angle_error_deg'].iloc[0]:.2f} deg")
    print(f"Final sun pointing error:   {df['angle_error_deg'].iloc[-1]:.2f} deg")
    print(f"Minimum error achieved:     {df['angle_error_deg'].min():.2f} deg")

    print(f"\nFinal angular velocity:")
    print(f"  omega_x: {df['omega_x'].iloc[-1]:.6f} rad/s")
    print(f"  omega_y: {df['omega_y'].iloc[-1]:.6f} rad/s")
    print(f"  omega_z: {df['omega_z'].iloc[-1]:.6f} rad/s")
    print(f"  norm:    {df['omega_norm'].iloc[-1]:.6f} rad/s")


if __name__ == "__main__":
    main()
