#!/usr/bin/env python3
"""
Plot sun direction error from sun pointing MTQ with virtual magnetic field test.
Reads tests/out/sp_mtq_virtual/sun_direction_error.csv and generates plots.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot sun pointing MTQ (virtual magfield) test results")
    parser.add_argument("--dir", default="out/sp_mtq_virtual", help="Output directory containing CSV files")
    args = parser.parse_args()

    # Read the CSV file
    csv_path = Path(args.dir) / "sun_direction_error.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Read CSV data (no header)
    # Columns: time, angle_error_deg, sat_z_x, sat_z_y, sat_z_z, sun_x, sun_y, sun_z,
    #          ang_mom_x, ang_mom_y, ang_mom_z, mag_dir_x, mag_dir_y, mag_dir_z,
    #          ctrl_err_x, ctrl_err_y, ctrl_err_z, omega_norm
    df = pd.read_csv(csv_path, header=None, names=[
        "time", "angle_error_deg",
        "sat_z_x", "sat_z_y", "sat_z_z",
        "sun_x", "sun_y", "sun_z",
        "ang_mom_x", "ang_mom_y", "ang_mom_z",
        "mag_dir_x", "mag_dir_y", "mag_dir_z",
        "ctrl_err_x", "ctrl_err_y", "ctrl_err_z",
        "omega_norm"
    ])

    print(f"CSV shape: {df.shape}")
    print(f"Time range: {df['time'].min():.2f} - {df['time'].max():.2f} sec")

    # Create output directory
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create main plot: Sun direction error
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Angle error plot (linear scale)
    ax = axes[0]
    ax.plot(df["time"], df["angle_error_deg"], linewidth=2, alpha=0.8, color='red')
    ax.axhline(y=30, color='green', linestyle='--', linewidth=1, label='Target: 30 deg')
    ax.axhline(y=10, color='blue', linestyle='--', linewidth=1, label='Goal: 10 deg')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle Error [deg]")
    ax.set_title("Sun Pointing Error (angle between satellite Z-axis and Sun direction)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Angle error in log scale
    ax = axes[1]
    ax.semilogy(df["time"], df["angle_error_deg"], linewidth=2, alpha=0.8, color='red')
    ax.axhline(y=30, color='green', linestyle='--', linewidth=1, label='Target: 30 deg')
    ax.axhline(y=10, color='blue', linestyle='--', linewidth=1, label='Goal: 10 deg')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle Error [deg]")
    ax.set_title("Sun Pointing Error (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Angular velocity norm
    ax = axes[2]
    ax.plot(df["time"], df["omega_norm"], linewidth=1.5, alpha=0.8, color='blue')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Velocity Norm [rad/s]")
    ax.set_title("Angular Velocity Magnitude")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Sun Pointing MTQ Controller (Virtual Magnetic Field): Convergence", fontsize=14, y=0.995)
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

    plt.suptitle("Sun Pointing MTQ (Virtual Magfield): Direction Vectors (ECI frame)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "direction_vectors.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create angular momentum plot
    df["ang_mom_mag"] = np.sqrt(df["ang_mom_x"]**2 + df["ang_mom_y"]**2 + df["ang_mom_z"]**2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Angular momentum components
    ax = axes[0]
    ax.plot(df["time"], df["ang_mom_x"], linewidth=1.5, alpha=0.8, label="H_x")
    ax.plot(df["time"], df["ang_mom_y"], linewidth=1.5, alpha=0.8, label="H_y")
    ax.plot(df["time"], df["ang_mom_z"], linewidth=1.5, alpha=0.8, label="H_z")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Momentum [kg m^2/s]")
    ax.set_title("Angular Momentum Components (ECI frame)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Angular momentum magnitude
    ax = axes[1]
    ax.plot(df["time"], df["ang_mom_mag"], linewidth=2, alpha=0.8, color='purple')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Momentum Magnitude [kg m^2/s]")
    ax.set_title("Angular Momentum Magnitude")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Sun Pointing MTQ (Virtual Magfield): Angular Momentum", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_momentum.png"
    plt.savefig(output_path, dpi=150)
    print(f"Angular momentum plot saved to: {output_path}")
    plt.close()

    # Create magnetic field direction plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for i, comp in enumerate(component_names):
        ax = axes[i]
        ax.plot(df["time"], df[f"mag_dir_{comp}"], linewidth=1.5, alpha=0.8, color='orange')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"B_{comp.upper()} (normalized)")
        ax.set_title(f"Magnetic Field Direction: {comp.upper()} component")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Virtual Magnetic Field Direction (ECI frame)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "magnetic_field.png"
    plt.savefig(output_path, dpi=150)
    print(f"Magnetic field plot saved to: {output_path}")
    plt.close()

    # Decompose angular momentum into magnetic field parallel/perpendicular
    h_parallel = (df["ang_mom_x"] * df["mag_dir_x"] +
                 df["ang_mom_y"] * df["mag_dir_y"] +
                 df["ang_mom_z"] * df["mag_dir_z"])

    h_perp_x = df["ang_mom_x"] - h_parallel * df["mag_dir_x"]
    h_perp_y = df["ang_mom_y"] - h_parallel * df["mag_dir_y"]
    h_perp_z = df["ang_mom_z"] - h_parallel * df["mag_dir_z"]
    h_perpendicular = np.sqrt(h_perp_x**2 + h_perp_y**2 + h_perp_z**2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    ax.plot(df["time"], h_parallel, linewidth=1.5, alpha=0.8, color='blue', label='H_parallel')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("H (Parallel) [kg m^2/s]")
    ax.set_title("Angular Momentum - Parallel to Magnetic Field")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(df["time"], h_perpendicular, linewidth=1.5, alpha=0.8, color='red', label='H_perpendicular')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("H (Perpendicular) [kg m^2/s]")
    ax.set_title("Angular Momentum - Perpendicular to Magnetic Field")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(df["time"], np.abs(h_parallel), linewidth=1.5, alpha=0.8, label='|H_parallel|')
    ax.plot(df["time"], h_perpendicular, linewidth=1.5, alpha=0.8, label='H_perpendicular')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Momentum [kg m^2/s]")
    ax.set_title("Angular Momentum - Both Components")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle("Angular Momentum Decomposition (Geomagnetic Frame)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_momentum_decomposed.png"
    plt.savefig(output_path, dpi=150)
    print(f"Angular momentum decomposition plot saved to: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nInitial sun pointing error: {df['angle_error_deg'].iloc[0]:.2f} deg")
    print(f"Final sun pointing error:   {df['angle_error_deg'].iloc[-1]:.2f} deg")
    print(f"Minimum error achieved:     {df['angle_error_deg'].min():.2f} deg")

    print(f"\nFinal angular velocity norm: {df['omega_norm'].iloc[-1]:.6f} rad/s")


if __name__ == "__main__":
    main()
