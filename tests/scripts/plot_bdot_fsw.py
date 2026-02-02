#!/usr/bin/env python3
"""
Plot angular velocity and angular momentum from bdot_fsw convergence test.
Reads tests/out/bdot/bdot_fsw/angular_velocity.csv and generates plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Read the CSV file
    csv_path = Path("out/bdot/bdot_fsw/angular_velocity.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Read CSV data (no header)
    # Columns: time, omega_x, omega_y, omega_z, omega_norm, H_x, H_y, H_z, H_norm
    df = pd.read_csv(csv_path, header=None, names=[
        "time", "omega_x", "omega_y", "omega_z", "omega_norm",
        "H_x", "H_y", "H_z", "H_norm"
    ])

    print(f"CSV shape: {df.shape}")
    print(f"Time range: {df['time'].min():.2f} - {df['time'].max():.2f} sec")

    # Create output directory
    output_dir = Path("out/bdot/bdot_fsw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create angular velocity plot (linear scale)
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

    plt.suptitle("B-dot FSW Controller: Angular Velocity Convergence", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_velocity.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create angular velocity plot (log scale)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.semilogy(df["time"], np.abs(df["omega_x"]), linewidth=1.5, alpha=0.8, label=r"$|\omega_x|$")
    ax.semilogy(df["time"], np.abs(df["omega_y"]), linewidth=1.5, alpha=0.8, label=r"$|\omega_y|$")
    ax.semilogy(df["time"], np.abs(df["omega_z"]), linewidth=1.5, alpha=0.8, label=r"$|\omega_z|$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Velocity [rad/s]")
    ax.set_title("Angular Velocity Components (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.semilogy(df["time"], df["omega_norm"], linewidth=2, alpha=0.8, color='red')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$|\omega|$ [rad/s]")
    ax.set_title("Angular Velocity Norm (log scale)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("B-dot FSW Controller: Angular Velocity Convergence (Log Scale)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_velocity_log.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create angular momentum plot (linear scale)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    component_names = ["H_x", "H_y", "H_z"]
    labels = [r"$H_x$", r"$H_y$", r"$H_z$"]

    for i, (comp_name, label) in enumerate(zip(component_names, labels)):
        ax = axes[i]
        ax.plot(df["time"], df[comp_name], linewidth=1.5, alpha=0.8, color='green')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [kg·m²/s]")
        ax.set_title(f"Angular Momentum: {label}")
        ax.grid(True, alpha=0.3)

    # Norm plot
    ax = axes[3]
    ax.plot(df["time"], df["H_norm"], linewidth=2, alpha=0.8, color='purple')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$|H|$ [kg·m²/s]")
    ax.set_title("Angular Momentum Norm")
    ax.grid(True, alpha=0.3)

    plt.suptitle("B-dot FSW Controller: Angular Momentum Convergence", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_momentum.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Create angular momentum plot (log scale)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.semilogy(df["time"], np.abs(df["H_x"]), linewidth=1.5, alpha=0.8, label=r"$|H_x|$")
    ax.semilogy(df["time"], np.abs(df["H_y"]), linewidth=1.5, alpha=0.8, label=r"$|H_y|$")
    ax.semilogy(df["time"], np.abs(df["H_z"]), linewidth=1.5, alpha=0.8, label=r"$|H_z|$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular Momentum [kg·m²/s]")
    ax.set_title("Angular Momentum Components (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.semilogy(df["time"], df["H_norm"], linewidth=2, alpha=0.8, color='purple')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$|H|$ [kg·m²/s]")
    ax.set_title("Angular Momentum Norm (log scale)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("B-dot FSW Controller: Angular Momentum Convergence (Log Scale)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "angular_momentum_log.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nFinal angular velocity:")
    print(f"  omega_x: {df['omega_x'].iloc[-1]:.6f} rad/s")
    print(f"  omega_y: {df['omega_y'].iloc[-1]:.6f} rad/s")
    print(f"  omega_z: {df['omega_z'].iloc[-1]:.6f} rad/s")
    print(f"  norm:    {df['omega_norm'].iloc[-1]:.6f} rad/s")

    print(f"\nFinal angular momentum:")
    print(f"  H_x:  {df['H_x'].iloc[-1]:.6f} kg·m²/s")
    print(f"  H_y:  {df['H_y'].iloc[-1]:.6f} kg·m²/s")
    print(f"  H_z:  {df['H_z'].iloc[-1]:.6f} kg·m²/s")
    print(f"  norm: {df['H_norm'].iloc[-1]:.6f} kg·m²/s")


if __name__ == "__main__":
    main()
