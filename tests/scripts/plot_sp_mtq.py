#!/usr/bin/env python3
"""
Plot sun direction error from sun pointing MTQ convergence test.
Reads tests/out/sp_mtq/sun_direction_error.csv and generates plots.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def add_eclipse_shading(ax, time, shadow_coeff, threshold=0.5):
    """Add gray background shading for eclipse periods (shadow_coeff < threshold)."""
    is_eclipse = shadow_coeff < threshold
    ylim = ax.get_ylim()
    ax.fill_between(time, ylim[0], ylim[1], where=is_eclipse,
                   alpha=0.2, color='gray', zorder=0)
    ax.set_ylim(ylim)  # Restore y limits


def main():
    parser = argparse.ArgumentParser(description="Plot sun pointing MTQ convergence test results")
    parser.add_argument("--dir", default="out/sp_mtq", help="Output directory containing CSV files")
    args = parser.parse_args()

    # Read the CSV file
    csv_path = Path(args.dir) / "sun_direction_error.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Read CSV data (no header)
    # Columns: time, angle_error_deg, sat_z_x, sat_z_y, sat_z_z, sun_x, sun_y, sun_z,
    #          ang_mom_x, ang_mom_y, ang_mom_z, mag_dir_x, mag_dir_y, mag_dir_z,
    #          ctrl_err_x, ctrl_err_y, ctrl_err_z, shadow_coeff
    df = pd.read_csv(csv_path, header=None, names=[
        "time", "angle_error_deg",
        "sat_z_x", "sat_z_y", "sat_z_z",
        "sun_x", "sun_y", "sun_z",
        "ang_mom_x", "ang_mom_y", "ang_mom_z",
        "mag_dir_x", "mag_dir_y", "mag_dir_z",
        "ctrl_err_x", "ctrl_err_y", "ctrl_err_z",
        "shadow_coeff"
    ])

    print(f"CSV shape: {df.shape}")
    print(f"Time range: {df['time'].min():.2f} - {df['time'].max():.2f} sec")

    # Create output directory
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if shadow coefficient is available
    has_eclipse_data = "shadow_coeff" in df.columns

    # Create main plot: Sun direction error
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Angle error plot
    ax = axes[0]
    ax.plot(df["time"], df["angle_error_deg"], linewidth=2, alpha=0.8, color='red')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target: 10 deg')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle Error [deg]")
    ax.set_title("Sun Pointing Error (angle between satellite Z-axis and Sun direction)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if has_eclipse_data:
        add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

    # Angle error in log scale
    ax = axes[1]
    ax.semilogy(df["time"], df["angle_error_deg"], linewidth=2, alpha=0.8, color='red')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Target: 10 deg')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle Error [deg]")
    ax.set_title("Sun Pointing Error (log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if has_eclipse_data:
        add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

    # Shadow coefficient plot
    if has_eclipse_data:
        ax = axes[2]
        ax.fill_between(df["time"], 0, 1, where=(df["shadow_coeff"] < 0.5),
                       alpha=0.3, color='gray', label='Eclipse')
        ax.plot(df["time"], df["shadow_coeff"], linewidth=1.5, alpha=0.8, color='orange')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Shadow Coefficient [-]")
        ax.set_title("Shadow Coefficient (0=Eclipse, 1=Full Sun)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Sun Pointing MTQ Controller: Convergence", fontsize=14, y=0.995)
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
        if has_eclipse_data:
            add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

    plt.suptitle("Sun Pointing MTQ: Direction Vectors (ECI frame)", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "direction_vectors.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nInitial sun pointing error: {df['angle_error_deg'].iloc[0]:.2f} deg")
    print(f"Final sun pointing error:   {df['angle_error_deg'].iloc[-1]:.2f} deg")
    print(f"Minimum error achieved:     {df['angle_error_deg'].min():.2f} deg")

    # Check if angular momentum columns exist
    if "ang_mom_x" in df.columns:
        # Calculate angular momentum magnitude
        df["ang_mom_mag"] = np.sqrt(df["ang_mom_x"]**2 + df["ang_mom_y"]**2 + df["ang_mom_z"]**2)

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
        if has_eclipse_data:
            add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

        # Angular momentum magnitude
        ax = axes[1]
        ax.plot(df["time"], df["ang_mom_mag"], linewidth=2, alpha=0.8, color='purple')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular Momentum Magnitude [kg·m²/s]")
        ax.set_title("Angular Momentum Magnitude")
        ax.grid(True, alpha=0.3)
        if has_eclipse_data:
            add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

        plt.suptitle("Sun Pointing MTQ Controller: Angular Momentum", fontsize=14, y=0.995)
        plt.tight_layout()

        output_path = output_dir / "angular_momentum.png"
        plt.savefig(output_path, dpi=150)
        print(f"Angular momentum plot saved to: {output_path}")
        plt.close()

    # Check if magnetic field columns exist for decomposition
    if "mag_dir_x" in df.columns:
        # Decompose sun pointing error into magnetic field parallel/perpendicular
        # Use cross product: sat_z × sun_dir (required torque direction)
        error_vec_x = df["sat_z_y"] * df["sun_z"] - df["sat_z_z"] * df["sun_y"]
        error_vec_y = df["sat_z_z"] * df["sun_x"] - df["sat_z_x"] * df["sun_z"]
        error_vec_z = df["sat_z_x"] * df["sun_y"] - df["sat_z_y"] * df["sun_x"]

        # Project onto magnetic field direction (parallel component - uncontrollable by MTQ)
        error_parallel = (error_vec_x * df["mag_dir_x"] +
                         error_vec_y * df["mag_dir_y"] +
                         error_vec_z * df["mag_dir_z"])

        # Perpendicular component magnitude (controllable by MTQ)
        perp_x = error_vec_x - error_parallel * df["mag_dir_x"]
        perp_y = error_vec_y - error_parallel * df["mag_dir_y"]
        perp_z = error_vec_z - error_parallel * df["mag_dir_z"]
        error_perpendicular = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

        # Create sun direction error decomposition plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        ax = axes[0]
        ax.plot(df["time"], error_parallel, linewidth=1.5, alpha=0.8, color='blue', label='Parallel to B (uncontrollable)')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error (Parallel) [-]")
        ax.set_title("Sun Direction Error - Parallel to Magnetic Field (Uncontrollable by MTQ)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if has_eclipse_data:
            add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

        ax = axes[1]
        ax.plot(df["time"], error_perpendicular, linewidth=1.5, alpha=0.8, color='red', label='Perpendicular to B (controllable)')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error (Perpendicular) [-]")
        ax.set_title("Sun Direction Error - Perpendicular to Magnetic Field (Controllable by MTQ)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if has_eclipse_data:
            add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

        ax = axes[2]
        ax.plot(df["time"], np.abs(error_parallel), linewidth=1.5, alpha=0.8, label='|Parallel| (uncontrollable)')
        ax.plot(df["time"], error_perpendicular, linewidth=1.5, alpha=0.8, label='Perpendicular (controllable)')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error Magnitude [-]")
        ax.set_title("Sun Direction Error - Both Components")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if has_eclipse_data:
            add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

        plt.suptitle("Sun Direction Error Decomposition (sat_z × sun_dir)", fontsize=14, y=0.995)
        plt.tight_layout()

        output_path = output_dir / "sun_error_decomposed.png"
        plt.savefig(output_path, dpi=150)
        print(f"Sun direction error decomposition plot saved to: {output_path}")
        plt.close()

        # Decompose angular momentum into magnetic field parallel/perpendicular
        if "ang_mom_x" in df.columns:
            # Project angular momentum onto magnetic field direction
            h_parallel = (df["ang_mom_x"] * df["mag_dir_x"] +
                         df["ang_mom_y"] * df["mag_dir_y"] +
                         df["ang_mom_z"] * df["mag_dir_z"])

            # Perpendicular component
            h_perp_x = df["ang_mom_x"] - h_parallel * df["mag_dir_x"]
            h_perp_y = df["ang_mom_y"] - h_parallel * df["mag_dir_y"]
            h_perp_z = df["ang_mom_z"] - h_parallel * df["mag_dir_z"]
            h_perpendicular = np.sqrt(h_perp_x**2 + h_perp_y**2 + h_perp_z**2)

            # Create angular momentum decomposition plot
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))

            ax = axes[0]
            ax.plot(df["time"], h_parallel, linewidth=1.5, alpha=0.8, color='blue', label='H_parallel')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("H (Parallel) [kg·m²/s]")
            ax.set_title("Angular Momentum - Parallel to Magnetic Field")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            ax = axes[1]
            ax.plot(df["time"], h_perpendicular, linewidth=1.5, alpha=0.8, color='red', label='H_perpendicular')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("H (Perpendicular) [kg·m²/s]")
            ax.set_title("Angular Momentum - Perpendicular to Magnetic Field")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            ax = axes[2]
            ax.plot(df["time"], np.abs(h_parallel), linewidth=1.5, alpha=0.8, label='|H_parallel|')
            ax.plot(df["time"], h_perpendicular, linewidth=1.5, alpha=0.8, label='H_perpendicular')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angular Momentum [kg·m²/s]")
            ax.set_title("Angular Momentum - Both Components")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            plt.suptitle("Angular Momentum Decomposition (Geomagnetic Frame)", fontsize=14, y=0.995)
            plt.tight_layout()

            output_path = output_dir / "angular_momentum_decomposed.png"
            plt.savefig(output_path, dpi=150)
            print(f"Angular momentum decomposition plot saved to: {output_path}")
            plt.close()

        # Check if control error columns exist
        if "ctrl_err_x" in df.columns:
            # Control error vector: sat_z × sun_dir (cross product)
            # This is the torque direction needed to align sat_z with sun_dir

            # Project control error onto magnetic field direction (parallel - cannot be controlled by MTQ)
            ctrl_err_parallel = (df["ctrl_err_x"] * df["mag_dir_x"] +
                                df["ctrl_err_y"] * df["mag_dir_y"] +
                                df["ctrl_err_z"] * df["mag_dir_z"])

            # Perpendicular component (can be controlled by MTQ)
            ctrl_perp_x = df["ctrl_err_x"] - ctrl_err_parallel * df["mag_dir_x"]
            ctrl_perp_y = df["ctrl_err_y"] - ctrl_err_parallel * df["mag_dir_y"]
            ctrl_perp_z = df["ctrl_err_z"] - ctrl_err_parallel * df["mag_dir_z"]
            ctrl_err_perpendicular = np.sqrt(ctrl_perp_x**2 + ctrl_perp_y**2 + ctrl_perp_z**2)

            # Control error magnitude
            ctrl_err_mag = np.sqrt(df["ctrl_err_x"]**2 + df["ctrl_err_y"]**2 + df["ctrl_err_z"]**2)

            # Create control error decomposition plot
            fig, axes = plt.subplots(4, 1, figsize=(14, 16))

            ax = axes[0]
            ax.plot(df["time"], ctrl_err_mag, linewidth=1.5, alpha=0.8, color='black', label='|e| = |sat_z × sun|')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Control Error Magnitude [-]")
            ax.set_title("Control Error Vector Magnitude (sat_z × sun_dir)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            ax = axes[1]
            ax.plot(df["time"], ctrl_err_parallel, linewidth=1.5, alpha=0.8, color='blue', label='e_parallel (uncontrollable by MTQ)')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Control Error (Parallel) [-]")
            ax.set_title("Control Error - Parallel to Magnetic Field (Cannot be controlled by MTQ)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            ax = axes[2]
            ax.plot(df["time"], ctrl_err_perpendicular, linewidth=1.5, alpha=0.8, color='red', label='e_perpendicular (controllable by MTQ)')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Control Error (Perpendicular) [-]")
            ax.set_title("Control Error - Perpendicular to Magnetic Field (Controllable by MTQ)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            ax = axes[3]
            ax.plot(df["time"], np.abs(ctrl_err_parallel), linewidth=1.5, alpha=0.8, label='|e_parallel| (uncontrollable)')
            ax.plot(df["time"], ctrl_err_perpendicular, linewidth=1.5, alpha=0.8, label='e_perpendicular (controllable)')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Control Error Magnitude [-]")
            ax.set_title("Control Error - Both Components")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if has_eclipse_data:
                add_eclipse_shading(ax, df["time"], df["shadow_coeff"])

            plt.suptitle("Control Error Vector Decomposition (sat_z × sun_dir)", fontsize=14, y=0.995)
            plt.tight_layout()

            output_path = output_dir / "control_error_decomposed.png"
            plt.savefig(output_path, dpi=150)
            print(f"Control error decomposition plot saved to: {output_path}")
            plt.close()

    # Try to plot control efficiency if available
    detail_csv = output_dir / "detailed_log.csv"
    if detail_csv.exists():
        try:
            df_detail = pd.read_csv(detail_csv, header=None)
            # Check if control efficiency column exists (column 12, index 12)
            if df_detail.shape[1] >= 13:
                fig, axes = plt.subplots(3, 1, figsize=(14, 12))

                time = df_detail.iloc[:, 0]
                angle_error = df_detail.iloc[:, 1]
                shadow_coef = df_detail.iloc[:, 2]
                control_eff = df_detail.iloc[:, 12]

                # Plot angle error
                ax = axes[0]
                ax.plot(time, angle_error, linewidth=1.5, alpha=0.8, color='red')
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Angle Error [deg]")
                ax.set_title("Sun Pointing Error")
                ax.grid(True, alpha=0.3)

                # Plot control efficiency
                ax = axes[1]
                ax.plot(time, control_eff, linewidth=1.5, alpha=0.8, color='blue')
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Control Efficiency")
                ax.set_title("MTQ Control Efficiency (sin of angle between B-field and required torque)")
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)

                # Plot shadow coefficient
                ax = axes[2]
                ax.plot(time, shadow_coef, linewidth=1.5, alpha=0.8, color='orange')
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Shadow Coefficient")
                ax.set_title("Shadow Coefficient (0=eclipse, 1=full sun)")
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)

                plt.suptitle("Sun Pointing MTQ Control: Detailed Analysis", fontsize=14, y=0.995)
                plt.tight_layout()

                output_path = output_dir / "detailed_analysis.png"
                plt.savefig(output_path, dpi=150)
                print(f"Detailed analysis plot saved to: {output_path}")
                plt.close()
        except Exception as e:
            print(f"Warning: Could not plot detailed analysis: {e}")


if __name__ == "__main__":
    main()
