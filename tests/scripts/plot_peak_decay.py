#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_envelope(energy, window=1000):
    """Compute running maximum envelope using numpy"""
    n = len(energy)
    envelope = np.zeros(n)
    half_w = window // 2
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        envelope[i] = np.max(energy[start:end])
    return envelope

def find_peaks(energy, window=20):
    """Find local maxima with prominence filtering"""
    peaks_idx = []
    peaks_val = []

    for i in range(window, len(energy) - window):
        center = energy[i]
        is_peak = all(energy[i-j] <= center for j in range(1, window+1))
        is_peak = is_peak and all(energy[i+j] <= center for j in range(1, window+1))

        if is_peak:
            left_min = min(energy[i-window:i])
            right_min = min(energy[i+1:i+window+1])
            prominence = center - max(left_min, right_min)
            threshold = max(center * 0.005, 1e-8)

            if prominence > threshold:
                peaks_idx.append(i)
                peaks_val.append(center)

    return peaks_idx, peaks_val

def main():
    output_base = Path("out")

    modes = ["lyapunovpointing", "conditionalpointing", "gainscheduled",
             "crossproduct", "slowcontrol", "periodicaveraging"]
    labels = ["Lyapunov", "Conditional", "Gain-Scheduled",
              "Cross-Product", "Slow Control", "Periodic Avg"]
    colors = ["red", "blue", "purple", "orange", "green", "brown"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 16))

    dt = 0.1
    envelope_data = {}

    for mode, label, color in zip(modes, labels, colors):
        csv_path = output_base / f"sp_mtq_peak_{mode}" / "energy_tracking.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path, header=None, names=[
            "time", "angle_error_deg", "kinetic_energy", "potential_energy",
            "total_energy", "omega_norm", "H_x", "H_y", "H_z"
        ])

        energy = df["total_energy"].values
        time = df["time"].values

        # Plot energy
        axes[0].plot(time, energy, linewidth=0.8, alpha=0.6, label=label, color=color)

        # Compute and plot envelope
        envelope = compute_envelope(energy, window=1000)  # 100 sec window
        axes[1].semilogy(time, envelope, linewidth=1.5, alpha=0.8, label=label, color=color)
        envelope_data[label] = (time, envelope)

        # Plot angle error
        axes[2].plot(time, df["angle_error_deg"], linewidth=1, alpha=0.7, label=label, color=color)

        # Plot kinetic energy (shows oscillation behavior)
        axes[3].plot(time, df["kinetic_energy"], linewidth=0.8, alpha=0.6, label=label, color=color)

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Total Energy V")
    axes[0].set_title("Lyapunov Energy V = (1/2)ω^T I ω + k_p(1-s·z)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Energy Envelope (log scale)")
    axes[1].set_title("Energy Envelope Decay (running max over 100s window)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angle Error [deg]")
    axes[2].set_title("Sun Pointing Error")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Kinetic Energy")
    axes[3].set_title("Kinetic Energy (1/2)ω^T I ω")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    plt.suptitle("MTQ Control: Energy Envelope Decay Analysis", fontsize=14, y=0.995)
    plt.tight_layout()

    output_dir = output_base / "sp_mtq_peak_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "peak_decay.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print decay analysis
    print("\n=== Envelope Decay Summary ===")
    for label, (times, env) in envelope_data.items():
        # Sample at regular intervals
        sample_idx = np.arange(0, len(env), 1000)
        sample_times = times[sample_idx]
        sample_env = env[sample_idx]
        valid = sample_env > 1e-10

        if np.sum(valid) >= 2:
            log_env = np.log(sample_env[valid])
            t_valid = sample_times[valid]
            slope, _ = np.polyfit(t_valid, log_env, 1)
            decay_rate = -slope
            reduction = (1 - sample_env[-1]/sample_env[0]) * 100
            print(f"{label}: decay_rate={decay_rate:.2e}/s, reduction={reduction:.1f}%")

if __name__ == "__main__":
    main()
