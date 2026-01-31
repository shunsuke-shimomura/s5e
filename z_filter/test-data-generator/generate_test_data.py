"""
generate_test_data.py
Structured test data generator for Z-filter validation.

Generates test data sets for multiple filter configurations in test_conditions.py.
Each test condition creates its own directory with:
  - config.json: Test condition parameters
  - b_coeffs.csv, a_coeffs.csv: Filter coefficients
  - signals/: Directory containing input/output signal pairs

Supports both single test condition generation and batch generation of all conditions.

Dependencies:
  - numpy, scipy (signal), matplotlib (only for the optional plot)
Tested with: Python 3.10+, NumPy 1.26+, SciPy 1.10+
"""

import argparse
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from test_conditions import (
    TEST_CONDITIONS,
    get_test_condition,
    save_condition_config,
    list_test_conditions,
)


def generate_test_data_for_condition(condition, base_output_dir: Path, trace: bool = False):
    """Generate test data for a single test condition."""
    # Create output directory
    output_dir = base_output_dir / condition.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create signals subdirectory
    signals_dir = output_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # Save condition config
    save_condition_config(condition, output_dir / "config.json")

    Ts = condition.Ts
    fs = 1.0 / Ts

    # Build analog transfer function
    if condition.num_coeffs and condition.den_coeffs:
        num_s = np.array(condition.num_coeffs, dtype=float)
        den_s = np.array(condition.den_coeffs, dtype=float)
    else:
        w0 = 2 * np.pi * condition.w0_hz
        zeta = condition.zeta

        if condition.num_coeffs:  # Custom numerator with standard denominator
            num_s = np.array(condition.num_coeffs, dtype=float)
            # Generate appropriate denominator based on order
            if condition.order == 2:
                den_s = np.array([1.0, 2 * zeta * w0, w0**2], dtype=float)
            elif condition.order == 4 and condition.name == "cascade_4th_order":
                # 4th order Butterworth approximation
                den_s = np.array(
                    [1.0, 2.613 * w0, 3.414 * w0**2, 2.613 * w0**3, w0**4],
                    dtype=float,
                )
            else:
                # Default: simple repeated pole structure
                den_s = np.array(
                    [1.0] + [2 * zeta * w0] * condition.order + [w0**condition.order],
                    dtype=float,
                )
        else:  # Standard lowpass filter
            if condition.order == 2:
                num_s = np.array([w0**2], dtype=float)
                den_s = np.array([1.0, 2 * zeta * w0, w0**2], dtype=float)
            elif condition.order == 4 and condition.name == "cascade_4th_order":
                # 4th order Butterworth
                num_s = np.array([w0**4], dtype=float)
                den_s = np.array(
                    [1.0, 2.613 * w0, 3.414 * w0**2, 2.613 * w0**3, w0**4],
                    dtype=float,
                )
            else:
                # Generic higher order lowpass
                num_s = np.array([w0**condition.order], dtype=float)
                # Simple approximation - can be improved with proper filter design
                coeffs = [1.0]
                for i in range(1, condition.order + 1):
                    coeffs.append(2 * w0)  # Simplified coefficient structure
                coeffs[-1] = w0**condition.order  # Last coefficient
                den_s = np.array(coeffs, dtype=float)

    # 1) c2d (bilinear/Tustin)
    # For now, use standard bilinear transform (prewarping can be added later)
    bz, az = signal.bilinear(num_s, den_s, fs)
    # cont2discrete returns 2D arrays for bz when MIMO shape; flatten
    bz = np.ravel(bz).astype(float)
    az = np.ravel(az).astype(float)

    # Normalize so that a[0]==1
    if not np.isclose(az[0], 1.0):
        bz = bz / az[0]
        az = az / az[0]

    # Save coefficients
    np.savetxt(output_dir / "b_coeffs.csv", bz[None, :], delimiter=",")
    np.savetxt(output_dir / "a_coeffs.csv", az[None, :], delimiter=",")

    # 2) Create test signals
    N = condition.Nsamps
    t = np.arange(N) * Ts

    signals = {}

    if condition.generate_impulse:
        imp = np.zeros(N)
        imp[0] = 1.0
        signals["impulse"] = imp

    if condition.generate_step:
        signals["step"] = np.ones(N)

    if condition.generate_sine:
        signals[f"sine{int(condition.sine_freq_hz)}"] = np.sin(
            2 * np.pi * condition.sine_freq_hz * t
        )

    if condition.generate_chirp:
        chirp_sig = signal.chirp(
            t,
            f0=condition.chirp_f0,
            t1=t[-1] if N > 1 else Ts,
            f1=condition.chirp_f1,
            method="linear",
        )
        signals[f"chirp{int(condition.chirp_f0)}to{int(condition.chirp_f1)}"] = chirp_sig

    if condition.generate_noise:
        rng = np.random.default_rng(condition.seed)
        signals["noise"] = rng.standard_normal(N) * condition.noise_std

    # 3) Reference outputs via lfilter (IIR, DF2T under the hood)
    for name, x in signals.items():
        y = signal.lfilter(bz, az, x)
        xy = np.column_stack([x, y])
        np.savetxt(signals_dir / f"{name}_xy.csv", xy, delimiter=",")

        # Generate signal processing graph
        try:
            plt.figure(figsize=(12, 8))

            # Plot input and output signals
            plt.subplot(2, 1, 1)
            plt.plot(t, x, "b-", label="Input", alpha=0.7)
            plt.plot(t, y, "r-", label="Output", alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.title(f"Signal Processing: {name} - {condition.name}")
            plt.legend()

            # Plot frequency domain comparison for some signals
            if name in ["impulse", "noise"] and len(x) > 64:
                plt.subplot(2, 1, 2)
                # Compute FFT
                X_fft = np.fft.fft(x)
                Y_fft = np.fft.fft(y)
                freqs = np.fft.fftfreq(len(x), Ts)

                # Only plot positive frequencies up to Nyquist
                pos_mask = freqs > 0
                freqs_pos = freqs[pos_mask]
                X_fft_pos = X_fft[pos_mask]
                Y_fft_pos = Y_fft[pos_mask]

                plt.loglog(freqs_pos, np.abs(X_fft_pos), "b-", label="Input FFT", alpha=0.7)
                plt.loglog(freqs_pos, np.abs(Y_fft_pos), "r-", label="Output FFT", alpha=0.7)
                plt.grid(True, alpha=0.3)
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Magnitude")
                plt.title("Frequency Domain Comparison")
                plt.legend()
            else:
                # For other signals, show a zoomed view of the time domain
                plt.subplot(2, 1, 2)
                # Show first 100 samples or less
                n_zoom = min(100, len(x))
                t_zoom = t[:n_zoom]
                x_zoom = x[:n_zoom]
                y_zoom = y[:n_zoom]

                plt.plot(t_zoom, x_zoom, "b.-", label="Input", alpha=0.7, markersize=3)
                plt.plot(t_zoom, y_zoom, "r.-", label="Output", alpha=0.7, markersize=3)
                plt.grid(True, alpha=0.3)
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.title(f"Zoomed View (first {n_zoom} samples)")
                plt.legend()

            plt.tight_layout()
            plt.savefig(signals_dir / f"{name}_processing.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Signal plot skipped for {condition.name}/{name}: {e}")

    # 4) Optional frequency response plot
    try:
        w, H = signal.freqz(bz, az, worN=2048, fs=fs)
        plt.figure(figsize=(10, 6))
        plt.plot(w, 20 * np.log10(np.maximum(np.abs(H), 1e-12)))
        plt.grid(True)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.title(f"Discrete H(z) - {condition.name}")
        plt.tight_layout()
        plt.savefig(output_dir / "bode_check.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Frequency plot skipped for {condition.name}: {e}")

    print(f"Generated test data for {condition.name} in {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate structured test data for Z-filter validation"
    )
    ap.add_argument("--condition", type=str, help="Generate data for specific test condition name")
    ap.add_argument("--list", action="store_true", help="List available test conditions")
    ap.add_argument("--all", action="store_true", help="Generate data for all test conditions")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="Also emit per-sample DF2T internal states for noise",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="../test-data",
        help="Base output directory (default: ../test-data)",
    )
    args = ap.parse_args()

    if args.list:
        print("Available test conditions:")
        for i, name in enumerate(list_test_conditions(), 1):
            condition = get_test_condition(name)
            print(f"{i: 2d}. {name: 20s} - {condition.description}")
        return 0

    base_output_dir = Path(args.output_dir)

    if args.condition:
        # Generate for specific condition
        condition = get_test_condition(args.condition)
        if condition is None:
            print(f"Error: Test condition '{args.condition}' not found.")
            print("Available conditions:")
            for name in list_test_conditions():
                print(f"  {name}")
            return 1

        generate_test_data_for_condition(condition, base_output_dir, args.trace)

    elif args.all:
        # Generate for all conditions
        print(f"Generating test data for {len(TEST_CONDITIONS)} conditions...")
        for condition in TEST_CONDITIONS:
            generate_test_data_for_condition(condition, base_output_dir, args.trace)
        print(f"\nCompleted generation of all test conditions in {base_output_dir}")

    else:
        print("Error: Must specify --condition, --all, or --list")
        print("Use --help for more information.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
