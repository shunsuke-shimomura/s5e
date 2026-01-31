"""
test_conditions.py
Defines test conditions for Z-filter validation tests.
Each test condition specifies filter parameters and expected behavior.
"""

from typing import List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class TestCondition:
    """Defines a single test condition for filter validation."""

    name: str
    description: str
    Ts: float  # Sample time [s]
    prewarp_hz: float  # Prewarp frequency in Hz (0=disabled)
    Nsamps: int  # Samples per test signal
    seed: int  # RNG seed for noise
    order: int  # Filter order (degree of denominator polynomial)

    # Filter definition (either standard form or custom coefficients)
    w0_hz: Optional[float] = None  # Natural frequency for standard LPF [Hz]
    zeta: Optional[float] = None  # Damping ratio for standard LPF
    num_coeffs: Optional[List[float]] = None  # Custom numerator coeffs (highest power first)
    den_coeffs: Optional[List[float]] = None  # Custom denominator coeffs (highest power first)

    # Test signals to generate
    generate_impulse: bool = True
    generate_step: bool = True
    generate_sine: bool = True
    generate_chirp: bool = True
    generate_noise: bool = True

    # Additional parameters
    sine_freq_hz: float = 30.0  # Frequency for sine test signal
    chirp_f0: float = 5.0  # Chirp start frequency
    chirp_f1: float = 200.0  # Chirp end frequency
    noise_std: float = 0.1  # Noise standard deviation


# Define test conditions
TEST_CONDITIONS = [
    # 2nd order filters
    TestCondition(
        name="lowpass_50hz_zeta07",
        description="2nd-order Butterworth-like lowpass, 50Hz cutoff, zeta=0.7",
        Ts=1 / 1000,  # 1kHz sampling
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=42,
        order=2,
        w0_hz=50.0,
        zeta=0.7,
    ),
    TestCondition(
        name="lowpass_100hz_prewarp",
        description="2nd-order lowpass with prewarping at 80Hz",
        Ts=1 / 500,  # 500Hz sampling
        prewarp_hz=80.0,
        Nsamps=1024,
        seed=123,
        order=2,
        w0_hz=100.0,
        zeta=0.8,
    ),
    TestCondition(
        name="custom_bandpass",
        description="Custom 2nd-order bandpass filter using explicit coefficients",
        Ts=1 / 2000,  # 2kHz sampling
        prewarp_hz=0.0,
        Nsamps=2048,
        seed=456,
        order=2,
        # Bandpass filter example: H(s) = s / (s^2 + 100*s + 10000)
        # This is a rough bandpass around ~16Hz with some damping
        num_coeffs=[0.0, 1.0, 0.0],  # s (numerator)
        den_coeffs=[1.0, 100.0, 10000.0],  # s^2 + 100*s + 10000 (denominator)
    ),
    TestCondition(
        name="highpass_20hz",
        description="2nd-order high-pass filter using s^2 numerator",
        Ts=1 / 1000,
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=789,
        order=2,
        # High-pass: H(s) = s^2 / (s^2 + 2*zeta*w0*s + w0^2)
        # w0 = 2*pi*20, zeta = 0.707
        w0_hz=20.0,
        zeta=0.707,
        num_coeffs=[1.0, 0.0, 0.0],  # s^2 (makes it high-pass)
    ),
    TestCondition(
        name="low_sample_rate_filter",
        description="2nd-order filter designed for low sample rate (100Hz)",
        Ts=1 / 100,  # 100Hz sampling
        prewarp_hz=0.0,
        Nsamps=512,
        seed=999,
        order=2,
        w0_hz=10.0,  # 10Hz cutoff
        zeta=0.6,
        generate_chirp=False,  # Skip chirp for low sample rate
        chirp_f1=40.0,  # Limit chirp range
    ),
    # 3rd order filters
    TestCondition(
        name="butterworth_3rd_30hz",
        description="3rd-order Butterworth lowpass filter, 30Hz cutoff",
        Ts=1 / 1000,
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=111,
        order=3,
        # 3rd order Butterworth: H(s) = w0^3 / (s^3 + 2*w0*s^2 + 2*w0^2*s + w0^3)
        # Butterworth has specific coefficients
        num_coeffs=[8 * (2 * 3.14159 * 30) ** 3],  # w0^3
        den_coeffs=[
            1.0,
            2 * (2 * 3.14159 * 30),
            2 * (2 * 3.14159 * 30) ** 2,
            (2 * 3.14159 * 30) ** 3,
        ],
    ),
    TestCondition(
        name="custom_3rd_order",
        description="Custom 3rd-order filter with complex poles",
        Ts=1 / 2000,
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=222,
        order=3,
        # Custom 3rd order: H(s) = 1000 / (s^3 + 10*s^2 + 100*s + 1000)
        num_coeffs=[1000.0],
        den_coeffs=[1.0, 10.0, 100.0, 1000.0],
    ),
    # 4th order filters
    TestCondition(
        name="cascade_4th_order",
        description="4th-order filter as cascade of two 2nd-order sections",
        Ts=1 / 1000,
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=333,
        order=4,
        # 4th order Butterworth: (s^2 + sqrt(2)*w0*s + w0^2) * ...
        # Simplified: H(s) = w0^4 / (s^4 + 2.613*w0*s^3 + ...)
        w0_hz=40.0,  # Will be used to compute coefficients
    ),
    TestCondition(
        name="elliptic_4th_order",
        description="4th-order elliptic-like filter with zeros",
        Ts=1 / 1000,
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=444,
        order=4,
        # Custom 4th order with numerator zeros
        num_coeffs=[1.0, 0.0, 16.0, 0.0, 64.0],  # (s^2 + 4^2) * (s^2 + 8^2)
        den_coeffs=[1.0, 5.0, 20.0, 50.0, 100.0],  # Custom denominator
    ),
    # 5th order filter
    TestCondition(
        name="bessel_5th_order",
        description="5th-order Bessel-like filter for linear phase",
        Ts=1 / 1000,
        prewarp_hz=0.0,
        Nsamps=1024,
        seed=555,
        order=5,
        # 5th order Bessel approximation
        # H(s) = 945 / (s^5 + 15*s^4 + 105*s^3 + 420*s^2 + 945*s + 945)
        num_coeffs=[945.0],
        den_coeffs=[1.0, 15.0, 105.0, 420.0, 945.0, 945.0],
    ),
]


def get_test_condition(name: str) -> Optional[TestCondition]:
    """Get test condition by name."""
    for condition in TEST_CONDITIONS:
        if condition.name == name:
            return condition
    return None


def list_test_conditions() -> List[str]:
    """List all available test condition names."""
    return [condition.name for condition in TEST_CONDITIONS]


def save_condition_config(condition: TestCondition, output_path: str):
    """Save test condition as JSON config file."""
    config_dict = asdict(condition)
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_condition_config(config_path: str) -> TestCondition:
    """Load test condition from JSON config file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TestCondition(**config_dict)


if __name__ == "__main__":
    # Print all available test conditions
    print("Available test conditions:")
    for i, condition in enumerate(TEST_CONDITIONS, 1):
        print(f"{i}. {condition.name}: {condition.description}")
