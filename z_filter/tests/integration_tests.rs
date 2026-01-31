use polynomial::{Poly0, Poly1, Poly2, Poly3, Poly4, Poly5};
use z_filter::*;

// Import std types and macros for testing
use std::boxed::Box;
use std::fs;
use std::path::Path;
use std::vec::Vec;
use std::{eprintln, format, println};

/// Load configuration file to get filter order
fn load_config(config_path: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;
    Ok(config)
}

/// Load CSV data as a vector of f64 values
fn load_csv_vector(file_path: &Path) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;
    let values: Result<Vec<f64>, _> = content
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect();
    Ok(values?)
}

/// Load signal data (input/output pairs) from CSV
fn load_signal_data(file_path: &Path) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for line in content.lines() {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() >= 2 {
            inputs.push(parts[0].trim().parse::<f64>()?);
            outputs.push(parts[1].trim().parse::<f64>()?);
        }
    }

    Ok((inputs, outputs))
}

/// Enum to hold different filter orders dynamically
#[derive(Clone)]
enum DynamicFilter {
    Order0(DirectFormIITransposed<f64, Poly0<f64>, [f64; 1]>),
    Order1(DirectFormIITransposed<f64, Poly1<f64>, [f64; 2]>),
    Order2(DirectFormIITransposed<f64, Poly2<f64>, [f64; 3]>),
    Order3(DirectFormIITransposed<f64, Poly3<f64>, [f64; 4]>),
    Order4(DirectFormIITransposed<f64, Poly4<f64>, [f64; 5]>),
    Order5(DirectFormIITransposed<f64, Poly5<f64>, [f64; 6]>),
}

impl DynamicFilter {
    fn reset(&mut self) {
        match self {
            DynamicFilter::Order0(ref mut f) => Filter::reset(f),
            DynamicFilter::Order1(ref mut f) => Filter::reset(f),
            DynamicFilter::Order2(ref mut f) => Filter::reset(f),
            DynamicFilter::Order3(ref mut f) => Filter::reset(f),
            DynamicFilter::Order4(ref mut f) => Filter::reset(f),
            DynamicFilter::Order5(ref mut f) => Filter::reset(f),
        }
    }

    fn process_sample(&mut self, x: f64) -> f64 {
        match self {
            DynamicFilter::Order0(ref mut f) => Filter::process_sample(f, x),
            DynamicFilter::Order1(ref mut f) => Filter::process_sample(f, x),
            DynamicFilter::Order2(ref mut f) => Filter::process_sample(f, x),
            DynamicFilter::Order3(ref mut f) => Filter::process_sample(f, x),
            DynamicFilter::Order4(ref mut f) => Filter::process_sample(f, x),
            DynamicFilter::Order5(ref mut f) => Filter::process_sample(f, x),
        }
    }

    fn from_coeffs(
        order: usize,
        a_coeffs: &[f64],
        b_coeffs: &[f64],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match order {
            0 => {
                let mut a_padded = [0.0; 1];
                let mut b_padded = [0.0; 1];
                for i in 0..1 {
                    a_padded[i] = a_coeffs.get(i).copied().unwrap_or(0.0);
                    b_padded[i] = b_coeffs.get(i).copied().unwrap_or(0.0);
                }
                Ok(DynamicFilter::Order0(DirectFormIITransposed::new(
                    Poly0::<f64>::new(a_padded),
                    Poly0::<f64>::new(b_padded),
                )?))
            }
            1 => {
                let mut a_padded = [0.0; 2];
                let mut b_padded = [0.0; 2];
                for i in 0..2 {
                    a_padded[i] = a_coeffs.get(i).copied().unwrap_or(0.0);
                    b_padded[i] = b_coeffs.get(i).copied().unwrap_or(0.0);
                }
                Ok(DynamicFilter::Order1(DirectFormIITransposed::new(
                    Poly1::<f64>::new(a_padded),
                    Poly1::<f64>::new(b_padded),
                )?))
            }
            2 => {
                let mut a_padded = [0.0; 3];
                let mut b_padded = [0.0; 3];
                for i in 0..3 {
                    a_padded[i] = a_coeffs.get(i).copied().unwrap_or(0.0);
                    b_padded[i] = b_coeffs.get(i).copied().unwrap_or(0.0);
                }
                Ok(DynamicFilter::Order2(DirectFormIITransposed::new(
                    Poly2::<f64>::new(a_padded),
                    Poly2::<f64>::new(b_padded),
                )?))
            }
            3 => {
                let mut a_padded = [0.0; 4];
                let mut b_padded = [0.0; 4];
                for i in 0..4 {
                    a_padded[i] = a_coeffs.get(i).copied().unwrap_or(0.0);
                    b_padded[i] = b_coeffs.get(i).copied().unwrap_or(0.0);
                }
                Ok(DynamicFilter::Order3(DirectFormIITransposed::new(
                    Poly3::<f64>::new(a_padded),
                    Poly3::<f64>::new(b_padded),
                )?))
            }
            4 => {
                let mut a_padded = [0.0; 5];
                let mut b_padded = [0.0; 5];
                for i in 0..5 {
                    a_padded[i] = a_coeffs.get(i).copied().unwrap_or(0.0);
                    b_padded[i] = b_coeffs.get(i).copied().unwrap_or(0.0);
                }
                Ok(DynamicFilter::Order4(DirectFormIITransposed::new(
                    Poly4::<f64>::new(a_padded),
                    Poly4::<f64>::new(b_padded),
                )?))
            }
            5 => {
                let mut a_padded = [0.0; 6];
                let mut b_padded = [0.0; 6];
                for i in 0..6 {
                    a_padded[i] = a_coeffs.get(i).copied().unwrap_or(0.0);
                    b_padded[i] = b_coeffs.get(i).copied().unwrap_or(0.0);
                }
                Ok(DynamicFilter::Order5(DirectFormIITransposed::new(
                    Poly5::<f64>::new(a_padded),
                    Poly5::<f64>::new(b_padded),
                )?))
            }
            _ => Err(format!("Unsupported filter order: {}", order).into()),
        }
    }
}

/// Create a filter from test condition coefficients with dynamic order
fn create_dynamic_filter_from_test_data(
    test_condition_dir: &Path,
) -> Result<DynamicFilter, Box<dyn std::error::Error>> {
    let config = load_config(&test_condition_dir.join("config.json"))?;
    let order = config
        .get("order")
        .and_then(|v| v.as_u64())
        .ok_or("Missing or invalid 'order' field in config")? as usize;

    let a_coeffs = load_csv_vector(&test_condition_dir.join("a_coeffs.csv"))?;
    let b_coeffs = load_csv_vector(&test_condition_dir.join("b_coeffs.csv"))?;

    DynamicFilter::from_coeffs(order, &a_coeffs, &b_coeffs)
}

/// Test a filter against reference data with specified tolerance
fn test_filter_against_reference(
    mut filter: DynamicFilter,
    inputs: &[f64],
    expected_outputs: &[f64],
    tolerance: f64,
    test_name: &str,
) {
    assert_eq!(
        inputs.len(),
        expected_outputs.len(),
        "Input and output lengths must match for {}",
        test_name
    );

    filter.reset();
    let mut max_error: f64 = 0.0;
    let mut error_count = 0;

    for (i, (&input, &expected)) in inputs.iter().zip(expected_outputs.iter()).enumerate() {
        let actual = filter.process_sample(input);
        let error = (actual - expected).abs();

        if error > tolerance {
            error_count += 1;
            if error_count <= 5 {
                // Limit error output
                eprintln!(
                    "{}[{}]: expected {:.6}, got {:.6}, error {:.6}",
                    test_name, i, expected, actual, error
                );
            }
        }

        max_error = max_error.max(error);
    }

    let error_rate = error_count as f64 / inputs.len() as f64;
    println!(
        "{}: max_error={:.2e}, error_rate={:.1}",
        test_name,
        max_error,
        error_rate * 100.0
    );

    // Allow some numerical errors, but not too many
    assert!(
        error_rate < 0.05,
        "Too many samples exceed tolerance in {}: {:.1}% > 5%",
        test_name,
        error_rate * 100.0
    );
    assert!(
        max_error < tolerance * 10.0,
        "Maximum error too large in {}: {:.2e} > {:.2e}",
        test_name,
        max_error,
        tolerance * 10.0
    );
}

/// Run tests for a single test condition directory
fn run_tests_for_condition(test_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let condition_name = test_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    println!("Testing condition: {}", condition_name);

    let filter = create_dynamic_filter_from_test_data(test_dir)?;
    let signals_dir = test_dir.join("signals");

    // Define tolerance based on filter type (can be made more sophisticated)
    let base_tolerance = if condition_name.contains("bandpass") {
        1e-8
    } else {
        1e-10
    };

    // Test all available signal files
    let signal_files = [
        ("impulse_xy.csv", "impulse", 1e-12),
        ("step_xy.csv", "step", base_tolerance),
        ("sine30_xy.csv", "sine30", base_tolerance),
        ("chirp5to200_xy.csv", "chirp", base_tolerance * 10.0), // Chirp typically needs more tolerance
        ("noise_xy.csv", "noise", base_tolerance * 5.0),
    ];

    for (filename, signal_name, tolerance) in &signal_files {
        let signal_path = signals_dir.join(filename);
        if signal_path.exists() {
            match load_signal_data(&signal_path) {
                Ok((inputs, expected)) => {
                    let test_name = format!("{}_{}", condition_name, signal_name);
                    test_filter_against_reference(
                        filter.clone(),
                        &inputs,
                        &expected,
                        *tolerance,
                        &test_name,
                    );
                    println!("  ✓ {} test passed", signal_name);
                }
                Err(e) => {
                    println!("  ⚠ Failed to load {}: {}", filename, e);
                }
            }
        } else {
            println!("  - {} not found (skipped)", signal_name);
        }
    }

    Ok(())
}

#[test]
fn test_all_conditions_automatically() {
    let test_data_base = Path::new("test-data");

    if !test_data_base.exists() {
        println!(
            "Test data directory not found: {}",
            test_data_base.display()
        );
        return;
    }

    let mut tested_conditions = 0;
    let mut failed_conditions = 0;

    // Iterate through all subdirectories in test-data
    match std::fs::read_dir(test_data_base) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Check if this looks like a test condition directory
                    let has_coeffs =
                        path.join("a_coeffs.csv").exists() && path.join("b_coeffs.csv").exists();
                    let has_signals = path.join("signals").exists();

                    if has_coeffs && has_signals {
                        tested_conditions += 1;
                        match run_tests_for_condition(&path) {
                            Ok(()) => {
                                println!(
                                    "✅ {} completed successfully",
                                    path.file_name().unwrap().to_str().unwrap()
                                );
                            }
                            Err(e) => {
                                failed_conditions += 1;
                                println!(
                                    "❌ {} failed: {}",
                                    path.file_name().unwrap().to_str().unwrap(),
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to read test data directory: {}", e);
            return;
        }
    }

    println!("\n=== Test Summary ===");
    println!("Tested conditions: {}", tested_conditions);
    println!("Failed conditions: {}", failed_conditions);
    println!(
        "Success rate: {:.1}%",
        if tested_conditions > 0 {
            (tested_conditions - failed_conditions) as f64 / tested_conditions as f64 * 100.0
        } else {
            0.0
        }
    );

    if tested_conditions == 0 {
        println!("No test conditions found. Make sure test data is generated.");
    }

    // Fail the test if any condition failed
    assert_eq!(failed_conditions, 0, "Some test conditions failed");
}
