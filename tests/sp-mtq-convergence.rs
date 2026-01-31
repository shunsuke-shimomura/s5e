use std::{cell::RefCell, fs, rc::Rc};

use astrodynamics::coordinate::BodyVector;
use chrono::{NaiveDateTime, TimeDelta};
use control_system::integrator::Prediction;
use csv::Writer;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

#[test]
fn sp_mtq_convergence_from_0s() -> anyhow::Result<()> {
    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 128000.0;
    let dt = 0.1;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s4e::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );
    let mut sun = s4e::sun::Sun::new(datetime);
    let mut earth = s4e::earth::Earth::new(datetime);
    let mut moon = s4e::moon::Moon::new(datetime);

    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542, 0.111208,
    );
    let mut sat = s4e::SpaceCraft::new(
        &tle,
        datetime,
        rng,
        // Some(UnitQuaternion::identity()),
        Some(UnitQuaternion::new_normalize(Quaternion::new(
            -1.0, 4.0, -3.0, 2.0,
        ))),
        Some(inertia),
        Some(BodyVector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }),
        false,
    );

    let mut cmd_port = s4e_port::S4EPublishPort::<s4e_port::GSCommandData>::new();

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/sp_mtq")?;
    let mut writer = Writer::from_path("tests/out/sp_mtq/sun_direction_error.csv")?;

    while time < end_time {
        // tick earth
        earth.tick(dt, datetime);
        // tick moon
        moon.tick(dt, datetime);
        // tick sun
        sun.tick(dt, datetime);

        // tick satellite
        sat.tick(
            dt,
            datetime,
            sun.light_source(),
            vec![earth.shadow_source(), moon.shadow_source()],
            &cmd_port,
        );

        // Calculate sun direction error
        let attitude_q = UnitQuaternion::from_quaternion(sat.attitude.get_now());
        let sat_z_axis = attitude_q.conjugate() * Vector3::z();
        let sun_dir = Vector3::from(sun.position.get_now() - sat.eci_position.get_now()).normalize();
        let angle_error = sat_z_axis.angle(&sun_dir);

        // Calculate angular momentum in ECI frame: H = R * I * ω
        // where R is body-to-ECI rotation, I is inertia, ω is angular velocity in body
        let omega_body = Vector3::new(
            sat.angular_velocity.get_now().x,
            sat.angular_velocity.get_now().y,
            sat.angular_velocity.get_now().z,
        );
        let angular_momentum_body = inertia * omega_body;
        let angular_momentum_eci = attitude_q.conjugate() * angular_momentum_body;

        // Get magnetic field direction in ECI (already in nT)
        let mag_field_eci = Vector3::new(
            sat.magnetic_field.x,
            sat.magnetic_field.y,
            sat.magnetic_field.z,
        );
        let mag_field_dir = mag_field_eci.normalize();

        // Calculate control error vector: sat_z × sun_dir (in ECI)
        // This is proportional to the torque needed to align sat_z with sun_dir
        let control_error_eci = sat_z_axis.cross(&sun_dir);

        // Record sun direction error data with angular momentum and magnetic field
        writer.write_record(&[
            time.to_string(),
            angle_error.to_degrees().to_string(),
            sat_z_axis.x.to_string(),
            sat_z_axis.y.to_string(),
            sat_z_axis.z.to_string(),
            sun_dir.x.to_string(),
            sun_dir.y.to_string(),
            sun_dir.z.to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
            mag_field_dir.x.to_string(),
            mag_field_dir.y.to_string(),
            mag_field_dir.z.to_string(),
            control_error_eci.x.to_string(),
            control_error_eci.y.to_string(),
            control_error_eci.z.to_string(),
            sat.shadow_coefficient.value.to_string(),
        ])?;

        // Dump status at around 13000 seconds for debugging
        if (time - 13000.0_f64).abs() < dt {
            let q = sat.attitude.get_now();
            let omega = sat.angular_velocity.get_now();
            println!("\n=== Status dump at t = {:.1} sec ===", time);
            println!("Datetime: {}", datetime);
            println!("Attitude quaternion (w, x, y, z): ({:.10}, {:.10}, {:.10}, {:.10})", q.w, q.i, q.j, q.k);
            println!("Angular velocity (body, rad/s): ({:.10}, {:.10}, {:.10})", omega.x, omega.y, omega.z);
            println!("Angle error: {:.4} deg", angle_error.to_degrees());
            println!("Sat Z-axis (ECI): ({:.6}, {:.6}, {:.6})", sat_z_axis.x, sat_z_axis.y, sat_z_axis.z);
            println!("Sun direction (ECI): ({:.6}, {:.6}, {:.6})", sun_dir.x, sun_dir.y, sun_dir.z);
            println!("Angular momentum (ECI, kg·m²/s): ({:.10}, {:.10}, {:.10})", angular_momentum_eci.x, angular_momentum_eci.y, angular_momentum_eci.z);
            println!("Magnetic field dir (ECI): ({:.6}, {:.6}, {:.6})", mag_field_dir.x, mag_field_dir.y, mag_field_dir.z);
            println!("Control error (sat_z × sun): ({:.6}, {:.6}, {:.6})", control_error_eci.x, control_error_eci.y, control_error_eci.z);
            println!("Shadow coefficient: {:.4}", sat.shadow_coefficient.value);
            println!("===================================\n");
        }

        // update time
        time += dt;
        datetime += TimeDelta::nanoseconds((dt * 1.0e9) as i64);
        // Clear states for next iteration
        sat.clear_state();
        sun.clear_state();
        earth.clear_state();
        moon.clear_state();
        cmd_port.clear();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "plot_sp_mtq.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    let sat_z_axis = UnitQuaternion::from_quaternion(sat.attitude.get_now()).conjugate()
        * Vector3::z();
    let sun_dir = Vector3::from(sun.position.get_now());
    let angle = sat_z_axis.angle(&sun_dir);
    if angle < 10.0f64.to_radians() {
        println!("Sun Pointing MTQ convergence test passed.");
        Ok(())
    } else {
        anyhow::bail!(
            "Sun Pointing MTQ convergence test failed. Final angle: {} deg",
            angle.to_degrees()
        );
    }
}

/// Test starting from conditions at t=13000s (from main convergence test)
/// Use this to investigate why perpendicular error can't be reduced after ~12000s
#[test]
fn sp_mtq_convergence_from_13000s() -> anyhow::Result<()> {
    // Initial conditions captured from t=13000s
    // These values should be updated after running the main test and getting the dump
    let initial_time = 13000.0;
    let initial_datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
            + TimeDelta::seconds(13000);

    // Values from status dump at t=13000s
    // Attitude quaternion (w, x, y, z): (0.9192171340, 0.2147177715, -0.3255651373, 0.0542597631)
    // Angular velocity (body, rad/s): (0.0001291136, 0.0036133553, -0.0588619138)
    // Angle error at this point: 42.68 deg
    let initial_attitude = UnitQuaternion::new_normalize(Quaternion::new(
        0.9192171340,  // w
        0.2147177715,  // x (i)
        -0.3255651373, // y (j)
        0.0542597631,  // z (k)
    ));
    let initial_angular_velocity = BodyVector {
        x: 0.0001291136,
        y: 0.0036133553,
        z: -0.0588619138,
    };

    let mut time = initial_time;
    let mut datetime = initial_datetime;
    let end_time = 18000.0; // Run for 5000 more seconds
    let dt = 0.1;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s4e::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );
    let mut sun = s4e::sun::Sun::new(initial_datetime);
    let mut earth = s4e::earth::Earth::new(initial_datetime);
    let mut moon = s4e::moon::Moon::new(initial_datetime);

    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542, 0.111208,
    );
    let mut sat = s4e::SpaceCraft::new(
        &tle,
        initial_datetime,
        rng,
        Some(initial_attitude),
        Some(inertia),
        Some(initial_angular_velocity),
        true, // Enable FSW debug output
    );

    let mut cmd_port = s4e_port::S4EPublishPort::<s4e_port::GSCommandData>::new();

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/sp_mtq_from_13000")?;
    let mut writer = Writer::from_path("tests/out/sp_mtq_from_13000/sun_direction_error.csv")?;

    println!("Starting simulation from t={}s", initial_time);

    while time < end_time {
        // tick earth
        earth.tick(dt, datetime);
        // tick moon
        moon.tick(dt, datetime);
        // tick sun
        sun.tick(dt, datetime);

        // tick satellite
        sat.tick(
            dt,
            datetime,
            sun.light_source(),
            vec![earth.shadow_source(), moon.shadow_source()],
            &cmd_port,
        );

        // Calculate sun direction error
        let attitude_q = UnitQuaternion::from_quaternion(sat.attitude.get_now());
        let sat_z_axis = attitude_q.conjugate() * Vector3::z();
        let sun_dir =
            Vector3::from(sun.position.get_now() - sat.eci_position.get_now()).normalize();
        let angle_error = sat_z_axis.angle(&sun_dir);

        // Calculate angular momentum in ECI frame
        let omega_body = Vector3::new(
            sat.angular_velocity.get_now().x,
            sat.angular_velocity.get_now().y,
            sat.angular_velocity.get_now().z,
        );
        let angular_momentum_body = inertia * omega_body;
        let angular_momentum_eci = attitude_q.conjugate() * angular_momentum_body;

        // Get magnetic field direction in ECI
        let mag_field_eci = Vector3::new(
            sat.magnetic_field.x,
            sat.magnetic_field.y,
            sat.magnetic_field.z,
        );
        let mag_field_dir = mag_field_eci.normalize();

        // Calculate control error vector
        let control_error_eci = sat_z_axis.cross(&sun_dir);

        // Decompose control error into B-parallel and B-perpendicular
        let ctrl_err_parallel = control_error_eci.dot(&mag_field_dir);
        let ctrl_err_perp_vec = control_error_eci - ctrl_err_parallel * mag_field_dir;
        let ctrl_err_perpendicular = ctrl_err_perp_vec.norm();

        // Record data
        writer.write_record(&[
            time.to_string(),
            angle_error.to_degrees().to_string(),
            sat_z_axis.x.to_string(),
            sat_z_axis.y.to_string(),
            sat_z_axis.z.to_string(),
            sun_dir.x.to_string(),
            sun_dir.y.to_string(),
            sun_dir.z.to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
            mag_field_dir.x.to_string(),
            mag_field_dir.y.to_string(),
            mag_field_dir.z.to_string(),
            control_error_eci.x.to_string(),
            control_error_eci.y.to_string(),
            control_error_eci.z.to_string(),
            sat.shadow_coefficient.value.to_string(),
        ])?;

        // Print status every 100 seconds
        if ((time - initial_time) % 100.0_f64).abs() < dt {
            println!(
                "t={:.0}s: angle_err={:.2}deg, |H|={:.6}, shadow={:.2}, ctrl_err_parallel={:.4}, ctrl_err_perp={:.4}",
                time,
                angle_error.to_degrees(),
                angular_momentum_eci.norm(),
                sat.shadow_coefficient.value,
                ctrl_err_parallel.abs(),
                ctrl_err_perpendicular
            );
        }

        // update time
        time += dt;
        datetime += TimeDelta::nanoseconds((dt * 1.0e9) as i64);
        // Clear states for next iteration
        sat.clear_state();
        sun.clear_state();
        earth.clear_state();
        moon.clear_state();
        cmd_port.clear();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "plot_sp_mtq.py", "--dir", "out/sp_mtq_from_13000"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    let sat_z_axis =
        UnitQuaternion::from_quaternion(sat.attitude.get_now()).conjugate() * Vector3::z();
    let sun_dir = Vector3::from(sun.position.get_now());
    let angle = sat_z_axis.angle(&sun_dir);

    println!("\nFinal angle error: {:.2} deg", angle.to_degrees());

    Ok(())
}
