use std::{cell::RefCell, fs, rc::Rc};

use astrodynamics::coordinate::BodyVector;
use chrono::{NaiveDateTime, TimeDelta};
use control_system::integrator::Prediction;
use csv::Writer;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

#[test]
fn bdot_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 450.0;
    let dt = 0.01;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s5e::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );
    let mut sun = s5e::sun::Sun::new(datetime);
    let mut earth = s5e::earth::Earth::new(datetime);
    let mut moon = s5e::moon::Moon::new(datetime);

    let mut sat = s5e::SpaceCraft::new(
        &tle,
        datetime,
        rng,
        // Some(UnitQuaternion::identity()),
        Some(UnitQuaternion::new_normalize(Quaternion::new(
            -1.0, 4.0, -3.0, 2.0,
        ))),
        Some(Matrix3::new(
            0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542,
            0.111208,
        )),
        Some(BodyVector {
            x: 0.01,
            y: 0.005,
            z: 0.007,
        }),
        false,
    );

    let mut cmd_port = s5e_port::S5EPublishPort::<s5e_port::GSCommandData>::new();

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/bdot")?;
    let mut writer = Writer::from_path("tests/out/bdot/angular_velocity.csv")?;

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

        // Record angular velocity data
        let omega = sat.angular_velocity.get_now();
        let omega_vec = Vector3::new(omega.x, omega.y, omega.z);
        let omega_norm = omega_vec.norm();
        writer.write_record(&[
            time.to_string(),
            omega.x.to_string(),
            omega.y.to_string(),
            omega.z.to_string(),
            omega_norm.to_string(),
        ])?;

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
        .args(["run", "python", "plot_bdot.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    if let fsw_s5e::controller::AttitudeControllMode::MTQ(mtq) =
        &sat.obc.fsw.attitude_controller.mode
        && matches!(
            mtq.mode,
            fsw_s5e::controller::mtq::MTQControlMode::SunPointing(_)
        )
    {
        println!(
            "Converged to sun pointing mode at time {:.2} sec (datetime: {})",
            time, datetime
        );
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Failed to converge to sun pointing mode within the given time."
        ))
    }
}
