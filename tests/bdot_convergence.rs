use std::{cell::RefCell, fs, rc::Rc};

use astrodynamics::coordinate::BodyVector;
use chrono::{NaiveDateTime, TimeDelta};
use control_system::integrator::Prediction;
use csv::Writer;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use s5e_port::{S5EPublishPort, S5ESubscribePort};

#[test]
fn bdot_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 450.0;
    let dt = 0.01;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s5e_lib::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );
    let mut sun = s5e_lib::sun::Sun::new(datetime);
    let mut earth = s5e_lib::earth::Earth::new(datetime);
    let mut moon = s5e_lib::moon::Moon::new(datetime);

    let sat_initial_state = c5a_sim::SpaceCraftInitialState {
        tle: tle.clone(),
        attitude: UnitQuaternion::new_normalize(Quaternion::new(
            -1.0, 4.0, -3.0, 2.0,
        )),
        angular_velocity: BodyVector {
            x: 0.01,
            y: 0.005,
            z: 0.007,
        },
        inertia: Matrix3::new(
            0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542,
            0.111208,
        ),
    };

    let components_profile = c5a_sim::SpaceCraftComponentsProfile {
        gyro_std: c5a::constants::GYRO_NOISE_STD,
        magnetometer_std: c5a::constants::MAGNETOMETER_NOISE_STD,
        star_tracker_std: c5a::constants::STAR_TRACKER_NOISE_STD,
        sun_sensor_dir_std: c5a::constants::SUN_SENSOR_NOISE_STD,
        rw_noise_std: 0.0,
        mtq_noise_std: 0.0,
        mtq_max_dipole_moment: c5a::constants::MTQ_MAX_DIPOLE_MOMENT,
    };

    let mut sat = c5a_sim::SpaceCraft::new(
        datetime,
        rng,
        c5a::Fsw::new(false),
        components_profile,
        sat_initial_state,
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

        if let c5a::controller::AttitudeControllMode::MTQ(mtq) =
            &sat.obc.fsw.attitude_controller.mode
            && matches!(
                mtq.mode,
                c5a::controller::mtq::MTQControlMode::SunPointing(_)
            )
        {
            println!(
                "Converged to sun pointing mode at time {:.2} sec (datetime: {})",
                time, datetime
            );
            break;
        }

        // Record angular velocity and angular momentum data
        let omega = sat.angular_velocity.get_now();
        let omega_vec = Vector3::new(omega.x, omega.y, omega.z);
        let omega_norm = omega_vec.norm();
        let angular_momentum = sat.inertia * omega_vec;
        let angular_momentum_norm = angular_momentum.norm();
        writer.write_record(&[
            time.to_string(),
            omega.x.to_string(),
            omega.y.to_string(),
            omega.z.to_string(),
            omega_norm.to_string(),
            angular_momentum.x.to_string(),
            angular_momentum.y.to_string(),
            angular_momentum.z.to_string(),
            angular_momentum_norm.to_string(),
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
        .args(["run", "python", "scripts/plot_bdot.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    if let c5a::controller::AttitudeControllMode::MTQ(mtq) =
        &sat.obc.fsw.attitude_controller.mode
        && matches!(
            mtq.mode,
            c5a::controller::mtq::MTQControlMode::SunPointing(_)
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

pub struct BdotFsw {
    pub time: c5a::Time,
    timer: c5a::estimation::Timer,

    pub gyro_driver: c5a::driver::sensor::GyroDriver,
    pub magnetometer_driver: c5a::driver::sensor::MagnetometerDriver,
    pub magnetorquer_driver: c5a::driver::actuator::MagnetorquerDriver,

    pub bdot_controller: c5a::controller::mtq::BdotDetumblingController,
}

impl Default for BdotFsw {
    fn default() -> Self {
        Self::new()
    }
}

impl BdotFsw {
    pub fn new() -> Self {
        Self {
            time: c5a::Time::new(),
            timer: c5a::estimation::Timer::new(),

            gyro_driver: c5a::driver::sensor::GyroDriver::new(
                UnitQuaternion::identity(),
            ),
            magnetometer_driver: c5a::driver::sensor::MagnetometerDriver::new(
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_2)
            ),
            magnetorquer_driver: c5a::driver::actuator::MagnetorquerDriver::new(),

            bdot_controller: c5a::controller::mtq::BdotDetumblingController::new(),
        }
    }

    pub fn init(&mut self) {
        println!("BdotFsw initialized");
        *self = Self::new();
    }

    pub fn main_loop(&mut self, dt: f64) {
        self.time = self.timer.main_loop(dt, None);

        // Update sensor readings
        let gyro_measurement = self.gyro_driver.main_loop();
        let magnetometer_measurement = self.magnetometer_driver.main_loop(dt);

        // B-dot control
        let mtq_control = if let(Some(mag_data), Some(gyro_data)) = (magnetometer_measurement.magnetic_field, gyro_measurement) {
            Some(self.bdot_controller.magnetic_moment(&mag_data.magnetic_field, &gyro_data.angular_velocity))
        } else {
            // Handle the case where sensor data is not available
            // For example, return a default magnetic moment or take other appropriate action
            None
        };        

        // Command magnetorquers
        self.magnetorquer_driver.main_loop(c5a::driver::actuator::MagnetorquerDriverInput {
            moment: mtq_control.map(|moment| c5a::data::MagnetorquerCtrlEvent {
                magnetic_moment: moment,
            }),
            exclusive_ctrl: magnetometer_measurement.exclusive_ctrl,

        });
    }
}

pub struct BdotFswInputPortSet<'a> {
    pub gyro: &'a mut S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer: &'a mut S5ESubscribePort<s5e_port::MagnetometerData>,
}

impl<'a> s5e_lib::SimInputTransfer<c5a_sim::SensorOutputPortSet<'a>> for BdotFswInputPortSet<'a> {
    fn transfer_from(&mut self, sensor_output: &c5a_sim::SensorOutputPortSet<'a>) {
        s5e_port::transfer(sensor_output.gyro, self.gyro);
        s5e_port::transfer(sensor_output.magnetometer, self.magnetometer);
    }
}

pub struct BdotFswOutputPortSet<'a> {
    pub magnetorquer_ctrl: &'a S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,
}

impl<'a> s5e_lib::SimOutputTransfer<c5a_sim::ActuatorInputPortSet<'a>> for BdotFswOutputPortSet<'a> {
    fn transfer_to(&self, actuator_input: &mut c5a_sim::ActuatorInputPortSet<'a>) {
        s5e_port::transfer(self.magnetorquer_ctrl, actuator_input.magnetorquer_ctrl);
    }
}

impl c5a_sim::C5ASimInterface for BdotFsw {
    type InputPortSet<'a> = BdotFswInputPortSet<'a> where Self: 'a;
    type OutputPortSet<'a> = BdotFswOutputPortSet<'a> where Self: 'a;
    
    fn init(&mut self) {
        self.init();
    }
    fn main_loop(&mut self, dt: f64) {
        self.main_loop(dt);
    }
    fn input_ports(&mut self) -> Self::InputPortSet<'_> {
        BdotFswInputPortSet {
            gyro: &mut self.gyro_driver.sim_port,
            magnetometer: &mut self.magnetometer_driver.sim_port,
        }
    }
    fn output_ports(&'_ self) -> Self::OutputPortSet<'_> {
        BdotFswOutputPortSet {
            magnetorquer_ctrl: &self.magnetorquer_driver.sim_port,
        }
    }
}

#[test]
fn bdot_fsw_convergence() -> anyhow::Result<()> {
    let debug_config = debug_s5e::DebugConfig {
        sensor_data: false,
        actuator_data: false,
    };
    debug_s5e::init_debug_config(debug_config);
    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 45000.0;
    let dt = 0.01;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s5e_lib::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );

    let sat_initial_state = c5a_sim::SpaceCraftInitialState {
        tle: tle.clone(),
        attitude: UnitQuaternion::new_normalize(Quaternion::new(
            -1.0, 4.0, -3.0, 2.0,
        )),
        angular_velocity: BodyVector {
            x: 0.01,
            y: 0.005,
            z: 0.007,
        },
        inertia: Matrix3::new(
            0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542,
            0.111208,
        ),
    };

    let components_profile = c5a_sim::SpaceCraftComponentsProfile {
        gyro_std: 0.0,
        magnetometer_std: 0.0,
        star_tracker_std: 0.0,
        sun_sensor_dir_std: 0.0,
        rw_noise_std: 0.0,
        mtq_noise_std: 0.0,
        mtq_max_dipole_moment: 1e1,
    };

    let mut sat = c5a_sim::SpaceCraft::new(
        datetime,
        rng,
        BdotFsw::new(),
        components_profile,
        sat_initial_state,
    );

    let mut cmd_port = s5e_port::S5EPublishPort::<s5e_port::GSCommandData>::new();

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/bdot/bdot_fsw")?;
    let mut writer = Writer::from_path("tests/out/bdot/bdot_fsw/angular_velocity.csv")?;

    let mut sun = s5e_lib::sun::Sun::new(datetime);

    while time < end_time {
        sun.tick(dt, datetime);
        // tick satellite
        sat.tick(
            dt,
            datetime,
            sun.light_source(),
            vec![],
            &cmd_port,
        );

        // Record angular velocity and angular momentum data
        let omega = sat.angular_velocity.get_now();
        let omega_vec = Vector3::new(omega.x, omega.y, omega.z);
        let omega_norm = omega_vec.norm();
        let angular_momentum = sat.inertia * omega_vec;
        let angular_momentum_norm = angular_momentum.norm();
        writer.write_record(&[
            time.to_string(),
            omega.x.to_string(),
            omega.y.to_string(),
            omega.z.to_string(),
            omega_norm.to_string(),
            angular_momentum.x.to_string(),
            angular_momentum.y.to_string(),
            angular_momentum.z.to_string(),
            angular_momentum_norm.to_string(),
        ])?;

        // update time
        time += dt;
        datetime += TimeDelta::nanoseconds((dt * 1.0e9) as i64);
        // Clear states for next iteration
        sat.clear_state();
        sun.clear_state();
        cmd_port.clear();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_bdot_fsw.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }
    Ok(())
}
