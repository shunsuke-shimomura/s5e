use std::{cell::RefCell, fs, rc::Rc};

use astrodynamics::coordinate::{BodyVector, ECIVector};
use chrono::{NaiveDateTime, TimeDelta};
use control_system::integrator::Prediction;
use csv::Writer;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, UnitVector3, Vector3};
use s5e_lib::{
    sat::{
        VirtualMagFieldSim, VirtualMagFieldSimActuatorInput, VirtualMagFieldSimInterface,
        VirtualMagFieldSimSensorOutput, VirtualMagneticFieldModel,
    },
    SimInputTransfer, SimOutputTransfer,
};

#[test]
fn sp_mtq_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 128.0;
    let dt = 0.1;
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
            x: 0.00,
            y: 0.00,
            z: 0.00,
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
        let angular_momentum_body = sat.inertia * omega_body;
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
        .args(["run", "python", "scripts/plot_sp_mtq.py"])
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

// ============================================================================
// MTQ Sun Pointing FSW for Virtual Magnetic Field Simulation
// ============================================================================

/// FSW for MTQ-based sun pointing control with virtual magnetic field
/// This is a simplified FSW that directly uses ports without drivers
pub struct SPMtqFsw {
    pub time: c5a::Time,
    #[allow(dead_code)]
    timer: c5a::estimation::Timer,

    // Direct sensor ports (no drivers for simplicity)
    pub gyro_port: s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer_port: s5e_port::S5ESubscribePort<s5e_port::MagnetometerData>,
    pub sun_dir_port: s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,

    // MTQ output port
    pub mtq_ctrl_port: s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,

    // Controller gains
    p_gain: Matrix3<f64>,
    d_gain: Matrix3<f64>,
}

impl Default for SPMtqFsw {
    fn default() -> Self {
        Self::new()
    }
}

impl SPMtqFsw {
    pub fn new() -> Self {
        Self {
            time: c5a::Time::new(),
            timer: c5a::estimation::Timer::new(),

            gyro_port: s5e_port::S5ESubscribePort::new(),
            magnetometer_port: s5e_port::S5ESubscribePort::new(),
            sun_dir_port: s5e_port::S5ESubscribePort::new(),
            mtq_ctrl_port: s5e_port::S5EPublishPort::new(),

            // Controller gains (tuned for MTQ control)
            // Lower gains for stability with limited MTQ authority
            p_gain: Matrix3::identity() * 0.001,
            d_gain: Matrix3::identity() * 0.01,
        }
    }

    pub fn init(&mut self) {
        println!("SPMtqFsw initialized");
        *self = Self::new();
    }

    pub fn main_loop(&mut self, _dt: f64) {
        // Get sensor readings directly from ports
        let gyro_data = self.gyro_port.subscribe();
        let mag_data = self.magnetometer_port.subscribe();
        let sun_data = self.sun_dir_port.subscribe();

        // Calculate MTQ control
        if let (Some(mag), Some(gyro), Some(sun)) = (mag_data, gyro_data, sun_data) {
            // Convert magnetic field from nT to T
            let mag_field_nt = mag.magnetic_field;
            let mag_field = mag_field_nt * 1e-9;  // nT -> T
            let ang_vel = gyro.angular_velocity;
            let sun_dir = UnitVector3::new_normalize(sun.light_direction);

            // Sun pointing control: align +Z axis with sun direction
            // Error = sun_dir × target_axis (rotation needed to align Z with sun)
            // Sign convention: positive error requires positive torque
            let target_axis = Vector3::z();
            let error = sun_dir.cross(&target_axis);

            // P-D control for sun pointing with MTQ
            // Torque = P * error - D * omega
            let torque_p = self.p_gain * error;
            let torque_d = self.d_gain * ang_vel;
            let desired_torque = torque_p - torque_d;

            // Convert torque to magnetic moment: m = (B × τ) / |B|²
            // This gives the magnetic moment that produces torque closest to desired
            // (B is in Tesla, τ is in Nm, so m is in Am²)
            let mag_field_norm_sq = mag_field.norm_squared();
            if mag_field_norm_sq > 1e-20 {
                let moment = mag_field.cross(&desired_torque) / mag_field_norm_sq;
                self.mtq_ctrl_port.publish(s5e_port::MagnetorquerCtrlEvent {
                    magnetic_moment: moment,
                });
            }
        }
    }
}

pub struct SPMtqFswInputPortSet<'a> {
    pub gyro: &'a mut s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer: &'a mut s5e_port::S5ESubscribePort<s5e_port::MagnetometerData>,
    pub sun_direction: &'a mut s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,
}

impl<'a> SimInputTransfer<VirtualMagFieldSimSensorOutput<'a>> for SPMtqFswInputPortSet<'a> {
    fn transfer_from(&mut self, sensor_output: &VirtualMagFieldSimSensorOutput<'a>) {
        s5e_port::transfer(sensor_output.gyro, self.gyro);
        s5e_port::transfer(sensor_output.magnetometer, self.magnetometer);
        s5e_port::transfer(sensor_output.sun_sensor, self.sun_direction);
    }
}

pub struct SPMtqFswOutputPortSet<'a> {
    pub mtq_ctrl: &'a s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,
}

impl<'a> SimOutputTransfer<VirtualMagFieldSimActuatorInput<'a>> for SPMtqFswOutputPortSet<'a> {
    fn transfer_to(&self, actuator_input: &mut VirtualMagFieldSimActuatorInput<'a>) {
        s5e_port::transfer(self.mtq_ctrl, actuator_input.mtq_ctrl);
    }
}

impl VirtualMagFieldSimInterface for SPMtqFsw {
    type InputPortSet<'a> = SPMtqFswInputPortSet<'a>;
    type OutputPortSet<'a> = SPMtqFswOutputPortSet<'a>;

    fn init(&mut self) {
        self.init();
    }

    fn main_loop(&mut self, dt: f64) {
        self.main_loop(dt);
    }

    fn input_ports(&mut self) -> Self::InputPortSet<'_> {
        SPMtqFswInputPortSet {
            gyro: &mut self.gyro_port,
            magnetometer: &mut self.magnetometer_port,
            sun_direction: &mut self.sun_dir_port,
        }
    }

    fn output_ports(&mut self) -> Self::OutputPortSet<'_> {
        SPMtqFswOutputPortSet {
            mtq_ctrl: &self.mtq_ctrl_port,
        }
    }
}

#[test]
fn sp_mtq_virtual_magfield_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let end_time = 2000.0;
    let dt = 0.1;

    // Initial conditions
    let initial_attitude = UnitQuaternion::new_normalize(Quaternion::new(-1.0, 4.0, -3.0, 2.0));
    let initial_angular_velocity = BodyVector {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698,
        0.001303, 0.183472, 0.000542,
        0.000698, 0.000542, 0.111208,
    );

    // Sun direction in ECI (fixed) - not parallel to initial magnetic field
    let sun_direction_eci = ECIVector {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    // Virtual magnetic field model
    // Simulates magnetic field rotation similar to LEO orbit (~90 min period)
    let magnetic_field_model = VirtualMagneticFieldModel::new(
        40000.0,  // 40000 nT (typical LEO)
        2.0 * std::f64::consts::PI / 5400.0,  // ~90 min period
        Vector3::new(0.0, 0.0, 1.0),  // Rotation around Z-axis
        Vector3::new(1.0, 0.0, 0.0),  // Initial direction along X
    );

    // MTQ max dipole moment (Am²)
    let mtq_max_dipole_moment = 1.0;

    // Create FSW and simulation
    let fsw = SPMtqFsw::new();
    let mut sim = VirtualMagFieldSim::new(
        inertia,
        fsw,
        initial_attitude,
        initial_angular_velocity,
        sun_direction_eci.clone(),
        magnetic_field_model,
        mtq_max_dipole_moment,
    );

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/sp_mtq_virtual")?;
    let mut writer = Writer::from_path("tests/out/sp_mtq_virtual/sun_direction_error.csv")?;

    while time < end_time {
        // Publish sensor data to FSW
        let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
        let omega = sim.angular_velocity.get_now();

        // Publish gyro data
        sim.gyro_port.publish(s5e_port::GyroSensorData {
            angular_velocity: Vector3::new(omega.x, omega.y, omega.z),
        });

        // Publish sun sensor data (sun direction in body frame)
        let sun_dir_body = BodyVector::from_eci(sun_direction_eci.clone(), attitude_q);
        sim.sun_sensor_port.publish(s5e_port::LightDirectionData {
            light_direction: Vector3::new(sun_dir_body.x, sun_dir_body.y, sun_dir_body.z),
        });

        // Publish magnetometer data (magnetic field in body frame)
        let mag_field_body = sim.magnetic_field_body();
        sim.magnetometer_port.publish(s5e_port::MagnetometerData {
            magnetic_field: Vector3::new(mag_field_body.x, mag_field_body.y, mag_field_body.z),
        });

        // Tick simulation
        sim.tick(dt);

        // Calculate sun direction error
        let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
        let sat_z_axis_eci = attitude_q.conjugate() * Vector3::z();
        let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
        let angle_error = sat_z_axis_eci.angle(&sun_dir_eci);

        // Calculate angular velocity and angular momentum
        let omega = sim.angular_velocity.get_now();
        let omega_vec = Vector3::new(omega.x, omega.y, omega.z);
        let omega_norm = omega_vec.norm();
        let angular_momentum_body = inertia * omega_vec;
        let angular_momentum_eci = attitude_q.conjugate() * angular_momentum_body;
        let _angular_momentum_norm = angular_momentum_eci.norm();

        // Get magnetic field direction in ECI
        let mag_field_eci = sim.magnetic_field_eci();
        let mag_field_dir = Vector3::new(mag_field_eci.x, mag_field_eci.y, mag_field_eci.z).normalize();

        // Control error: sat_z × sun_dir (in ECI)
        let control_error_eci = sat_z_axis_eci.cross(&sun_dir_eci);

        // Write to CSV
        writer.write_record(&[
            time.to_string(),
            angle_error.to_degrees().to_string(),
            sat_z_axis_eci.x.to_string(),
            sat_z_axis_eci.y.to_string(),
            sat_z_axis_eci.z.to_string(),
            sun_dir_eci.x.to_string(),
            sun_dir_eci.y.to_string(),
            sun_dir_eci.z.to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
            mag_field_dir.x.to_string(),
            mag_field_dir.y.to_string(),
            mag_field_dir.z.to_string(),
            control_error_eci.x.to_string(),
            control_error_eci.y.to_string(),
            control_error_eci.z.to_string(),
            omega_norm.to_string(),
        ])?;

        // Update time
        time += dt;
        sim.clear_state();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_sp_mtq_virtual.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    // Check convergence
    let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
    let sat_z_axis_eci = attitude_q.conjugate() * Vector3::z();
    let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
    let final_angle_error = sat_z_axis_eci.angle(&sun_dir_eci);

    println!("Final sun pointing error: {:.2} deg", final_angle_error.to_degrees());

    if final_angle_error < 30.0_f64.to_radians() {
        println!("Sun Pointing MTQ (virtual magfield) convergence test passed.");
        Ok(())
    } else {
        anyhow::bail!(
            "Sun Pointing MTQ (virtual magfield) convergence test failed. Final angle: {:.2} deg",
            final_angle_error.to_degrees()
        );
    }
} 