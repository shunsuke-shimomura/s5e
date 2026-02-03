use std::fs;

use astrodynamics::coordinate::{BodyVector, ECIVector};
use control_system::integrator::Prediction;
use csv::Writer;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, UnitVector3, Vector3};
use s5e_lib::{SimInputTransfer, SimOutputTransfer, sat::{IdealAttitudeSim, IdealAttitudeSimActuatorInput, IdealAttitudeSimInterface, IdealAttitudeSimSensorOutput}};

pub struct SPIdealFsw {
    pub time: c5a::Time,
    timer: c5a::estimation::Timer,

    pub angular_velocity_port: s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub sun_dir_port: s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,

    pub ideal_torque_port: s5e_port::S5EPublishPort<s5e_port::IdealTorquerCtrlEvent>,

    pub pointing_controller: c5a::controller::SunPointingController,
}

impl SPIdealFsw {
    fn new() -> Self {
        Self {
            time: c5a::Time::new(),
            timer: c5a::estimation::Timer::new(),

            angular_velocity_port: s5e_port::S5ESubscribePort::new(),
            sun_dir_port: s5e_port::S5ESubscribePort::new(),

            ideal_torque_port: s5e_port::S5EPublishPort::new(),

            pointing_controller: c5a::controller::SunPointingController::new(
                nalgebra::Matrix3::identity() * 0.1,
                nalgebra::Matrix3::identity(),
            ),
        }
    }
    fn init(&mut self) {
        println!("SPIdealFsw initialized.");
        *self = Self::new();
    }

    pub fn main_loop(&mut self, dt: f64) {
        self.time = self.timer.main_loop(dt, None);
        let angular_velocity = self
            .angular_velocity_port
            .subscribe()
            .map(|data| data.angular_velocity);
        let sun_direction = self
            .sun_dir_port
            .subscribe()
            .map(|data| UnitVector3::new_normalize(data.light_direction));
        if let (Some(sun_dir), Some(ang_vel)) = (sun_direction, angular_velocity) {
            let torque = self
                .pointing_controller
                .torque(&sun_dir, &ang_vel);
            self.ideal_torque_port.publish(s5e_port::IdealTorquerCtrlEvent { torque });
        }
    }
}

pub struct SPIdealFswInputPortSet<'a> {
    pub angular_velocity: &'a mut  s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub sun_direction: &'a mut s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,
}

impl SimInputTransfer<IdealAttitudeSimSensorOutput<'_>> for SPIdealFswInputPortSet<'_> {
    fn transfer_from(&mut self, sensor_output: &IdealAttitudeSimSensorOutput<'_>) {
        s5e_port::transfer(sensor_output.gyro, self.angular_velocity);
        s5e_port::transfer(sensor_output.sun_sensor, self.sun_direction);
    }
}

pub struct SPIdealFswOutputPortSet<'a> {
    pub ideal_torque: &'a s5e_port::S5EPublishPort<s5e_port::IdealTorquerCtrlEvent>,
}

impl SimOutputTransfer<IdealAttitudeSimActuatorInput<'_>> for SPIdealFswOutputPortSet<'_> {
    fn transfer_to(&self, actuator_input: &mut IdealAttitudeSimActuatorInput<'_>) {
        s5e_port::transfer(self.ideal_torque, actuator_input.torquer_ctrl);
    }
}

impl IdealAttitudeSimInterface for SPIdealFsw {
    type InputPortSet<'a> = SPIdealFswInputPortSet<'a>;
    type OutputPortSet<'a> = SPIdealFswOutputPortSet<'a>;

    fn input_ports(&mut self) -> Self::InputPortSet<'_> {
        SPIdealFswInputPortSet {
            angular_velocity: &mut self.angular_velocity_port,
            sun_direction: &mut self.sun_dir_port,
        }
    }

    fn output_ports(&mut self) -> Self::OutputPortSet<'_> {
        SPIdealFswOutputPortSet {
            ideal_torque: &self.ideal_torque_port,
        }
    }
    fn init(&mut self) {
        self.init();
    }
    fn main_loop(&mut self, dt: f64) {
        self.main_loop(dt);
    }
}

#[test]
fn sp_ideal_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let end_time = 100.0;
    let dt = 0.01;

    // Initial conditions
    let initial_attitude = UnitQuaternion::new_normalize(Quaternion::new(-1.0, 4.0, -3.0, 2.0));
    let initial_angular_velocity = BodyVector {
        x: 0.01,
        y: 0.005,
        z: 0.007,
    };
    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698,
        0.001303, 0.183472, 0.000542,
        0.000698, 0.000542, 0.111208,
    );

    // Sun direction in ECI (fixed for ideal simulation)
    let sun_direction_eci = ECIVector {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    // Create FSW and simulation
    let fsw = SPIdealFsw::new();
    let mut sim = IdealAttitudeSim::new(
        inertia,
        fsw,
        initial_attitude,
        initial_angular_velocity,
        sun_direction_eci.clone(),
    );

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/sp_ideal")?;
    let mut writer = Writer::from_path("tests/out/sp_ideal/sun_direction_error.csv")?;

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

        // Tick simulation (this calls FSW main_loop and propagates dynamics)
        sim.tick(dt);

        // Calculate sun direction error (satellite Z-axis vs sun direction in ECI)
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
        let angular_momentum_norm = angular_momentum_eci.norm();

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
            omega.x.to_string(),
            omega.y.to_string(),
            omega.z.to_string(),
            omega_norm.to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
            angular_momentum_norm.to_string(),
        ])?;

        // Update time
        time += dt;
        sim.clear_state();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_sp_ideal.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    // Check convergence (final angle error < 10 degrees)
    let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
    let sat_z_axis_eci = attitude_q.conjugate() * Vector3::z();
    let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
    let final_angle_error = sat_z_axis_eci.angle(&sun_dir_eci);

    println!("Final sun pointing error: {:.2} deg", final_angle_error.to_degrees());

    if final_angle_error < 10.0_f64.to_radians() {
        println!("Sun Pointing Ideal convergence test passed.");
        Ok(())
    } else {
        anyhow::bail!(
            "Sun Pointing Ideal convergence test failed. Final angle: {:.2} deg",
            final_angle_error.to_degrees()
        );
    }
}