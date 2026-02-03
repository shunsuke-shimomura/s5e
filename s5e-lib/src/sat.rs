use astrodynamics::coordinate::{BodyVector, ECIVector};
use chrono::NaiveDateTime;
use control_system::integrator::{Prediction, TimeIntegrator, rk4::RK4Solver};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

use crate::{SimInputTransfer, SimOutputTransfer};

pub struct IdealAttitudeSim<Fsw> {
    pub inertia: Matrix3<f64>,

    pub fsw: Fsw,

    pub attitude: RK4Solver<Quaternion<f64>, NaiveDateTime, f64>,
    pub angular_velocity: RK4Solver<BodyVector, NaiveDateTime, f64>,
    pub torque: BodyVector,

    pub gyro_port: s5e_port::S5EPublishPort<s5e_port::GyroSensorData>,
    pub sun_sensor_port: s5e_port::S5EPublishPort<s5e_port::LightDirectionData>,
    pub star_tracker_port: s5e_port::S5EPublishPort<s5e_port::StarTrackerData>,
    
    pub torquer_ctrl_port: s5e_port::S5ESubscribePort<s5e_port::IdealTorquerCtrlEvent>,

    pub sun_vector_body: BodyVector,
}

impl<Fsw> IdealAttitudeSim<Fsw> {
    pub fn new(
        inertia: Matrix3<f64>,
        fsw: Fsw,
        initial_attitude: UnitQuaternion<f64>,
        initial_angular_velocity: BodyVector,
        inertial_sun_vector: ECIVector,
    ) -> Self {
        let initial_sun_vector_body = BodyVector::from_eci(inertial_sun_vector, initial_attitude);
        Self {
            inertia,
            fsw,
            attitude: RK4Solver::new(*initial_attitude.quaternion()),
            angular_velocity: RK4Solver::new(initial_angular_velocity),
            torque: BodyVector { x: 0.0, y: 0.0, z: 0.0 },
            sun_vector_body: initial_sun_vector_body,
            gyro_port: s5e_port::S5EPublishPort::new(),
            sun_sensor_port: s5e_port::S5EPublishPort::new(),
            star_tracker_port: s5e_port::S5EPublishPort::new(),
            torquer_ctrl_port: s5e_port::S5ESubscribePort::new(),
        }
    }

    pub fn tick(
        &mut self,
        dt: f64,
    )
    where
        Fsw: IdealAttitudeSimInterface
    {
        {
            let sensor_ports = IdealAttitudeSimSensorOutput {
                gyro: &self.gyro_port,
                sun_sensor: &self.sun_sensor_port,
                star_tracker: &self.star_tracker_port,
            };
            let mut fsw_input_ports = self.fsw.input_ports();
            fsw_input_ports.transfer_from(&sensor_ports);
        }
        self.fsw.main_loop(dt);
        {
            let fsw_output_ports = self.fsw.output_ports();
            let mut actuator_input_ports = IdealAttitudeSimActuatorInput {
                torquer_ctrl: &mut self.torquer_ctrl_port,
            };
            fsw_output_ports.transfer_to(&mut actuator_input_ports);
        }
        if let Some(ctrl_event) = self.torquer_ctrl_port.subscribe() {
            self.torque = BodyVector::from(ctrl_event.torque);
        }
        {
            let f = |_, _, angular_velocity: BodyVector| {
                let input_torque = self.torque.clone();
                let gyroscopic_torque = Vector3::from(angular_velocity.clone()).cross(&(
                    self.inertia * Vector3::from(angular_velocity.clone())
                ));
                let angular_acceleration = self.inertia.try_inverse().unwrap() * 
                    (Vector3::from(input_torque) - gyroscopic_torque);
                BodyVector::from(angular_acceleration)
            };
            self.angular_velocity.propagate(f, dt, NaiveDateTime::default());
        }
        {
            let f = |phase, _, q| {
                let angular_velocity = self.angular_velocity.get(phase).unwrap();
                1.0 / 2.0
                    * Quaternion::new(
                        0.0,
                        angular_velocity.x,
                        angular_velocity.y,
                        angular_velocity.z,
                    )
                    * q
            };

            self.attitude.propagate(f, dt, NaiveDateTime::default());
        }
    }

    pub fn clear_state(&mut self) {
        self.angular_velocity.clear();
        self.attitude.clear();
        self.gyro_port.clear();
        self.sun_sensor_port.clear();
        self.star_tracker_port.clear();
    }
}

pub struct IdealAttitudeSimSensorOutput<'a> {
    pub gyro: &'a s5e_port::S5EPublishPort<s5e_port::GyroSensorData>,
    pub sun_sensor: &'a s5e_port::S5EPublishPort<s5e_port::LightDirectionData>,
    pub star_tracker: &'a s5e_port::S5EPublishPort<s5e_port::StarTrackerData>,
}

pub struct IdealAttitudeSimActuatorInput<'a> {
    pub torquer_ctrl: &'a mut s5e_port::S5ESubscribePort<s5e_port::IdealTorquerCtrlEvent>,
}

pub trait IdealAttitudeSimInterface {
    type InputPortSet<'a>: crate::SimInputTransfer<IdealAttitudeSimSensorOutput<'a>>
    where
        Self: 'a;
    type OutputPortSet<'a>: crate::SimOutputTransfer<IdealAttitudeSimActuatorInput<'a>>
    where
        Self: 'a;
    fn init(&mut self);
    fn main_loop(&mut self, dt: f64);
    fn input_ports(&mut self) -> Self::InputPortSet<'_>;
    fn output_ports(&mut self) -> Self::OutputPortSet<'_>;
}

// ============================================================================
// Virtual Magnetic Field Simulation
// ============================================================================

/// Virtual magnetic field model that provides time-varying magnetic field in ECI frame
/// without requiring orbital mechanics
pub struct VirtualMagneticFieldModel {
    /// Magnetic field magnitude in nT
    pub magnitude: f64,
    /// Angular velocity of magnetic field rotation in ECI frame (rad/s)
    pub rotation_rate: f64,
    /// Current phase angle (rad)
    phase: f64,
    /// Rotation axis in ECI frame
    pub rotation_axis: Vector3<f64>,
    /// Initial magnetic field direction in ECI frame
    initial_direction: Vector3<f64>,
}

impl VirtualMagneticFieldModel {
    /// Create a new virtual magnetic field model
    ///
    /// # Arguments
    /// * `magnitude` - Magnetic field magnitude in nT (typical LEO: 25000-65000 nT)
    /// * `rotation_rate` - Angular velocity of magnetic field rotation (rad/s)
    ///                     For LEO orbit, typical period ~90min → ~0.00116 rad/s
    /// * `rotation_axis` - Axis around which the magnetic field rotates (normalized)
    /// * `initial_direction` - Initial magnetic field direction (normalized)
    pub fn new(
        magnitude: f64,
        rotation_rate: f64,
        rotation_axis: Vector3<f64>,
        initial_direction: Vector3<f64>,
    ) -> Self {
        Self {
            magnitude,
            rotation_rate,
            rotation_axis: rotation_axis.normalize(),
            phase: 0.0,
            initial_direction: initial_direction.normalize(),
        }
    }

    /// Get current magnetic field vector in ECI frame
    pub fn magnetic_field_eci(&self) -> ECIVector {
        let rotation = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(self.rotation_axis),
            self.phase,
        );
        let direction = rotation * self.initial_direction;
        ECIVector {
            x: direction.x * self.magnitude,
            y: direction.y * self.magnitude,
            z: direction.z * self.magnitude,
        }
    }

    /// Update the magnetic field model
    pub fn tick(&mut self, dt: f64) {
        self.phase += self.rotation_rate * dt;
        // Keep phase in [0, 2π)
        self.phase = self.phase.rem_euclid(2.0 * std::f64::consts::PI);
    }
}

/// Simulation with virtual magnetic field for MTQ-based attitude control testing
pub struct VirtualMagFieldSim<Fsw> {
    pub inertia: Matrix3<f64>,

    pub fsw: Fsw,

    pub attitude: RK4Solver<Quaternion<f64>, NaiveDateTime, f64>,
    pub angular_velocity: RK4Solver<BodyVector, NaiveDateTime, f64>,
    pub magnetic_moment: BodyVector,

    // Sensor ports
    pub gyro_port: s5e_port::S5EPublishPort<s5e_port::GyroSensorData>,
    pub sun_sensor_port: s5e_port::S5EPublishPort<s5e_port::LightDirectionData>,
    pub magnetometer_port: s5e_port::S5EPublishPort<s5e_port::MagnetometerData>,

    // Actuator ports
    pub mtq_ctrl_port: s5e_port::S5ESubscribePort<s5e_port::MagnetorquerCtrlEvent>,

    // Environment
    pub magnetic_field_model: VirtualMagneticFieldModel,
    pub sun_direction_eci: ECIVector,

    // Max dipole moment for MTQ (Am²)
    pub mtq_max_dipole_moment: f64,
}

impl<Fsw> VirtualMagFieldSim<Fsw> {
    pub fn new(
        inertia: Matrix3<f64>,
        fsw: Fsw,
        initial_attitude: UnitQuaternion<f64>,
        initial_angular_velocity: BodyVector,
        sun_direction_eci: ECIVector,
        magnetic_field_model: VirtualMagneticFieldModel,
        mtq_max_dipole_moment: f64,
    ) -> Self {
        Self {
            inertia,
            fsw,
            attitude: RK4Solver::new(*initial_attitude.quaternion()),
            angular_velocity: RK4Solver::new(initial_angular_velocity),
            magnetic_moment: BodyVector { x: 0.0, y: 0.0, z: 0.0 },
            gyro_port: s5e_port::S5EPublishPort::new(),
            sun_sensor_port: s5e_port::S5EPublishPort::new(),
            magnetometer_port: s5e_port::S5EPublishPort::new(),
            mtq_ctrl_port: s5e_port::S5ESubscribePort::new(),
            magnetic_field_model,
            sun_direction_eci,
            mtq_max_dipole_moment,
        }
    }

    /// Get current magnetic field in ECI frame
    pub fn magnetic_field_eci(&self) -> ECIVector {
        self.magnetic_field_model.magnetic_field_eci()
    }

    /// Get current magnetic field in body frame
    pub fn magnetic_field_body(&self) -> BodyVector {
        let attitude_q = UnitQuaternion::from_quaternion(self.attitude.get_now());
        BodyVector::from_eci(self.magnetic_field_eci(), attitude_q)
    }

    pub fn tick(&mut self, dt: f64)
    where
        Fsw: VirtualMagFieldSimInterface,
    {
        // Transfer sensor data to FSW
        {
            let sensor_ports = VirtualMagFieldSimSensorOutput {
                gyro: &self.gyro_port,
                sun_sensor: &self.sun_sensor_port,
                magnetometer: &self.magnetometer_port,
            };
            let mut fsw_input_ports = self.fsw.input_ports();
            fsw_input_ports.transfer_from(&sensor_ports);
        }

        // Run FSW
        self.fsw.main_loop(dt);

        // Transfer actuator commands from FSW
        {
            let fsw_output_ports = self.fsw.output_ports();
            let mut actuator_input_ports = VirtualMagFieldSimActuatorInput {
                mtq_ctrl: &mut self.mtq_ctrl_port,
            };
            fsw_output_ports.transfer_to(&mut actuator_input_ports);
        }

        // Get MTQ command and apply saturation
        if let Some(ctrl_event) = self.mtq_ctrl_port.subscribe() {
            let moment = Vector3::from(ctrl_event.magnetic_moment);
            let moment_norm = moment.norm();
            let saturated_moment = if moment_norm > self.mtq_max_dipole_moment {
                moment * (self.mtq_max_dipole_moment / moment_norm)
            } else {
                moment
            };
            self.magnetic_moment = BodyVector::from(saturated_moment);
        }

        // Calculate torque from MTQ: τ = m × B (in body frame)
        let mag_field_body = self.magnetic_field_body();
        let moment_vec = Vector3::from(self.magnetic_moment.clone());
        let mag_field_vec = Vector3::new(mag_field_body.x, mag_field_body.y, mag_field_body.z);
        // Convert nT to T for torque calculation
        let torque_body = moment_vec.cross(&(mag_field_vec * 1e-9));

        // Propagate angular velocity
        {
            let torque = torque_body;
            let f = |_, _, angular_velocity: BodyVector| {
                let gyroscopic_torque = Vector3::from(angular_velocity.clone())
                    .cross(&(self.inertia * Vector3::from(angular_velocity.clone())));
                let angular_acceleration =
                    self.inertia.try_inverse().unwrap() * (torque - gyroscopic_torque);
                BodyVector::from(angular_acceleration)
            };
            self.angular_velocity.propagate(f, dt, NaiveDateTime::default());
        }

        // Propagate attitude
        {
            let f = |phase, _, q| {
                let angular_velocity = self.angular_velocity.get(phase).unwrap();
                0.5 * Quaternion::new(
                    0.0,
                    angular_velocity.x,
                    angular_velocity.y,
                    angular_velocity.z,
                ) * q
            };
            self.attitude.propagate(f, dt, NaiveDateTime::default());
        }

        // Update magnetic field model
        self.magnetic_field_model.tick(dt);
    }

    pub fn clear_state(&mut self) {
        self.angular_velocity.clear();
        self.attitude.clear();
        self.gyro_port.clear();
        self.sun_sensor_port.clear();
        self.magnetometer_port.clear();
        self.mtq_ctrl_port.reset();
    }
}

pub struct VirtualMagFieldSimSensorOutput<'a> {
    pub gyro: &'a s5e_port::S5EPublishPort<s5e_port::GyroSensorData>,
    pub sun_sensor: &'a s5e_port::S5EPublishPort<s5e_port::LightDirectionData>,
    pub magnetometer: &'a s5e_port::S5EPublishPort<s5e_port::MagnetometerData>,
}

pub struct VirtualMagFieldSimActuatorInput<'a> {
    pub mtq_ctrl: &'a mut s5e_port::S5ESubscribePort<s5e_port::MagnetorquerCtrlEvent>,
}

pub trait VirtualMagFieldSimInterface {
    type InputPortSet<'a>: crate::SimInputTransfer<VirtualMagFieldSimSensorOutput<'a>>
    where
        Self: 'a;
    type OutputPortSet<'a>: crate::SimOutputTransfer<VirtualMagFieldSimActuatorInput<'a>>
    where
        Self: 'a;
    fn init(&mut self);
    fn main_loop(&mut self, dt: f64);
    fn input_ports(&mut self) -> Self::InputPortSet<'_>;
    fn output_ports(&mut self) -> Self::OutputPortSet<'_>;
}