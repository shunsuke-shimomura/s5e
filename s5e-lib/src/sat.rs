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