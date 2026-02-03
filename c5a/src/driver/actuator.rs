use nalgebra::{Matrix3, Vector3};
use s5e_port::S5EPublishPort;

use crate::{
    constants::{RW_INERTIA, RW_MAX_ANGULAR_ACCELERATION},
    data,
};

pub struct MagnetorquerDriverInput {
    pub moment: Option<data::MagnetorquerCtrlEvent>,
    pub exclusive_ctrl: Option<data::MagneticFieldExclusiveCtrlEvent>,
}

pub struct MagnetorquerDriver {
    pub max_dipole_moment: f64,
    pub sim_port: S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,
    pub is_demagnetizing: bool,
}

impl Default for MagnetorquerDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl MagnetorquerDriver {
    pub fn new() -> Self {
        Self {
            max_dipole_moment: crate::constants::MTQ_MAX_DIPOLE_MOMENT,
            sim_port: S5EPublishPort::new(),
            is_demagnetizing: false,
        }
    }

    pub fn main_loop(&mut self, input: MagnetorquerDriverInput) {
        self.is_demagnetizing = input
            .exclusive_ctrl
            .and_then(|ctrl| ctrl.demag.then_some(()))
            .is_some();
        if self.is_demagnetizing {
            self.sim_port.publish(s5e_port::MagnetorquerCtrlEvent {
                magnetic_moment: Vector3::new(0.0, 0.0, 0.0),
            });
        } else if let Some(target_moment) = input.moment {
            let clipped_moment = Vector3::new(
                target_moment
                    .magnetic_moment
                    .x
                    .clamp(-self.max_dipole_moment, self.max_dipole_moment),
                target_moment
                    .magnetic_moment
                    .y
                    .clamp(-self.max_dipole_moment, self.max_dipole_moment),
                target_moment
                    .magnetic_moment
                    .z
                    .clamp(-self.max_dipole_moment, self.max_dipole_moment),
            );
            self.sim_port.publish(s5e_port::MagnetorquerCtrlEvent {
                magnetic_moment: clipped_moment,
            });
        }
    }

    pub fn clear(&mut self) {
        self.sim_port.clear();
    }
}

pub struct ReactionWheelDriver {
    pub max_angular_acceleration: f64,
    pub inertia: Matrix3<f64>,
    pub sim_port: S5EPublishPort<s5e_port::ReactionWheelCtrlEvent>,
}

impl Default for ReactionWheelDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl ReactionWheelDriver {
    pub fn new() -> Self {
        Self {
            max_angular_acceleration: RW_MAX_ANGULAR_ACCELERATION,
            inertia: RW_INERTIA,
            sim_port: S5EPublishPort::new(),
        }
    }

    pub fn main_loop(&mut self, input: Option<data::ReactionWheelCtrlEvent>) {
        if let Some(event) = input {
            let acc = -self.inertia.try_inverse().unwrap() * event.torque;
            let clipped_acceleration = Vector3::new(
                acc.x.clamp(
                    -self.max_angular_acceleration,
                    self.max_angular_acceleration,
                ),
                acc.y.clamp(
                    -self.max_angular_acceleration,
                    self.max_angular_acceleration,
                ),
                acc.z.clamp(
                    -self.max_angular_acceleration,
                    self.max_angular_acceleration,
                ),
            );
            self.sim_port.publish(s5e_port::ReactionWheelCtrlEvent {
                angular_acceleration: self.inertia.try_inverse().unwrap() * clipped_acceleration,
            });
        }
    }

    pub fn clear(&mut self) {
        self.sim_port.clear();
    }
}
