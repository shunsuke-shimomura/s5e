use core::fmt;
use core::fmt::{Display, Formatter};

use nalgebra::{Matrix3, UnitQuaternion, UnitVector3, Vector3};

use crate::constants::SATELLITE_INERTIA;
use crate::{
    constants::SUN_POINTING_TARGET_AXIS,
    controller::{mtq::MTQControl, rw::RWControl},
    data,
};

pub mod mtq;
pub mod rw;

pub enum AttitudeControllMode {
    MTQ(MTQControl),
    RW(RWControl),
}

impl Display for AttitudeControllMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AttitudeControllMode::MTQ(_) => write!(f, "MTQ Control Mode"),
            AttitudeControllMode::RW(_) => write!(f, "RW Control Mode"),
        }
    }
}

impl AttitudeControllMode {
    fn transition_to_rw(&mut self) -> Result<(), &'static str> {
        match self {
            AttitudeControllMode::MTQ(mtq_control) => {
                if mtq_control.rw_ready() {
                    let rw_control = RWControl::new();
                    *self = AttitudeControllMode::RW(rw_control);
                    Ok(())
                } else {
                    Err("RW not ready for transition")
                }
            }
            AttitudeControllMode::RW(_) => Ok(()),
        }
    }
    fn transition_to_three_axis(
        &mut self,
        target_quaternion: UnitQuaternion<f64>,
    ) -> Result<(), &'static str> {
        match self {
            AttitudeControllMode::MTQ(mtq_control) => mtq_control.to_three_axis(target_quaternion),
            AttitudeControllMode::RW(rw_control) => rw_control.to_three_axis(target_quaternion),
        }
    }
    fn transition_to_sun_pointing(&mut self) -> Result<(), &'static str> {
        match self {
            AttitudeControllMode::MTQ(mtq_control) => mtq_control.to_sun_pointing(),
            AttitudeControllMode::RW(rw_control) => rw_control.to_sun_pointing(),
        }
    }
}

pub struct ControllerInput {
    pub angular_velocity: Option<data::AngularVelocityData>,
    pub attitude: Option<data::AttitudeDeterminationData>,
    pub sun_direction: Option<data::SunDirectionEstimationData>,
    pub magnetic_field: Option<data::MagneticFieldEstimationData>,
    pub rw_momentum: Option<data::ReactionWheelMomentumData>,
    pub cmd: Option<data::ControllerCommand>,
}

pub struct ControllerOutput {
    pub mtq_ctrl: Option<data::MagnetorquerCtrlEvent>,
    pub rw_ctrl: Option<data::ReactionWheelCtrlEvent>,
}

pub struct AttitudeController {
    pub mode: AttitudeControllMode,
}

impl Default for AttitudeController {
    fn default() -> Self {
        Self::new()
    }
}

impl AttitudeController {
    pub fn new() -> Self {
        Self {
            mode: AttitudeControllMode::MTQ(MTQControl::new()),
        }
    }

    pub fn main_loop(&mut self, input: &ControllerInput, dt: f64) -> ControllerOutput {
        let sun_direction_data = input.sun_direction.as_ref();
        let magnetic_field_data = input.magnetic_field.as_ref();
        let attitude_data = input.attitude.as_ref();
        let angular_velocity_data = input.angular_velocity.as_ref();
        let rw_momentum_data = input.rw_momentum.as_ref();
        let cmd_data = input.cmd.as_ref();

        let magnetic_field = magnetic_field_data
            .map(|data| data.magnetic_field_norm * data.magnetic_field_direction.into_inner());
        let sun_direction = sun_direction_data.map(|data| data.sun_direction);
        let angular_velocity = angular_velocity_data.map(|data| data.angular_velocity);
        let attitude_quaternion = attitude_data.map(|data| data.attitude);
        let rw_angular_momentum = rw_momentum_data.map(|data| data.momentum);

        if let Some(cmd) = cmd_data {
            match cmd {
                data::ControllerCommand::RWControlTransition => {
                    self.mode.transition_to_rw().unwrap()
                }
                data::ControllerCommand::ThreeAxisControlTransition(target_quaternion) => self
                    .mode
                    .transition_to_three_axis(*target_quaternion)
                    .unwrap(),
                data::ControllerCommand::SunPointingControlTransition => {
                    self.mode.transition_to_sun_pointing().unwrap()
                }
            }
        }

        let (mag_moment_opt, rw_torque_opt) = match &mut self.mode {
            AttitudeControllMode::MTQ(mtq_control) => {
                mtq_control.auto_transition(&magnetic_field, &angular_velocity, dt);
                let mag_moment_opt = mtq_control.magnet_moment(
                    &magnetic_field,
                    &sun_direction,
                    &angular_velocity,
                    &attitude_quaternion,
                );
                (mag_moment_opt, None)
            }
            AttitudeControllMode::RW(rw_control) => {
                rw_control.auto_transition(&rw_angular_momentum);
                let mag_moment_opt =
                    rw_control.magnetic_moment(&magnetic_field, &rw_angular_momentum);
                let rw_torque_opt =
                    rw_control.rw_torque(&sun_direction, &angular_velocity, &attitude_quaternion);
                (mag_moment_opt, rw_torque_opt)
            }
        };

        ControllerOutput {
            mtq_ctrl: mag_moment_opt.map(|m| data::MagnetorquerCtrlEvent { magnetic_moment: m }),
            rw_ctrl: rw_torque_opt.map(|t| data::ReactionWheelCtrlEvent { torque: t }),
        }
    }
}

pub struct SunPointingController {
    pointing_gain: Matrix3<f64>,
    angular_momentum_gain: Matrix3<f64>,
    target_axis: UnitVector3<f64>,
}

impl SunPointingController {
    pub fn new(pointing_gain: Matrix3<f64>, angular_momentum_gain: Matrix3<f64>) -> Self {
        Self {
            pointing_gain,
            angular_momentum_gain,
            target_axis: UnitVector3::new_normalize(SUN_POINTING_TARGET_AXIS),
        }
    }

    pub fn torque(
        &self,
        sun_direction_body: &UnitVector3<f64>,
        angular_velocity_body: &Vector3<f64>,
    ) -> Vector3<f64> {
        let pointing_torque = self.pointing_gain * SATELLITE_INERTIA.try_inverse().unwrap() * sun_direction_body.cross(&self.target_axis);
        let angular_momentum_torque = self.angular_momentum_gain * angular_velocity_body;

        pointing_torque - angular_momentum_torque
    }
}

pub struct ThreeAxisController {
    attitude_gain: Matrix3<f64>,
    angular_velocity_gain: Matrix3<f64>,
    target_quaternion: UnitQuaternion<f64>,
}

impl ThreeAxisController {
    pub fn new(
        attitude_gain: Matrix3<f64>,
        angular_velocity_gain: Matrix3<f64>,
        target_quaternion: UnitQuaternion<f64>,
    ) -> Self {
        Self {
            attitude_gain,
            angular_velocity_gain,
            target_quaternion,
        }
    }

    pub fn torque(
        &self,
        attitude_quaternion: &UnitQuaternion<f64>,
        angular_velocity_body: &Vector3<f64>,
    ) -> Vector3<f64> {
        let attitude_error_quaternion = self.target_quaternion * attitude_quaternion.inverse();
        let attitude_torque = self.attitude_gain * attitude_error_quaternion.vector();
        let angular_velocity_torque = self.angular_velocity_gain * angular_velocity_body;

        -attitude_torque - angular_velocity_torque
    }
}

fn torque_to_magnetic_moment(
    torque: &Vector3<f64>,
    magnetic_field_body_nt: &Vector3<f64>,
) -> Vector3<f64> {
    let b_norm_sq = magnetic_field_body_nt.norm_squared();
    if b_norm_sq < 1e-10 {
        return Vector3::new(0.0, 0.0, 0.0);
    }
    // magnetic_field_body is in nT; the physical simulator uses B in Tesla (1e-9 factor).
    // m = B_T × τ / |B_T|² = (B_nT × τ / |B_nT|²) * 1e9
    magnetic_field_body_nt.cross(torque) / b_norm_sq * 1e9
}
