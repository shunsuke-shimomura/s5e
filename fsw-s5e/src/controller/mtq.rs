use core::fmt::{self, Display, Formatter};

use nalgebra::{Matrix3, UnitQuaternion, UnitVector3, Vector3};

use crate::{
    constants::{
        BDOT_DETUMBLING_GAIN, BDOT_TRANSITION_TIMER_THRESHOLD, FROM_BDOT_ANGULAR_VELOCITY_THRESHOLD, MTQ_D_GAIN, MTQ_P_GAIN, SATELLITE_INERTIA, SUN_POINTING_TARGET_AXIS, TO_BDOT_ANGULAR_VELOCITY_THRESHOLD
    },
    controller::{SunPointingController, ThreeAxisController, torque_to_magnetic_moment},
};

pub enum MTQControlMode {
    Initial,
    BdotDetumbling(BdotDetumblingController),
    SunPointing(SunPointingController),
    ThreeAxis(ThreeAxisController),
}

impl Display for MTQControlMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MTQControlMode::Initial => write!(f, "Initial"),
            MTQControlMode::BdotDetumbling(_) => write!(f, "BdotDetumbling"),
            MTQControlMode::SunPointing(_) => write!(f, "SunPointing"),
            MTQControlMode::ThreeAxis(_) => write!(f, "ThreeAxis"),
        }
    }
}

pub struct MTQControl {
    pub mode: MTQControlMode,
    from_bdot_angular_velocity_threshold: f64,
    transition_timer: f64,
    timer_threshold: f64,
    to_bdot_angular_velocity_threshold: f64,
}

impl Default for MTQControl {
    fn default() -> Self {
        Self::new()
    }
}

impl MTQControl {
    pub fn new() -> Self {
        Self {
            mode: MTQControlMode::Initial,
            from_bdot_angular_velocity_threshold: FROM_BDOT_ANGULAR_VELOCITY_THRESHOLD,
            to_bdot_angular_velocity_threshold: TO_BDOT_ANGULAR_VELOCITY_THRESHOLD,
            transition_timer: 0.0,
            timer_threshold: BDOT_TRANSITION_TIMER_THRESHOLD,
        }
    }

    pub fn magnet_moment(
        &mut self,
        magnetic_field: &Option<Vector3<f64>>,
        sun_direction: &Option<UnitVector3<f64>>,
        angular_velocity: &Option<Vector3<f64>>,
        attitude_quaternion: &Option<nalgebra::UnitQuaternion<f64>>,
    ) -> Option<Vector3<f64>> {
        match &mut self.mode {
            MTQControlMode::Initial => None,
            MTQControlMode::BdotDetumbling(controller) => {
                if let (Some(magnetic_field_body), Some(angular_velocity)) =
                    (magnetic_field, angular_velocity)
                {
                    Some(controller.magnetic_moment(magnetic_field_body, angular_velocity))
                } else {
                    None
                }
            }
            MTQControlMode::SunPointing(controller) => {
                if let (Some(sun_dir), Some(ang_vel), Some(mag_field)) =
                    (sun_direction, angular_velocity, magnetic_field)
                {
                    let error = SUN_POINTING_TARGET_AXIS.cross(&sun_dir);
                    let pointing_moment = if error.norm() > 0.15 {
                        let torque = MTQ_P_GAIN * SATELLITE_INERTIA.try_inverse().unwrap() * error;
                        torque_to_magnetic_moment(&torque, mag_field)
                    } else {
                        Vector3::new(0.0, 0.0, 0.0)
                    };
                    let angular_momentum_moment = MTQ_D_GAIN * mag_field.cross(ang_vel);

                    Some(-pointing_moment - angular_momentum_moment)
                } else {
                    None
                }
            }
            MTQControlMode::ThreeAxis(controller) => {
                if let (Some(ang_vel), Some(target_q), Some(mag_field)) =
                    (angular_velocity, attitude_quaternion, magnetic_field)
                {
                    let torque = controller.torque(target_q, ang_vel);
                    Some(torque_to_magnetic_moment(&torque, mag_field))
                } else {
                    None
                }
            }
        }
    }

    pub fn auto_transition(
        &mut self,
        magnetic_field: &Option<Vector3<f64>>,
        angular_velocity: &Option<Vector3<f64>>,
        dt: f64,
    ) {
        match &self.mode {
            MTQControlMode::Initial => {
                if magnetic_field.is_some() && angular_velocity.is_some() {
                    self.mode = MTQControlMode::BdotDetumbling(BdotDetumblingController::new());
                }
            }
            MTQControlMode::BdotDetumbling(_) => {
                if let Some(ang_vel) = angular_velocity {
                    if ang_vel.norm() < self.from_bdot_angular_velocity_threshold {
                        self.transition_timer += dt;
                    } else {
                        self.transition_timer = 0.0;
                    }
                    if self.transition_timer > self.timer_threshold {
                        self.mode = MTQControlMode::SunPointing(SunPointingController::new(
                            MTQ_P_GAIN, MTQ_D_GAIN,
                        ));
                        self.transition_timer = 0.0;
                    }
                }
            }
            _ => {
                if let Some(ang_vel) = angular_velocity
                    && ang_vel.norm() > self.to_bdot_angular_velocity_threshold
                {
                    self.mode = MTQControlMode::BdotDetumbling(BdotDetumblingController::new());
                }
            }
        }
    }

    pub fn to_three_axis(
        &mut self,
        target_quaternion: UnitQuaternion<f64>,
    ) -> Result<(), &'static str> {
        match &self.mode {
            MTQControlMode::SunPointing(_) => {
                self.mode = MTQControlMode::ThreeAxis(ThreeAxisController::new(
                    MTQ_P_GAIN,
                    MTQ_D_GAIN,
                    target_quaternion,
                ));
                Ok(())
            }
            _ => Err(
                "MTQControl: can only switch to ThreeAxis from SunPointing mode by manual transition",
            ),
        }
    }

    pub fn to_sun_pointing(&mut self) -> Result<(), &'static str> {
        match &self.mode {
            MTQControlMode::ThreeAxis(_) => {
                self.mode =
                    MTQControlMode::SunPointing(SunPointingController::new(MTQ_P_GAIN, MTQ_D_GAIN));
                Ok(())
            }
            _ => Err(
                "MTQControl: can only switch to SunPointing from ThreeAxis mode by manual transition",
            ),
        }
    }

    pub fn rw_ready(&self) -> bool {
        matches!(
            &self.mode,
            MTQControlMode::ThreeAxis(_) | MTQControlMode::SunPointing(_)
        )
    }
}

pub struct BdotDetumblingController {
    bdot_gain: Matrix3<f64>,
}

impl Default for BdotDetumblingController {
    fn default() -> Self {
        Self::new()
    }
}

impl BdotDetumblingController {
    pub fn new() -> Self {
        Self {
            bdot_gain: BDOT_DETUMBLING_GAIN,
        }
    }

    pub fn magnetic_moment(
        &mut self,
        magnetic_field_body: &Vector3<f64>,
        angular_velocity: &Vector3<f64>,
    ) -> Vector3<f64> {
        let magnetic_field = magnetic_field_body;
        let moment = self.bdot_gain * magnetic_field.cross(angular_velocity);
        -moment
    }
}
