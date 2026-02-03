use core::fmt::{self, Display, Formatter};
use nalgebra::{Matrix3, UnitQuaternion, UnitVector3, Vector3};

use crate::{
    constants::{
        RW_D_GAIN, RW_P_GAIN, UNLOADING_END_THRESHOLD, UNLOADING_GAIN, UNLOADING_START_THRESHOLD,
    },
    controller::{SunPointingController, ThreeAxisController},
};

pub enum RWControlMode {
    Initial,
    SunPointing(SunPointingController),
    ThreeAxis(ThreeAxisController),
}

impl Display for RWControlMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RWControlMode::Initial => write!(f, "Initial"),
            RWControlMode::SunPointing(_) => write!(f, "SunPointing"),
            RWControlMode::ThreeAxis(_) => write!(f, "ThreeAxis"),
        }
    }
}

pub enum MTQUnloadingMode {
    Idle,
    Unloading(UnloadingController),
}

impl Display for MTQUnloadingMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MTQUnloadingMode::Idle => write!(f, "Idle"),
            MTQUnloadingMode::Unloading(_) => write!(f, "Unloading"),
        }
    }
}

pub struct RWControl {
    pub mode: RWControlMode,
    pub unloading_mode: MTQUnloadingMode,
    rw_start_threshold: f64,
    rw_end_threshold: f64,
}

impl Default for RWControl {
    fn default() -> Self {
        Self::new()
    }
}

impl RWControl {
    pub fn new() -> Self {
        Self {
            mode: RWControlMode::Initial,
            unloading_mode: MTQUnloadingMode::Idle,
            rw_start_threshold: UNLOADING_START_THRESHOLD,
            rw_end_threshold: UNLOADING_END_THRESHOLD,
        }
    }

    pub fn magnetic_moment(
        &mut self,
        magnetic_field: &Option<Vector3<f64>>,
        angular_momentum_total: &Option<Vector3<f64>>,
    ) -> Option<Vector3<f64>> {
        match &self.unloading_mode {
            MTQUnloadingMode::Idle => None,
            MTQUnloadingMode::Unloading(controller) => {
                if let (Some(magnetic_field_body), Some(angular_momentum_total)) =
                    (magnetic_field, angular_momentum_total)
                {
                    Some(controller.magnetic_moment(magnetic_field_body, angular_momentum_total))
                } else {
                    None
                }
            }
        }
    }

    pub fn rw_torque(
        &mut self,
        sun_direction: &Option<UnitVector3<f64>>,
        angular_velocity: &Option<Vector3<f64>>,
        attitude_quaternion: &Option<UnitQuaternion<f64>>,
    ) -> Option<Vector3<f64>> {
        match &self.mode {
            RWControlMode::Initial => None,
            RWControlMode::SunPointing(controller) => {
                if let (Some(sun_direction_body), Some(angular_velocity_body)) =
                    (sun_direction, angular_velocity)
                {
                    Some(controller.torque(sun_direction_body, angular_velocity_body))
                } else {
                    None
                }
            }
            RWControlMode::ThreeAxis(controller) => {
                if let (Some(attitude_quaternion), Some(angular_velocity_body)) =
                    (attitude_quaternion, angular_velocity)
                {
                    Some(controller.torque(attitude_quaternion, angular_velocity_body))
                } else {
                    None
                }
            }
        }
    }

    pub fn auto_transition(&mut self, rw_angular_momentum: &Option<Vector3<f64>>) {
        if let RWControlMode::Initial = self.mode {
            self.mode =
                RWControlMode::SunPointing(SunPointingController::new(RW_P_GAIN, RW_D_GAIN));
        }
        match &self.unloading_mode {
            MTQUnloadingMode::Idle => {
                if let Some(rw_angular_momentum) = rw_angular_momentum
                    && rw_angular_momentum.norm() > self.rw_start_threshold
                {
                    self.unloading_mode = MTQUnloadingMode::Unloading(UnloadingController::new());
                }
            }
            MTQUnloadingMode::Unloading(_) => {
                if let Some(rw_angular_momentum) = rw_angular_momentum
                    && rw_angular_momentum.norm() < self.rw_end_threshold
                {
                    self.unloading_mode = MTQUnloadingMode::Idle;
                }
            }
        }
    }

    pub fn to_three_axis(
        &mut self,
        target_quaternion: UnitQuaternion<f64>,
    ) -> Result<(), &'static str> {
        match &self.mode {
            RWControlMode::SunPointing(_) => {
                self.mode = RWControlMode::ThreeAxis(ThreeAxisController::new(
                    RW_P_GAIN,
                    RW_D_GAIN,
                    target_quaternion,
                ));
                Ok(())
            }
            _ => Err(
                "RWControl: can only switch to ThreeAxis from SunPointing mode by manual transition",
            ),
        }
    }

    pub fn to_sun_pointing(&mut self) -> Result<(), &'static str> {
        match &self.mode {
            RWControlMode::ThreeAxis(_) => {
                self.mode =
                    RWControlMode::SunPointing(SunPointingController::new(RW_P_GAIN, RW_D_GAIN));
                Ok(())
            }
            _ => Err(
                "RWControl: can only switch to SunPointing from ThreeAxis mode by manual transition",
            ),
        }
    }
}

pub struct UnloadingController {
    unloading_gain: Matrix3<f64>,
}

impl Default for UnloadingController {
    fn default() -> Self {
        Self::new()
    }
}

impl UnloadingController {
    pub fn new() -> Self {
        Self {
            unloading_gain: UNLOADING_GAIN,
        }
    }

    pub fn magnetic_moment(
        &self,
        magnetic_field_body: &Vector3<f64>,
        angular_momentum_total: &Vector3<f64>,
    ) -> Vector3<f64> {
        let moment = self.unloading_gain * magnetic_field_body.cross(angular_momentum_total)
            / magnetic_field_body.norm_squared();
        -moment
    }
}
