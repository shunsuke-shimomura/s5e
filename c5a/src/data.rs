use std::fmt::Debug;

use nalgebra::{Matrix1, Matrix2, Matrix3, UnitVector3};
pub use nalgebra::{UnitQuaternion, Vector3};

#[derive(Clone, Debug)]
pub struct MagnetorquerCtrlEvent {
    pub magnetic_moment: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct MagneticFieldExclusiveCtrlEvent {
    pub demag: bool,
}

#[derive(Clone, Debug)]
pub struct ReactionWheelCtrlEvent {
    pub torque: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct ReactionWheelMomentumData {
    pub momentum: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct MagnetometerData {
    pub magnetic_field: Vector3<f64>,
    pub std: f64,
}

#[derive(Clone, Debug)]
pub struct GyroSensorData {
    pub angular_velocity: Vector3<f64>,
    pub std: f64,
}

#[derive(Clone, Debug)]
pub struct SunSensorData {
    pub sun_direction: UnitVector3<f64>,
    pub std: f64,
}

#[derive(Clone, Debug)]
pub struct StarTrackerData {
    pub attitude: UnitQuaternion<f64>,
    pub std: f64,
}

#[derive(Clone, Debug)]
pub struct ECIGnssData {
    pub component_position: Vector3<f64>,
    pub position_std: f64,
    pub component_velocity: Vector3<f64>,
    pub velocity_std: f64,
}

#[derive(Clone, Debug)]
pub struct TimeSourceData {
    pub datetime: crate::AbsoluteTime,
}

#[derive(Clone, Debug)]
pub struct ECIObservationData {
    pub position: Vector3<f64>,
    pub position_std: f64,
    pub velocity: Vector3<f64>,
    pub velocity_std: f64,
}

#[derive(Clone, Debug)]
pub struct InertialMagneticFieldData {
    pub magnetic_field_eci: Vector3<f64>,
    pub magnetic_field_eci_variance: Matrix3<f64>,
}

#[derive(Clone, Debug)]
pub struct InertialSunDirectionData {
    pub sun_direction_eci: UnitVector3<f64>,
    pub std: f64,
}

#[derive(Clone, Debug)]
pub struct SunDirectionEstimationData {
    pub sun_direction: UnitVector3<f64>,
    pub sun_direction_variance: Matrix2<f64>,
}

#[derive(Clone, Debug)]
pub struct MagneticFieldEstimationData {
    pub magnetic_field_direction: UnitVector3<f64>,
    pub magnetic_field_direction_variance: Matrix2<f64>,
    pub magnetic_field_norm: f64,
    pub magnetic_field_norm_variance: Matrix1<f64>,
}

#[derive(Clone, Debug)]
pub struct AttitudeDeterminationData {
    pub attitude: UnitQuaternion<f64>,
    pub attitude_variance: Matrix3<f64>,
}

#[derive(Clone, Debug)]
pub struct OrbitEstimationData {
    pub position: Vector3<f64>,
    pub position_variance: Matrix3<f64>,
    pub velocity: Vector3<f64>,
    pub velocity_variance: Matrix3<f64>,
}

#[derive(Clone, Debug)]
pub struct AngularVelocityData {
    pub angular_velocity: Vector3<f64>,
    pub angular_velocity_variance: Matrix3<f64>,
}

#[derive(Clone, Debug)]
pub struct GyroBiasData {
    pub gyro_bias: Vector3<f64>,
    pub gyro_bias_variance: Matrix3<f64>,
}

#[derive(Clone, Debug)]
pub enum ControllerCommand {
    RWControlTransition,
    ThreeAxisControlTransition(UnitQuaternion<f64>),
    SunPointingControlTransition,
}
