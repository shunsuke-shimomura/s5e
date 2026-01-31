use std::fmt::Debug;

pub use nalgebra::{UnitQuaternion, Vector3};

#[derive(Clone, Debug)]
pub struct S4EPublishPort<T: Clone + Debug> {
    data: Option<T>,
}

impl<T: Clone + Debug> S4EPublishPort<T> {
    pub fn new() -> Self {
        Self { data: None }
    }

    pub fn publish(&mut self, data: T) {
        self.data = Some(data);
    }

    pub fn clear(&mut self) {
        self.data = None;
    }
}

#[derive(Clone, Debug)]
pub struct S4ESubscribePort<T: Clone + Debug> {
    data: Option<T>,
}

impl<T: Clone + Debug> S4ESubscribePort<T> {
    pub fn new() -> Self {
        Self { data: None }
    }

    pub fn subscribe(&mut self) -> Option<T> {
        self.data.take()
    }

    pub fn reset(&mut self) {
        self.data = None;
    }
}

pub fn transfer<T: Clone + Debug>(from: &S4EPublishPort<T>, to: &mut S4ESubscribePort<T>) {
    to.data = from.data.clone();
}

#[derive(Clone, Debug)]
pub struct SensorSwitchEvent {
    pub power_on: bool,
}

#[derive(Clone, Debug)]
pub struct LightDirectionData {
    pub light_direction: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct IrradianceSensorData {
    pub irradiance: f64,
}

#[derive(Clone, Debug, Copy)]
pub struct LightDetectionSystemData {
    pub light_direction: Option<Vector3<f64>>,
    pub irradiance: f64,
}

#[derive(Clone, Debug)]
pub struct MagnetometerData {
    pub magnetic_field: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct GyroSensorData {
    pub angular_velocity: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct AccelerometerData {
    pub acceleration: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct TemperatureSensorData {
    pub temperature: f64,
}

#[derive(Clone, Debug)]
pub struct StarTrackerData {
    pub attitude_quaternion: UnitQuaternion<f64>,
}

#[derive(Clone, Debug)]
pub struct ReactionWheelRotationData {
    pub speed_rpm: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct ObcEvent;

#[derive(Clone, Debug)]
pub struct MagnetorquerCtrlEvent {
    pub magnetic_moment: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct IdealTorquerCtrlEvent {
    pub torque: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct ReactionWheelCtrlEvent {
    pub angular_acceleration: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct ECIComponentPositionData {
    pub component_position: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct ECIComponentVelocityData {
    pub component_velocity: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct ECIGnssData {
    pub component_position: Vector3<f64>,
    pub component_velocity: Vector3<f64>,
    pub time: TimeData,
}

#[derive(Clone, Debug)]
pub struct TimeData {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub nanosecond: u32,
}

#[derive(Clone, Debug)]
pub struct GSCommandData {
    pub command: Command,
}

#[derive(Clone, Debug)]
pub enum Command {
    ControllerCommand(ControllerCommand),
}

#[derive(Clone, Debug)]
pub enum ControllerCommand {
    RWControlTransition,
    ThreeAxisControlTransition(UnitQuaternion<f64>),
    SunPointingControlTransition,
}
