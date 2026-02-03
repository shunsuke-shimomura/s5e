use nalgebra::{Matrix3, UnitQuaternion, UnitVector3};
use s5e_port::S5ESubscribePort;

use crate::{
    constants::{
        GNSS_POSITION_NOISE_STD, GNSS_VELOCITY_NOISE_STD, GYRO_NOISE_STD,
        MAGNETOMETER_DEMAGNETIZATION_TIME, MAGNETOMETER_NOISE_STD, MAGNETOMETER_OBSERVATION_CYCLE,
        MAGNETOMETER_OBSERVATION_TIME_LIMIT, RW_INERTIA, STAR_TRACKER_NOISE_STD,
        SUN_SENSOR_NOISE_STD,
    },
    data,
};

pub enum ExclusiveTimerMode {
    Demagnetization { remaining_time: f64 },
    Observation { finished: bool, remaining_time: f64 },
    Control { remaining_time: f64 },
}

impl ExclusiveTimerMode {
    fn new_observation(observation_time_limit: f64) -> Self {
        ExclusiveTimerMode::Observation {
            finished: false,
            remaining_time: observation_time_limit,
        }
    }

    fn new_demagnetization(demagnetization_time: f64) -> Self {
        ExclusiveTimerMode::Demagnetization {
            remaining_time: demagnetization_time,
        }
    }
    fn new_control(observation_cycle: f64, demagnetization_time: f64) -> Self {
        ExclusiveTimerMode::Control {
            remaining_time: observation_cycle - demagnetization_time,
        }
    }

    pub fn is_observation(&self) -> bool {
        matches!(self, ExclusiveTimerMode::Observation { .. })
    }

    pub fn observation_complete(&mut self) {
        if let ExclusiveTimerMode::Observation { finished, .. } = self {
            *finished = true;
        }
    }

    pub fn tick(&mut self, dt: f64) {
        match self {
            ExclusiveTimerMode::Demagnetization { remaining_time } => {
                *remaining_time -= dt;
            }
            ExclusiveTimerMode::Observation {
                finished: _,
                remaining_time,
            } => {
                *remaining_time -= dt;
            }
            ExclusiveTimerMode::Control { remaining_time } => {
                *remaining_time -= dt;
            }
        }
    }
}

pub struct ExclusiveTimer {
    pub mode: ExclusiveTimerMode,
    pub observation_cycle: f64,
    pub demagnetization_time: f64,
    pub observation_time_limit: f64,
}

impl Default for ExclusiveTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl ExclusiveTimer {
    pub fn new() -> Self {
        Self {
            mode: ExclusiveTimerMode::new_observation(MAGNETOMETER_OBSERVATION_TIME_LIMIT),
            observation_cycle: MAGNETOMETER_OBSERVATION_CYCLE,
            demagnetization_time: MAGNETOMETER_DEMAGNETIZATION_TIME,
            observation_time_limit: MAGNETOMETER_OBSERVATION_TIME_LIMIT,
        }
    }

    pub fn main_loop(&mut self, dt: f64) -> Option<data::MagneticFieldExclusiveCtrlEvent> {
        self.mode.tick(dt);
        match &self.mode {
            ExclusiveTimerMode::Demagnetization { remaining_time } => {
                if *remaining_time <= 0.0 {
                    self.mode = ExclusiveTimerMode::new_observation(self.observation_time_limit);
                }
                None
            }
            ExclusiveTimerMode::Observation {
                finished,
                remaining_time,
            } => {
                let finished = *finished || *remaining_time <= 0.0;
                if finished {
                    self.mode = ExclusiveTimerMode::new_control(
                        self.observation_cycle,
                        self.demagnetization_time,
                    );
                    Some(data::MagneticFieldExclusiveCtrlEvent { demag: false })
                } else {
                    None
                }
            }
            ExclusiveTimerMode::Control { remaining_time } => {
                if *remaining_time <= 0.0 {
                    self.mode = ExclusiveTimerMode::new_demagnetization(self.demagnetization_time);
                    Some(data::MagneticFieldExclusiveCtrlEvent { demag: true })
                } else {
                    None
                }
            }
        }
    }

    pub fn observable(&self) -> bool {
        self.mode.is_observation()
    }

    pub fn observation_complete(&mut self) {
        self.mode.observation_complete();
    }
}

pub struct MagnetometerDriverOutput {
    pub magnetic_field: Option<data::MagnetometerData>,
    pub exclusive_ctrl: Option<data::MagneticFieldExclusiveCtrlEvent>,
}

pub struct MagnetometerDriver {
    exclusive_timer: ExclusiveTimer,
    pub sim_port: S5ESubscribePort<s5e_port::MagnetometerData>,
    noise_std: f64,
    alignment: UnitQuaternion<f64>,
}

impl MagnetometerDriver {
    pub fn new(alignment: UnitQuaternion<f64>) -> Self {
        Self {
            exclusive_timer: ExclusiveTimer::new(),
            sim_port: S5ESubscribePort::new(),
            noise_std: MAGNETOMETER_NOISE_STD,
            alignment,
        }
    }
    pub fn main_loop(&mut self, dt: f64) -> MagnetometerDriverOutput {
        // Exclusive Control
        let exclusive_ctrl = self.exclusive_timer.main_loop(dt);

        // Transfer data
        let magnetic_field = self
            .exclusive_timer
            .observable()
            .then_some(())
            .and(self.sim_port.subscribe())
            .map(|data| {
                let magnetic_field = self
                    .alignment
                    .inverse_transform_vector(&data.magnetic_field);
                data::MagnetometerData {
                    magnetic_field,
                    std: self.noise_std,
                }
            });
        MagnetometerDriverOutput {
            magnetic_field,
            exclusive_ctrl,
        }
    }
}

pub struct GyroDriver {
    pub sim_port: S5ESubscribePort<s5e_port::GyroSensorData>,
    noise_std: f64,
    alignment: UnitQuaternion<f64>,
}

impl GyroDriver {
    pub fn new(alignment: UnitQuaternion<f64>) -> Self {
        Self {
            alignment,
            sim_port: S5ESubscribePort::new(),
            noise_std: GYRO_NOISE_STD,
        }
    }
    pub fn main_loop(&mut self) -> Option<data::GyroSensorData> {
        self.sim_port.subscribe().map(|data| data::GyroSensorData {
            angular_velocity: self
                .alignment
                .inverse_transform_vector(&data.angular_velocity),
            std: self.noise_std,
        })
    }
}

#[derive(Clone, Debug)]
pub struct EciGnssDriverOutput {
    pub posvel: Option<data::ECIGnssData>,
    pub time: Option<data::TimeSourceData>,
}

pub struct EciGnssDriver {
    pub sim_port: S5ESubscribePort<s5e_port::ECIGnssData>,
    position_noise_std: f64,
    velocity_noise_std: f64,
}

impl Default for EciGnssDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl EciGnssDriver {
    pub fn new() -> Self {
        Self {
            sim_port: S5ESubscribePort::new(),
            position_noise_std: GNSS_POSITION_NOISE_STD,
            velocity_noise_std: GNSS_VELOCITY_NOISE_STD,
        }
    }
    pub fn main_loop(&mut self) -> EciGnssDriverOutput {
        self.sim_port
            .subscribe()
            .map(|data| {
                (
                    data::ECIGnssData {
                        component_position: data.component_position,
                        position_std: self.position_noise_std,
                        component_velocity: data.component_velocity,
                        velocity_std: self.velocity_noise_std,
                    },
                    crate::AbsoluteTime {
                        year: data.time.year,
                        month: data.time.month,
                        day: data.time.day,
                        hour: data.time.hour,
                        minute: data.time.minute,
                        second: data.time.second,
                        nanosecond: data.time.nanosecond,
                    },
                )
            })
            .map(|(posvel, datetime)| EciGnssDriverOutput {
                posvel: Some(posvel),
                time: Some(data::TimeSourceData { datetime }),
            })
            .unwrap_or(EciGnssDriverOutput {
                posvel: None,
                time: None,
            })
    }
}

pub struct SunSensorDriver {
    pub sim_port: S5ESubscribePort<s5e_port::LightDetectionSystemData>,
    noise_std: f64,
    alignment: UnitQuaternion<f64>,
}

impl SunSensorDriver {
    pub fn new(alignment: UnitQuaternion<f64>) -> Self {
        Self {
            sim_port: S5ESubscribePort::new(),
            noise_std: SUN_SENSOR_NOISE_STD,
            alignment,
        }
    }

    pub fn main_loop(&mut self) -> Option<data::SunSensorData> {
        self.sim_port.subscribe().and_then(|data| {
            data.light_direction.map(|sun_d| {
                let sun_d = self.alignment.inverse_transform_vector(&sun_d);
                data::SunSensorData {
                    sun_direction: UnitVector3::new_normalize(sun_d),
                    // 本当は照度に応じたノイズを載せるべきだが、簡単のため一定値とする
                    std: self.noise_std,
                }
            })
        })
    }
}

pub struct StarTrackerDriver {
    pub sim_port: S5ESubscribePort<s5e_port::StarTrackerData>,
    noise_std: f64,
    alignment: UnitQuaternion<f64>,
}

impl StarTrackerDriver {
    pub fn new(alignment: UnitQuaternion<f64>) -> Self {
        Self {
            sim_port: S5ESubscribePort::new(),
            noise_std: STAR_TRACKER_NOISE_STD,
            alignment,
        }
    }

    pub fn main_loop(&mut self) -> Option<data::StarTrackerData> {
        self.sim_port.subscribe().map(|data| data::StarTrackerData {
            attitude: self.alignment.conjugate() * data.attitude_quaternion,
            std: self.noise_std,
        })
    }
}

pub struct ReactionWheelStatusSensorDriver {
    pub sim_port: S5ESubscribePort<s5e_port::ReactionWheelRotationData>,
    inertia: Matrix3<f64>,
}

impl Default for ReactionWheelStatusSensorDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl ReactionWheelStatusSensorDriver {
    pub fn new() -> Self {
        Self {
            sim_port: S5ESubscribePort::new(),
            inertia: RW_INERTIA,
        }
    }

    pub fn main_loop(&mut self) -> Option<data::ReactionWheelMomentumData> {
        self.sim_port.subscribe().map(|data| {
            let momentum = self.inertia * (data.speed_rpm * (2.0 * std::f64::consts::PI / 60.0));
            data::ReactionWheelMomentumData { momentum }
        })
    }
}
