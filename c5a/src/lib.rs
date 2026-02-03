use core::f64;
use std::ops::{Add, AddAssign, Sub};
use std::time::{Duration, Instant};

use nalgebra::{UnitQuaternion, Vector3};
use s5e_port::S5ESubscribePort;

use crate::{
    controller::{AttitudeControllMode, ControllerInput},
    driver::actuator::MagnetorquerDriverInput,
    estimation::{
        attitude_determination::AttitudeDeterminationInput,
        direction_estimation::DirectionEstimatorInput,
    },
};

pub mod constants;
pub mod controller;
pub mod data;
pub mod driver;
pub mod environment;
pub mod estimation;
pub mod selector;
pub mod transformer;

#[derive(Default)]
pub struct FswProfile {
    pub sensor_drivers: Duration,
    pub inertial_sun: Duration,
    pub orbit_det: Duration,
    pub inertial_mag: Duration,
    pub direction_estimator: Duration,
    pub attitude_determination: Duration,
    pub controller: Duration,
    pub logging: Duration,
    pub call_count: u64,
}

impl FswProfile {
    pub fn print_summary(&self) {
        let drivers_total =
            self.sensor_drivers + self.inertial_sun + self.orbit_det + self.inertial_mag;
        let total = drivers_total
            + self.direction_estimator
            + self.attitude_determination
            + self.controller
            + self.logging;
        let pct = |d: Duration| {
            if total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / total.as_secs_f64() * 100.0
            }
        };
        println!(
            "=== Fsw::main_loop() Profile ({} calls, total {:.3}s) ===",
            self.call_count,
            total.as_secs_f64()
        );
        println!(
            "  drivers (total):          {:>8.3}ms ({:>5.1}%)",
            drivers_total.as_secs_f64() * 1000.0,
            pct(drivers_total)
        );
        println!(
            "    sensor_drivers:         {:>8.3}ms ({:>5.1}%)",
            self.sensor_drivers.as_secs_f64() * 1000.0,
            pct(self.sensor_drivers)
        );
        println!(
            "    inertial_sun:           {:>8.3}ms ({:>5.1}%)",
            self.inertial_sun.as_secs_f64() * 1000.0,
            pct(self.inertial_sun)
        );
        println!(
            "    orbit_det:              {:>8.3}ms ({:>5.1}%)",
            self.orbit_det.as_secs_f64() * 1000.0,
            pct(self.orbit_det)
        );
        println!(
            "    inertial_mag:           {:>8.3}ms ({:>5.1}%)",
            self.inertial_mag.as_secs_f64() * 1000.0,
            pct(self.inertial_mag)
        );
        println!(
            "  direction_estimator:      {:>8.3}ms ({:>5.1}%)",
            self.direction_estimator.as_secs_f64() * 1000.0,
            pct(self.direction_estimator)
        );
        println!(
            "  attitude_determination:   {:>8.3}ms ({:>5.1}%)",
            self.attitude_determination.as_secs_f64() * 1000.0,
            pct(self.attitude_determination)
        );
        println!(
            "  controller:               {:>8.3}ms ({:>5.1}%)",
            self.controller.as_secs_f64() * 1000.0,
            pct(self.controller)
        );
        println!(
            "  logging:                  {:>8.3}ms ({:>5.1}%)",
            self.logging.as_secs_f64() * 1000.0,
            pct(self.logging)
        );
    }
}

#[derive(Clone, Debug, std::marker::Copy)]
pub struct Time {
    absolute: Option<AbsoluteTime>,
    relative: f64,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl Time {
    pub fn new() -> Self {
        Time {
            absolute: None,
            relative: 0.0,
        }
    }

    #[cfg(test)]
    pub fn from_seconds(seconds: f64) -> Self {
        Time {
            absolute: None,
            relative: seconds,
        }
    }
}

impl Sub for Time {
    type Output = f64;

    fn sub(self, other: Time) -> f64 {
        self.relative - other.relative
    }
}

impl Add<f64> for Time {
    type Output = Time;

    fn add(self, other: f64) -> Time {
        Time {
            absolute: self.absolute.map(|abs| abs + other),
            relative: self.relative + other,
        }
    }
}

impl AddAssign<f64> for Time {
    fn add_assign(&mut self, other: f64) {
        self.relative += other;
        if let Some(mut absolute) = self.absolute {
            absolute += other;
            self.absolute = Some(absolute);
        }
    }
}

#[derive(Clone, Debug, std::marker::Copy)]
pub struct AbsoluteTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub nanosecond: u32,
}

impl Add<f64> for &AbsoluteTime {
    type Output = AbsoluteTime;
    fn add(self, other: f64) -> AbsoluteTime {
        let mut result = *self;
        let total_nanoseconds = (other * 1e9) as u64;
        let seconds = (total_nanoseconds / 1_000_000_000) as u32;
        let nanoseconds = (total_nanoseconds % 1_000_000_000) as u32;

        result.nanosecond += nanoseconds;
        if result.nanosecond >= 1_000_000_000 {
            result.second += 1;
            result.nanosecond -= 1_000_000_000;
        }

        result.second += seconds;
        if result.second >= 60 {
            result.minute += result.second / 60;
            result.second %= 60;
        }

        if result.minute >= 60 {
            result.hour += result.minute / 60;
            result.minute %= 60;
        }

        if result.hour >= 24 {
            result.day += result.hour / 24;
            result.hour %= 24;
        }
        result
    }
}

impl Add<f64> for AbsoluteTime {
    type Output = AbsoluteTime;
    fn add(self, other: f64) -> AbsoluteTime {
        (&self).add(other)
    }
}

impl AddAssign<f64> for AbsoluteTime {
    fn add_assign(&mut self, other: f64) {
        let result = *self + other;
        *self = result;
    }
}

pub struct Fsw {
    // ロガー
    controller_logger: Option<csv::Writer<std::fs::File>>,
    attitude_logger: Option<csv::Writer<std::fs::File>>,
    orbit_logger: Option<csv::Writer<std::fs::File>>,
    direction_logger: Option<csv::Writer<std::fs::File>>,

    // 時刻
    pub time: crate::Time,
    timer: estimation::Timer,

    // コマンド受信機
    pub cmd_receiver: driver::CommandReceiver,

    // ドライバ
    pub gyro_driver: driver::sensor::GyroDriver,
    pub magnetometer_driver: driver::sensor::MagnetometerDriver,
    pub magnetorquer_driver: driver::actuator::MagnetorquerDriver,
    pub eci_gnss_driver: driver::sensor::EciGnssDriver,
    pub ss_pz_driver: driver::sensor::SunSensorDriver,
    pub ss_py_driver: driver::sensor::SunSensorDriver,
    pub ss_mz_driver: driver::sensor::SunSensorDriver,
    pub ss_my_driver: driver::sensor::SunSensorDriver,
    pub star_tracker_driver: driver::sensor::StarTrackerDriver,
    pub rw_driver: driver::actuator::ReactionWheelDriver,
    pub rw_status_driver: driver::sensor::ReactionWheelStatusSensorDriver,

    // セレクタ
    pub sun_direction_selector: selector::sun_direction::SunDirectionSelector<4>,
    pub angular_velocity_selector: selector::angular_velocity::AngularVelocitySelector<2>,

    // 変換
    pub eci_gnss_pass_through: transformer::gnss_eci::EciGnssPassThroughConversion,

    // 慣性系参照
    pub inertial_mag_field_calc: environment::magnetic_field::InertialMagneticFieldCalculation,
    pub inertial_sun_direction_calc: environment::sun_direction::InertialSunDirectionCalculation,

    // 状態推定
    pub attitude_determination: estimation::AttitudeDetermination,
    pub orbit_determination: estimation::OrbitDetermination,
    pub direction_estimator: estimation::DirectionEstimator,

    // 制御器
    pub attitude_controller: controller::AttitudeController,

    // 状態量（シミュレーション側から参照しやすくする目的）
    pub attitude: Option<UnitQuaternion<f64>>,
    pub angular_velocity: Option<Vector3<f64>>,
    pub gyro_bias: Option<Vector3<f64>>,
    pub magnetic_dir: Option<nalgebra::UnitVector3<f64>>,
    pub sun_direction: Option<nalgebra::UnitVector3<f64>>,
    debug_print_flag: bool,
    pub profile: FswProfile,
}

fn format_option<T: std::fmt::Display>(opt: Option<T>) -> String {
    match opt {
        Some(val) => format!("{}", val),
        None => String::new(),
    }
}

impl Fsw {
    pub fn new(debug_print_flag: bool) -> Self {
        let time = crate::Time::new();
        let alignment_pz =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f64::consts::FRAC_PI_2);
        let alignment_py =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), f64::consts::FRAC_PI_2);
        let alignment_mz = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), f64::consts::PI);
        let alignment_my =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -f64::consts::FRAC_PI_2);
        Self {
            controller_logger: debug_print_flag
                .then_some(csv::Writer::from_path("out/fsw/controller.csv").ok())
                .flatten(),
            attitude_logger: debug_print_flag
                .then_some(csv::Writer::from_path("out/fsw/attitude.csv").ok())
                .flatten(),
            orbit_logger: debug_print_flag
                .then_some(csv::Writer::from_path("out/fsw/orbit.csv").ok())
                .flatten(),
            direction_logger: debug_print_flag
                .then_some(csv::Writer::from_path("out/fsw/direction.csv").ok())
                .flatten(),
            timer: estimation::Timer::new(),
            attitude_determination: estimation::AttitudeDetermination::new(),
            orbit_determination: estimation::OrbitDetermination::new(),
            direction_estimator: estimation::DirectionEstimator::new(),
            gyro_driver: driver::sensor::GyroDriver::new(UnitQuaternion::identity()),
            magnetometer_driver: driver::sensor::MagnetometerDriver::new(
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f64::consts::FRAC_PI_2),
            ),
            magnetorquer_driver: driver::actuator::MagnetorquerDriver::new(),
            eci_gnss_driver: driver::sensor::EciGnssDriver::new(),
            eci_gnss_pass_through: transformer::gnss_eci::EciGnssPassThroughConversion::new(),
            inertial_mag_field_calc:
                environment::magnetic_field::InertialMagneticFieldCalculation::new(),
            inertial_sun_direction_calc:
                environment::sun_direction::InertialSunDirectionCalculation::new(),
            sun_direction_selector: selector::sun_direction::SunDirectionSelector::new(),
            ss_my_driver: driver::sensor::SunSensorDriver::new(alignment_my),
            ss_py_driver: driver::sensor::SunSensorDriver::new(alignment_py),
            ss_mz_driver: driver::sensor::SunSensorDriver::new(alignment_mz),
            ss_pz_driver: driver::sensor::SunSensorDriver::new(alignment_pz),
            star_tracker_driver: driver::sensor::StarTrackerDriver::new(UnitQuaternion::identity()),
            angular_velocity_selector: selector::angular_velocity::AngularVelocitySelector::new(),
            attitude_controller: controller::AttitudeController::new(),
            rw_driver: driver::actuator::ReactionWheelDriver::new(),
            rw_status_driver: driver::sensor::ReactionWheelStatusSensorDriver::new(),
            cmd_receiver: driver::CommandReceiver::new(),
            time,
            attitude: None,
            angular_velocity: None,
            gyro_bias: None,
            magnetic_dir: None,
            sun_direction: None,
            debug_print_flag,
            profile: FswProfile::default(),
        }
    }

    // FSW の初期化ハンドラ
    pub fn init(&mut self) {
        println!("FSW init");
        let debug_print_flag = self.debug_print_flag;
        *self = Self::new(debug_print_flag);
    }

    pub fn main_loop(&mut self, dt: f64) {
        self.profile.call_count += 1;

        let t0 = Instant::now();
        let gyro_data = self.gyro_driver.main_loop();
        let magnetometer_data = self.magnetometer_driver.main_loop(dt);
        let eci_gnss_data = self.eci_gnss_driver.main_loop();
        let ss_pz_data = self.ss_pz_driver.main_loop();
        let ss_py_data = self.ss_py_driver.main_loop();
        let ss_nz_data = self.ss_mz_driver.main_loop();
        let ss_ny_data = self.ss_my_driver.main_loop();
        let star_tracker_data = self.star_tracker_driver.main_loop();
        let rw_status_data = self.rw_status_driver.main_loop();
        let cmd_data = self.cmd_receiver.main_loop();

        let eci_obs_data = self.eci_gnss_pass_through.main_loop(eci_gnss_data.posvel);

        let ss_data = self
            .sun_direction_selector
            .main_loop([ss_pz_data, ss_py_data, ss_nz_data, ss_ny_data]);

        self.time = self.timer.main_loop(dt, eci_gnss_data.time);
        self.profile.sensor_drivers += t0.elapsed();

        let t0 = Instant::now();
        let inertial_sun_data = self.inertial_sun_direction_calc.main_loop(&self.time);
        self.profile.inertial_sun += t0.elapsed();

        let t0 = Instant::now();
        let orbit_data = self.orbit_determination.main_loop(&self.time, eci_obs_data);
        self.profile.orbit_det += t0.elapsed();

        let t0 = Instant::now();
        let inertial_mag_data = self
            .inertial_mag_field_calc
            .main_loop(&self.time, orbit_data.clone());
        self.profile.inertial_mag += t0.elapsed();

        let t0 = Instant::now();
        let direction_estimator_data = self.direction_estimator.main_loop(
            &self.time,
            &DirectionEstimatorInput {
                sun_direction: ss_data.clone(),
                magnetic_field: magnetometer_data.magnetic_field.clone(),
                inertial_mag: inertial_mag_data.clone(),
                inertial_sun: inertial_sun_data.clone(),
                gyro_data: gyro_data.clone(),
                star_tracker: star_tracker_data.clone(),
            },
        );
        self.profile.direction_estimator += t0.elapsed();

        let t0 = Instant::now();
        let attitude_determination_data = self.attitude_determination.main_loop(
            &self.time,
            &AttitudeDeterminationInput {
                magnetic_field: magnetometer_data.magnetic_field.clone(),
                gyro_data: gyro_data.clone(),
                sun_direction: ss_data.clone(),
                inertial_mag: inertial_mag_data.clone(),
                inertial_sun: inertial_sun_data.clone(),
                star_tracker: star_tracker_data.clone(),
            },
        );
        self.profile.attitude_determination += t0.elapsed();

        let t0 = Instant::now();
        let angular_velocity = self.angular_velocity_selector.main_loop([
            direction_estimator_data.angular_velocity,
            attitude_determination_data.angular_velocity,
        ]);

        let controller_data = self.attitude_controller.main_loop(
            &ControllerInput {
                angular_velocity: angular_velocity.clone(),
                attitude: attitude_determination_data.attitude.clone(),
                sun_direction: direction_estimator_data.sun_direction.clone(),
                magnetic_field: direction_estimator_data.magnetic_field.clone(),
                rw_momentum: rw_status_data,
                cmd: cmd_data,
            },
            dt,
        );
        self.magnetorquer_driver.main_loop(MagnetorquerDriverInput {
            moment: controller_data.mtq_ctrl.clone(),
            exclusive_ctrl: magnetometer_data.exclusive_ctrl,
        });

        self.rw_driver.main_loop(controller_data.rw_ctrl.clone());
        self.profile.controller += t0.elapsed();

        self.attitude = attitude_determination_data
            .attitude
            .as_ref()
            .map(|d| d.attitude);
        self.angular_velocity = angular_velocity.as_ref().map(|d| d.angular_velocity);
        self.gyro_bias = self
            .attitude_determination
            .gyro_bias()
            .as_ref()
            .map(|d| d.gyro_bias);
        self.magnetic_dir = direction_estimator_data
            .magnetic_field
            .as_ref()
            .map(|d| d.magnetic_field_direction);
        self.sun_direction = direction_estimator_data
            .sun_direction
            .as_ref()
            .map(|d| d.sun_direction);

        let t0 = Instant::now();
        if let Some(logger) = self.controller_logger.as_mut() {
            logger
                .serialize((
                    self.time.relative,
                    self.attitude_controller.mode.to_string(),
                    match &self.attitude_controller.mode {
                        AttitudeControllMode::MTQ(mtq) => mtq.mode.to_string(),
                        AttitudeControllMode::RW(rw_control) => format!(
                            "{:?} / {:?}",
                            rw_control.mode.to_string(),
                            rw_control.unloading_mode.to_string()
                        ),
                    },
                    controller_data
                        .mtq_ctrl
                        .clone()
                        .map(|m| m.magnetic_moment.get(0).cloned()),
                    controller_data
                        .mtq_ctrl
                        .clone()
                        .map(|m| m.magnetic_moment.get(1).cloned()),
                    controller_data
                        .mtq_ctrl
                        .clone()
                        .map(|m| m.magnetic_moment.get(2).cloned()),
                    controller_data
                        .rw_ctrl
                        .clone()
                        .map(|t| t.torque.get(0).cloned()),
                    controller_data
                        .rw_ctrl
                        .clone()
                        .map(|t| t.torque.get(1).cloned()),
                    controller_data
                        .rw_ctrl
                        .clone()
                        .map(|t| t.torque.get(2).cloned()),
                ))
                .unwrap();
        };
        if let Some(logger) = self.controller_logger.as_mut() {
            logger.flush().unwrap();
        }

        let gyro_bias = self.attitude_determination.gyro_bias();

        // Write CSV row manually to avoid tuple size limitation (max 16 elements)
        let mut row = vec![format!("{}", self.time.relative)];

        // Quaternion components
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude.scalar()),
        ));
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude.coords.x),
        ));
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude.coords.y),
        ));
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude.coords.z),
        ));

        // Angular velocity
        row.push(format_option(
            angular_velocity.as_ref().map(|d| d.angular_velocity.x),
        ));
        row.push(format_option(
            angular_velocity.as_ref().map(|d| d.angular_velocity.y),
        ));
        row.push(format_option(
            angular_velocity.as_ref().map(|d| d.angular_velocity.z),
        ));

        // Attitude variance
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude_variance[(0, 0)]),
        ));
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude_variance[(1, 1)]),
        ));
        row.push(format_option(
            attitude_determination_data
                .attitude
                .as_ref()
                .map(|d| d.attitude_variance[(2, 2)]),
        ));

        // Angular velocity variance
        row.push(format_option(
            angular_velocity
                .as_ref()
                .map(|d| d.angular_velocity_variance[(0, 0)]),
        ));
        row.push(format_option(
            angular_velocity
                .as_ref()
                .map(|d| d.angular_velocity_variance[(1, 1)]),
        ));
        row.push(format_option(
            angular_velocity
                .as_ref()
                .map(|d| d.angular_velocity_variance[(2, 2)]),
        ));

        // Gyro bias
        row.push(format_option(gyro_bias.as_ref().map(|d| d.gyro_bias.x)));
        row.push(format_option(gyro_bias.as_ref().map(|d| d.gyro_bias.y)));
        row.push(format_option(gyro_bias.as_ref().map(|d| d.gyro_bias.z)));

        // Gyro bias variance
        row.push(format_option(
            gyro_bias.as_ref().map(|d| d.gyro_bias_variance[(0, 0)]),
        ));
        row.push(format_option(
            gyro_bias.as_ref().map(|d| d.gyro_bias_variance[(1, 1)]),
        ));
        row.push(format_option(
            gyro_bias.as_ref().map(|d| d.gyro_bias_variance[(2, 2)]),
        ));

        if let Some(logger) = self.attitude_logger.as_mut() {
            logger.write_record(&row).unwrap();
        }
        if let Some(logger) = self.attitude_logger.as_mut() {
            logger.flush().unwrap();
        }

        if let Some(logger) = self.orbit_logger.as_mut() {
            logger
                .serialize((
                    self.time.relative,
                    orbit_data.clone().map(|data| data.position.x),
                    orbit_data.clone().map(|data| data.position.y),
                    orbit_data.clone().map(|data| data.position.z),
                    orbit_data.clone().map(|data| data.velocity.x),
                    orbit_data.clone().map(|data| data.velocity.y),
                    orbit_data.clone().map(|data| data.velocity.z),
                ))
                .unwrap();
        }
        if let Some(logger) = self.orbit_logger.as_mut() {
            logger.flush().unwrap();
        }

        let magnetic_field = direction_estimator_data
            .magnetic_field
            .clone()
            .map(|data| data.magnetic_field_norm * data.magnetic_field_direction.into_inner());

        if let Some(logger) = self.direction_logger.as_mut() {
            logger
                .serialize((
                    self.time.relative,
                    direction_estimator_data
                        .sun_direction
                        .clone()
                        .map(|data| data.sun_direction.x),
                    direction_estimator_data
                        .sun_direction
                        .clone()
                        .map(|data| data.sun_direction.y),
                    direction_estimator_data
                        .sun_direction
                        .clone()
                        .map(|data| data.sun_direction.z),
                    magnetic_field.map(|data| data.x),
                    magnetic_field.map(|data| data.y),
                    magnetic_field.map(|data| data.z),
                    // Sun direction variance diagonal elements (2x2, sigma^2)
                    direction_estimator_data
                        .sun_direction
                        .clone()
                        .map(|data| data.sun_direction_variance[(0, 0)]),
                    direction_estimator_data
                        .sun_direction
                        .clone()
                        .map(|data| data.sun_direction_variance[(1, 1)]),
                    // Magnetic field variance diagonal elements (3x3, sigma^2)
                    direction_estimator_data
                        .magnetic_field
                        .clone()
                        .map(|data| data.magnetic_field_direction_variance[(0, 0)]),
                    direction_estimator_data
                        .magnetic_field
                        .clone()
                        .map(|data| data.magnetic_field_direction_variance[(1, 1)]),
                    direction_estimator_data
                        .magnetic_field
                        .clone()
                        .map(|data| data.magnetic_field_norm_variance[(0, 0)]),
                ))
                .unwrap();
        }
        self.profile.logging += t0.elapsed();
    }

    pub fn input_ports(&'_ mut self) -> C5AInputPortSet<'_> {
        C5AInputPortSet {
            gyro: &mut self.gyro_driver.sim_port,
            magnetometer: &mut self.magnetometer_driver.sim_port,
            eci_gnss: &mut self.eci_gnss_driver.sim_port,
            sun_sensor_pz: &mut self.ss_pz_driver.sim_port,
            sun_sensor_py: &mut self.ss_py_driver.sim_port,
            sun_sensor_mz: &mut self.ss_mz_driver.sim_port,
            sun_sensor_my: &mut self.ss_my_driver.sim_port,
            star_tracker: &mut self.star_tracker_driver.sim_port,
            rw_status: &mut self.rw_status_driver.sim_port,
            cmd: &mut self.cmd_receiver.sim_port,
        }
    }

    pub fn output_ports(&'_ self) -> C5AOutputPortSet<'_> {
        C5AOutputPortSet {
            magnetorquer_ctrl: &self.magnetorquer_driver.sim_port,
            reaction_wheel_ctrl: &self.rw_driver.sim_port,
        }
    }
}

pub struct C5AInputPortSet<'a> {
    pub gyro: &'a mut S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer: &'a mut S5ESubscribePort<s5e_port::MagnetometerData>,
    pub eci_gnss: &'a mut S5ESubscribePort<s5e_port::ECIGnssData>,
    pub sun_sensor_pz: &'a mut S5ESubscribePort<s5e_port::LightDetectionSystemData>,
    pub sun_sensor_py: &'a mut S5ESubscribePort<s5e_port::LightDetectionSystemData>,
    pub sun_sensor_mz: &'a mut S5ESubscribePort<s5e_port::LightDetectionSystemData>,
    pub sun_sensor_my: &'a mut S5ESubscribePort<s5e_port::LightDetectionSystemData>,
    pub star_tracker: &'a mut S5ESubscribePort<s5e_port::StarTrackerData>,
    pub rw_status: &'a mut S5ESubscribePort<s5e_port::ReactionWheelRotationData>,
    pub cmd: &'a mut S5ESubscribePort<s5e_port::GSCommandData>,
}

pub struct C5AOutputPortSet<'a> {
    pub magnetorquer_ctrl: &'a s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,
    pub reaction_wheel_ctrl: &'a s5e_port::S5EPublishPort<s5e_port::ReactionWheelCtrlEvent>,
}
