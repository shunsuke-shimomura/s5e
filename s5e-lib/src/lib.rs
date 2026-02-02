use astrodynamics::time::JulianDate;
use chrono::{NaiveDateTime, TimeDelta};
use control_system::integrator::Prediction;
use s5e_port::{S5EPublishPort, S5ESubscribePort};
use core::f64;
use debug_s5e::debug_actuator;
use nalgebra::geometry::UnitQuaternion;
use nalgebra::{Matrix3, Quaternion, Vector3};
use rand_distr::num_traits::One;
use sgp4::Elements;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::actuator::{Actuator, Magnetorquer, ReactionWheel};

use crate::orbit::{TLE, Trajectory};
use crate::sensor::{
    ECIGnssSensor, ECIGnssSensorInput, GyroSensor, Magnetometer, Sensor, SensorWithInterval,
    StarTracker, SunPositionAndIrradiance, SunSensorWithDetectionLogic,
};
use astrodynamics::coordinate::{
    BodyVector, ECEFPosition, ECEFVector, ECEFVelocity, ECIPosition, ECIVector, ECIVelocity,
    GeodeticPosition, GeodeticVector,
};
use control_system::integrator::{
    TimeIntegrator,
    rk4::{RK4Input, RK4InputPrediction, RK4Phase, RK4Solver},
};

pub mod actuator;
pub mod earth;
pub mod moon;
pub mod orbit;
pub mod sensor;
pub mod spice_if;
pub mod sun;

pub struct S5ETimeEvent {
    function: Option<Arc<dyn Fn()>>,
    trigger_time: f64,
}

impl S5ETimeEvent {
    pub fn new<F>(function: F, trigger_time: f64) -> Self
    where
        F: Fn() + 'static,
    {
        Self {
            function: Some(Arc::new(function)),
            trigger_time,
        }
    }

    pub fn tick(&mut self, current_time: f64) {
        if current_time >= self.trigger_time
            && let Some(function) = &self.function.take()
        {
            function();
        }
    }
}

pub struct S5ETriggerEvent<T> {
    #[allow(clippy::type_complexity)]
    function: Option<Arc<dyn Fn(&mut T)>>,
}

impl<T: Clone> Clone for S5ETriggerEvent<T> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
        }
    }
}

impl<T> S5ETriggerEvent<T> {
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&mut T) + 'static,
    {
        Self {
            function: Some(Arc::new(function)),
        }
    }

    pub fn trigger(&mut self, target: &mut T) {
        if let Some(function) = self.function.take() {
            function(target);
        }
    }
}

#[derive(Default)]
pub struct TickProfile {
    pub shadow_calc: Duration,
    pub sensor_ticks: Duration,
    pub fsw_input: Duration,
    pub fsw_main_loop: Duration,
    pub actuator_ticks: Duration,
    pub attitude_dynamics: Duration,
    pub attitude_kinematics: Duration,
    pub orbit_propagation: Duration,
    pub coord_conversion: Duration,
    pub igrf_calc: Duration,
    pub call_count: u64,
}

impl TickProfile {
    pub fn print_summary(&self) {
        let total = self.shadow_calc
            + self.sensor_ticks
            + self.fsw_input
            + self.fsw_main_loop
            + self.actuator_ticks
            + self.attitude_dynamics
            + self.attitude_kinematics
            + self.orbit_propagation
            + self.coord_conversion
            + self.igrf_calc;
        let pct = |d: Duration| {
            if total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / total.as_secs_f64() * 100.0
            }
        };
        println!(
            "=== SpaceCraft::tick() Profile ({} calls, total {:.3}s) ===",
            self.call_count,
            total.as_secs_f64()
        );
        println!(
            "  shadow_calc:          {:>8.3}ms ({:>5.1}%)",
            self.shadow_calc.as_secs_f64() * 1000.0,
            pct(self.shadow_calc)
        );
        println!(
            "  sensor_ticks:         {:>8.3}ms ({:>5.1}%)",
            self.sensor_ticks.as_secs_f64() * 1000.0,
            pct(self.sensor_ticks)
        );
        println!(
            "  fsw_input:            {:>8.3}ms ({:>5.1}%)",
            self.fsw_input.as_secs_f64() * 1000.0,
            pct(self.fsw_input)
        );
        println!(
            "  fsw_main_loop:        {:>8.3}ms ({:>5.1}%)",
            self.fsw_main_loop.as_secs_f64() * 1000.0,
            pct(self.fsw_main_loop)
        );
        println!(
            "  actuator_ticks:       {:>8.3}ms ({:>5.1}%)",
            self.actuator_ticks.as_secs_f64() * 1000.0,
            pct(self.actuator_ticks)
        );
        println!(
            "  attitude_dynamics:    {:>8.3}ms ({:>5.1}%)",
            self.attitude_dynamics.as_secs_f64() * 1000.0,
            pct(self.attitude_dynamics)
        );
        println!(
            "  attitude_kinematics:  {:>8.3}ms ({:>5.1}%)",
            self.attitude_kinematics.as_secs_f64() * 1000.0,
            pct(self.attitude_kinematics)
        );
        println!(
            "  orbit_propagation:    {:>8.3}ms ({:>5.1}%)",
            self.orbit_propagation.as_secs_f64() * 1000.0,
            pct(self.orbit_propagation)
        );
        println!(
            "  coord_conversion:     {:>8.3}ms ({:>5.1}%)",
            self.coord_conversion.as_secs_f64() * 1000.0,
            pct(self.coord_conversion)
        );
        println!(
            "  igrf_calc:            {:>8.3}ms ({:>5.1}%)",
            self.igrf_calc.as_secs_f64() * 1000.0,
            pct(self.igrf_calc)
        );
    }
}

pub struct SpaceCraftInitialState {
    pub tle: TLE,
    pub attitude: UnitQuaternion<f64>,
    pub inertia: Matrix3<f64>,
    pub angular_velocity: BodyVector,
}

pub struct SpaceCraftComponentsProfile {
    pub gyro_std: f64,
    pub magnetometer_std: f64,
    pub star_tracker_std: f64,
    pub sun_sensor_dir_std: f64,
    pub rw_noise_std: f64,
    pub mtq_noise_std: f64,
    pub mtq_max_dipole_moment: f64,
}

pub struct SpaceCraft<FSW> 
{
    pub inertia: Matrix3<f64>,

    // component
    pub obc: Obc<FSW>,
    pub mtq: Magnetorquer,
    pub rw: ReactionWheel,

    pub ss_pz: SunSensorWithDetectionLogic,
    pub ss_py: SunSensorWithDetectionLogic,
    pub ss_mz: SunSensorWithDetectionLogic,
    pub ss_my: SunSensorWithDetectionLogic,
    pub magnetometer: Magnetometer,
    pub gyro_sensor: GyroSensor,
    pub star_tracker: SensorWithInterval<StarTracker>,
    pub eci_gnss_sensor: ECIGnssSensor,

    pub attitude: RK4Solver<Quaternion<f64>, NaiveDateTime, f64>,

    pub angular_velocity: RK4Solver<BodyVector, NaiveDateTime, f64>,

    pub eci_position: RK4Input<ECIPosition, NaiveDateTime, f64>,
    pub eci_velocity: RK4Input<ECIVelocity, NaiveDateTime, f64>,

    ecef_position: RK4Input<ECEFPosition, NaiveDateTime, f64>,
    ecef_velocity: RK4Input<ECEFVelocity, NaiveDateTime, f64>,

    pub geodetic_position: RK4Input<GeodeticPosition, NaiveDateTime, f64>,

    pub magnetic_field: ECIVector, // 単位はnT

    pub sun_vector_body: BodyVector, // 単位はm（ボディ座標系）

    pub shadow_coefficient: sun::ShadowCoefficient,
    pub solar_irradiance: sun::Irradiance, // 単位はW/m²

    trajectory: Trajectory,

    pcdu_voltage: f64,

    pub profile: TickProfile,
}

impl<FSW> SpaceCraft<FSW>
{
    pub fn new(
        datetime: NaiveDateTime,
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        fsw: FSW,
        components_profile: SpaceCraftComponentsProfile,
        sat_initial_state: SpaceCraftInitialState,
    ) -> Self {
        let attitude = RK4Solver::new(sat_initial_state.attitude.normalize());
        let inertia = sat_initial_state.inertia;
        let angular_velocity = RK4Solver::new(sat_initial_state.angular_velocity.clone());
        let trajectory = Trajectory::new(
            Elements::from_tle(None, sat_initial_state.tle.line1.as_bytes(), sat_initial_state.tle.line2.as_bytes()).unwrap(),
        )
        .unwrap();
        let (eci_position, eci_velocity) = trajectory.initial_kinematics(datetime);
        let ecef_position =
            ECEFPosition::from_eci(eci_position.clone(), JulianDate::from(datetime));
        let geodetic_position = GeodeticPosition::from(ecef_position.clone());
        let mag_field_geodetic = {
            let field_info = igrf::declination(
                geodetic_position.latitude,
                geodetic_position.longitude,
                geodetic_position.altitude as u32,
                time::Date::from_julian_day(JulianDate::from(datetime).value as i32).unwrap(),
            )
            .unwrap();
            GeodeticVector {
                x: field_info.x,
                y: field_info.y,
                z: field_info.z,
            }
        };

        let mag_field_ecef = ECEFVector::from_geodetic(mag_field_geodetic, geodetic_position);
        let magnetic_field = ECIVector::from_ecef(mag_field_ecef, JulianDate::from(datetime));
        let alignment_pz =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f64::consts::FRAC_PI_2);
        let alignment_py =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), f64::consts::FRAC_PI_2);
        let alignment_mz = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), f64::consts::PI);
        let alignment_my =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -f64::consts::FRAC_PI_2);

        SpaceCraft {
            attitude,
            angular_velocity,
            eci_position: RK4Input::new(eci_position.clone()),
            eci_velocity: RK4Input::new(eci_velocity.clone()),
            ecef_position: RK4Input::new(ECEFPosition::from_eci(
                eci_position.clone(),
                JulianDate::from(datetime),
            )),
            ecef_velocity: RK4Input::new(ECEFVelocity::from_eci(
                eci_velocity.clone(),
                eci_position.clone(),
                JulianDate::from(datetime),
            )),
            geodetic_position: RK4Input::new(GeodeticPosition::from(ECEFPosition::from_eci(
                eci_position.clone(),
                JulianDate::from(datetime),
            ))),
            magnetic_field,
            trajectory,
            inertia,
            pcdu_voltage: 10.0,
            obc: Obc {
                fsw,
                is_on: false,
                voltage: 10.0,
            },
            mtq: Magnetorquer::new(rng.clone(), components_profile.mtq_noise_std),
            rw: ReactionWheel::new(
                Matrix3::new(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
                components_profile.rw_noise_std,
                rng.clone(),
            ),
            magnetometer: Magnetometer::new(
                rng.clone(),
                components_profile.magnetometer_std,
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), f64::consts::FRAC_PI_2),
            ),
            ss_my: SunSensorWithDetectionLogic::new(
                alignment_my,
                components_profile.sun_sensor_dir_std,
                sun::SOLAR_CONSTANT * 0.01,
                f64::consts::PI / 180.0 * 60.0,
                sun::SOLAR_CONSTANT * 0.3,
                rng.clone(),
            ),
            ss_mz: SunSensorWithDetectionLogic::new(
                alignment_mz,
                components_profile.sun_sensor_dir_std,
                sun::SOLAR_CONSTANT * 0.01,
                f64::consts::PI / 180.0 * 60.0,
                sun::SOLAR_CONSTANT * 0.3,
                rng.clone(),
            ),
            ss_py: SunSensorWithDetectionLogic::new(
                alignment_py,
                components_profile.sun_sensor_dir_std,
                sun::SOLAR_CONSTANT * 0.01,
                f64::consts::PI / 180.0 * 60.0,
                sun::SOLAR_CONSTANT * 0.3,
                rng.clone(),
            ),
            ss_pz: SunSensorWithDetectionLogic::new(
                alignment_pz,
                components_profile.sun_sensor_dir_std,
                sun::SOLAR_CONSTANT * 0.01,
                f64::consts::PI / 180.0 * 60.0,
                sun::SOLAR_CONSTANT * 0.3,
                rng.clone(),
            ),
            gyro_sensor: GyroSensor::new(rng.clone(), components_profile.gyro_std, UnitQuaternion::one()),
            star_tracker: SensorWithInterval::new(
                StarTracker::new(rng.clone(), components_profile.star_tracker_std, UnitQuaternion::one()),
                1.0,
            ),
            eci_gnss_sensor: ECIGnssSensor::new(
                rng.clone(),
                1.0,
                1.0,
                BodyVector {
                    x: 0.1,
                    y: 0.0,
                    z: 0.0,
                },
            ),

            shadow_coefficient: sun::ShadowCoefficient { value: 1.0 },
            solar_irradiance: sun::Irradiance { value: 0.0 },
            sun_vector_body: BodyVector {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            profile: TickProfile::default(),
        }
    }

    pub fn tick(
        &mut self,
        dt: f64,
        datetime: NaiveDateTime,
        sun_light_source: sun::LightSource,
        shadow_sources: Vec<sun::ShadowSource>,
        cmd_port: &s5e_port::S5EPublishPort<s5e_port::GSCommandData>,
    )
    where
        FSW: S5EFswInterface
    {
        self.profile.call_count += 1;

        let t0 = Instant::now();
        self.shadow_coefficient = sun::shadow_coefficient(
            sun_light_source.clone(),
            shadow_sources,
            self.eci_position.get_now(),
        );
        self.solar_irradiance = self.shadow_coefficient.clone()
            * sun::irradiance(sun_light_source.clone(), self.eci_position.get_now());

        let sun_vector_eci = sun_light_source.position - self.eci_position.get_now();
        self.obc.voltage = self.pcdu_voltage;

        self.sun_vector_body = BodyVector::from_eci(
            sun_vector_eci,
            UnitQuaternion::from_quaternion(self.attitude.get_now()),
        );
        self.profile.shadow_calc += t0.elapsed();

        let t0 = Instant::now();
        let sun_position_and_irradiance = SunPositionAndIrradiance {
            sun_position: self.sun_vector_body.clone(),
            irradiance: self.solar_irradiance.clone(),
        };
        // tick sensors
        self.ss_pz.sensor_tick(sun_position_and_irradiance.clone());
        self.ss_py.sensor_tick(sun_position_and_irradiance.clone());
        self.ss_mz.sensor_tick(sun_position_and_irradiance.clone());
        self.ss_my.sensor_tick(sun_position_and_irradiance.clone());
        self.magnetometer.sensor_tick(BodyVector::from_eci(
            self.magnetic_field.clone(),
            UnitQuaternion::new_normalize(self.attitude.get_now()),
        ));
        self.gyro_sensor
            .sensor_tick(self.angular_velocity.get_now());
        self.star_tracker
            .sensor_tick(UnitQuaternion::new_normalize(self.attitude.get_now()), dt);
        self.eci_gnss_sensor.sensor_tick(ECIGnssSensorInput {
            position: self.eci_position.get_now(),
            velocity: self.eci_velocity.get_now(),
            attitude: UnitQuaternion::new_normalize(self.attitude.get_now()),
            angular_velocity: self.angular_velocity.get_now(),
            time: datetime,
        });
        self.rw.sensor_tick(());
        self.profile.sensor_ticks += t0.elapsed();

        // tick software logic
        let t0 = Instant::now();
        {
            let sensor_output = SensorOutputPortSet {
                gyro: self.gyro_sensor.sensor_port(),
                magnetometer: self.magnetometer.sensor_port(),
                eci_gnss: self.eci_gnss_sensor.sensor_port(),
                sun_sensor_pz: self.ss_pz.sensor_port(),
                sun_sensor_py: self.ss_py.sensor_port(),
                sun_sensor_mz: self.ss_mz.sensor_port(),
                sun_sensor_my: self.ss_my.sensor_port(),
                star_tracker: self.star_tracker.port(),
                rw_status: self.rw.sensor_port(),
                cmd: cmd_port,
            };
            let mut fsw_input_port = self.obc.fsw.input_ports();
            fsw_input_port.transfer_from(&sensor_output);
        }
        self.profile.fsw_input += t0.elapsed();

        let t0 = Instant::now();
        self.obc.tick(dt, self.pcdu_voltage);
        self.profile.fsw_main_loop += t0.elapsed();

        let t0 = Instant::now();
        // send control data (logic -> actuator)
        {
            let fsw_output = self.obc.fsw.output_ports();
            let mut actuator_input_port = ActuatorInputPortSet {
                magnetorquer_ctrl: self.mtq.actuator_port(),
                reaction_wheel_ctrl: self.rw.actuator_port(),
            };
            fsw_output.transfer_to(&mut actuator_input_port);
        }

        // actuator tick
        self.mtq.actuator_tick((), datetime, dt);
        self.rw.actuator_tick((), datetime, dt);
        self.profile.actuator_ticks += t0.elapsed();

        // physical simulation

        /*
        姿勢ダイナミクス
            dω/dt = I^(-1) * (τ - ω × (I * ω))
        ここで、τはボディ座標でのトルクベクトル、ωはボディ座標での角速度ベクトル、Iはボディ座標での慣性テンソル（不変）。
        */
        let t0 = Instant::now();
        {
            let f = |phase: RK4Phase, _, angular_velocity: BodyVector| {
                let attitude = match self.attitude.get(phase) {
                    Some(a) => a,
                    None => self.attitude.get_now(),
                };
                let mtq_moment = self.mtq.output(phase);
                let mtq_moment_eci = ECIVector::from_body(
                    mtq_moment.clone(),
                    UnitQuaternion::from_quaternion(attitude),
                );
                #[allow(non_snake_case)]
                let magnetic_field_T = 1.0e-9 * Vector3::from(self.magnetic_field.clone());
                let torque_eci = Vector3::from(mtq_moment_eci).cross(&magnetic_field_T);
                let mag_torque = BodyVector::from_eci(
                    ECIVector::from(torque_eci),
                    UnitQuaternion::from_quaternion(attitude),
                );

                debug_actuator!("Magnetic Torque Calculation:");
                debug_actuator!(
                    "  MTQ moment (body): x={:.6}, y={:.6}, z={:.6} Am²",
                    mtq_moment.x,
                    mtq_moment.y,
                    mtq_moment.z
                );
                debug_actuator!(
                    "  Magnetic field: x={:.6}, y={:.6}, z={:.6} T",
                    magnetic_field_T.x,
                    magnetic_field_T.y,
                    magnetic_field_T.z
                );
                debug_actuator!(
                    "  Torque (body): x={:.6}, y={:.6}, z={:.6} Nm",
                    mag_torque.x,
                    mag_torque.y,
                    mag_torque.z
                );

                let rw_torque = Vector3::from(self.rw.output(phase));
                let input_torque = Vector3::from(mag_torque.clone()) + rw_torque;

                let gyroscopic_torque = Vector3::from(angular_velocity.clone())
                    .cross(&(self.inertia * Vector3::from(angular_velocity.clone())));
                let angular_acceleration =
                    self.inertia.try_inverse().unwrap() * (input_torque - gyroscopic_torque);

                BodyVector::from(angular_acceleration)
            };

            self.angular_velocity.propagate(f, dt, datetime);
        }
        self.profile.attitude_dynamics += t0.elapsed();

        /*
        姿勢のダイナミクス
            dQ/dt = 1/2 * Q * (0, ωx, ωy, ωz)
        ここで、Qは慣性座標からボディ座標へのクォータニオン、ωはボディ座標での角速度ベクトル。
        */
        let t0 = Instant::now();
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

            self.attitude.propagate(f, dt, datetime);
        }
        self.profile.attitude_kinematics += t0.elapsed();

        let t0 = Instant::now();
        let (eci_position, eci_velocity) = self.trajectory.propagate(datetime, dt);

        self.eci_position.set(eci_position);
        self.eci_velocity.set(eci_velocity);
        self.profile.orbit_propagation += t0.elapsed();

        let t0 = Instant::now();
        self.ecef_position.set(RK4InputPrediction {
            after_halfdt: ECEFPosition::from_eci(
                self.eci_position.get(RK4Phase::HalfDt).unwrap(),
                JulianDate::from(datetime + TimeDelta::nanoseconds((dt / 2.0 * 1.0e9) as i64)),
            ),
            after_dt: ECEFPosition::from_eci(
                self.eci_position.get(RK4Phase::Dt).unwrap(),
                JulianDate::from(datetime + TimeDelta::nanoseconds((dt * 1.0e9) as i64)),
            ),
            dt,
            time: datetime,
        });

        self.ecef_velocity.set(RK4InputPrediction {
            after_halfdt: ECEFVelocity::from_eci(
                self.eci_velocity.get(RK4Phase::HalfDt).unwrap(),
                self.eci_position.get(RK4Phase::HalfDt).unwrap(),
                JulianDate::from(datetime + TimeDelta::nanoseconds((dt / 2.0 * 1.0e9) as i64)),
            ),
            after_dt: ECEFVelocity::from_eci(
                self.eci_velocity.get(RK4Phase::Dt).unwrap(),
                self.eci_position.get(RK4Phase::Dt).unwrap(),
                JulianDate::from(datetime + TimeDelta::nanoseconds((dt * 1.0e9) as i64)),
            ),
            dt,
            time: datetime,
        });

        self.geodetic_position.set(RK4InputPrediction {
            after_halfdt: GeodeticPosition::from(self.ecef_position.get(RK4Phase::HalfDt).unwrap()),
            after_dt: GeodeticPosition::from(self.ecef_position.get(RK4Phase::Dt).unwrap()),
            dt,
            time: datetime,
        });
        self.profile.coord_conversion += t0.elapsed();

        let t0 = Instant::now();
        let mag_field_geodetic = {
            let field_info = igrf::declination(
                self.geodetic_position.get_now().latitude,
                self.geodetic_position.get_now().longitude,
                self.geodetic_position.get_now().altitude as u32,
                time::Date::from_julian_day(JulianDate::from(datetime).value as i32).unwrap(),
            )
            .unwrap();
            GeodeticVector {
                x: field_info.x,
                y: field_info.y,
                z: field_info.z,
            }
        };

        let mag_field_ecef =
            ECEFVector::from_geodetic(mag_field_geodetic, self.geodetic_position.get_now());
        self.magnetic_field = ECIVector::from_ecef(mag_field_ecef, JulianDate::from(datetime));
        self.profile.igrf_calc += t0.elapsed();
    }

    pub fn clear_state(&mut self) {
        self.attitude.clear();
        self.angular_velocity.clear();
        self.eci_position.clear();
        self.eci_velocity.clear();
        self.ecef_position.clear();
        self.ecef_velocity.clear();
        self.geodetic_position.clear();
        self.ss_my.sensor_clear();
        self.ss_mz.sensor_clear();
        self.ss_py.sensor_clear();
        self.ss_pz.sensor_clear();
        self.magnetometer.sensor_clear();
        self.gyro_sensor.sensor_clear();
        self.star_tracker.sensor_clear();
        self.eci_gnss_sensor.sensor_clear();
        self.rw.sensor_clear();
        self.mtq.actuator_clear();
        self.rw.actuator_clear();
    }
}

pub struct SensorOutputPortSet<'a> {
    pub magnetometer: &'a S5EPublishPort<s5e_port::MagnetometerData>,
    pub gyro: &'a S5EPublishPort<s5e_port::GyroSensorData>,
    pub eci_gnss: &'a S5EPublishPort<s5e_port::ECIGnssData>,
    pub sun_sensor_pz: &'a S5EPublishPort<s5e_port::LightDetectionSystemData>,
    pub sun_sensor_py: &'a S5EPublishPort<s5e_port::LightDetectionSystemData>,
    pub sun_sensor_mz: &'a S5EPublishPort<s5e_port::LightDetectionSystemData>,
    pub sun_sensor_my: &'a S5EPublishPort<s5e_port::LightDetectionSystemData>,
    pub star_tracker: &'a S5EPublishPort<s5e_port::StarTrackerData>,
    pub rw_status: &'a S5EPublishPort<s5e_port::ReactionWheelRotationData>,
    pub cmd: &'a S5EPublishPort<s5e_port::GSCommandData>,
}

pub struct ActuatorInputPortSet<'a> {
    pub magnetorquer_ctrl: &'a mut S5ESubscribePort<s5e_port::MagnetorquerCtrlEvent>,
    pub reaction_wheel_ctrl: &'a mut S5ESubscribePort<s5e_port::ReactionWheelCtrlEvent>,
}

pub trait FswInputTransfer<SensorOutputPorts> {
    fn transfer_from(&mut self, sensor_output: &SensorOutputPorts);
}

pub trait FswOutputTransfer<ActuatorInputPorts> {
    fn transfer_to(&self, actuator_input: &mut ActuatorInputPorts);
}

pub trait S5EFswInterface {
    type InputPortSet<'a>: FswInputTransfer<SensorOutputPortSet<'a>> where Self: 'a;
    type OutputPortSet<'a>: FswOutputTransfer<ActuatorInputPortSet<'a>> where Self: 'a;
    fn input_ports(&mut self) -> Self::InputPortSet<'_>;
    fn output_ports(&self) -> Self::OutputPortSet<'_>;
    fn init(&mut self);
    fn main_loop(&mut self, dt: f64);
}

pub struct Obc<FSW> {
    is_on: bool,
    pub fsw: FSW,
    voltage: f64,
}

impl<FSW> Obc<FSW>
where
    FSW: S5EFswInterface,
{
    fn tick(&mut self, dt: f64, voltage: f64) {
        if !self.is_on && voltage > 5.0 {
            // PCDU からの電圧が何V以上になってたら、みたいな
            self.is_on = true;
            self.fsw.init();
        } 
        if self.is_on {
            self.fsw.main_loop(dt);
        } 
    }
}
