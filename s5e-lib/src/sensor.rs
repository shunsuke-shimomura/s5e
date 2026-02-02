use chrono::{Datelike, Timelike};
use debug_s5e::debug_sensor;
use nalgebra::{UnitQuaternion, Vector3};
use rand_distr::{Distribution, Normal};
use std::rc::Rc;
use std::{cell::RefCell, fmt::Debug};

use crate::sun::Irradiance;
use astrodynamics::coordinate::{
    BodyVector, ComponentDirection, ComponentVector, DirectionNose, ECIPosition, ECIVelocity,
    MIN_NORMALIZE_THRESHOLD,
};

pub trait Sensor {
    type IN: Clone;
    type OUT: Clone + Debug;

    fn sensor_tick(&mut self, input: Self::IN);
    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<Self::OUT>;
    fn sensor_clear(&mut self);
}

pub struct TimeCounter<T: Clone + Debug> {
    pub interval: f64,
    pub time_count: u32,
    pub in_port: s5e_port::S5ESubscribePort<T>,
    pub out_port: s5e_port::S5EPublishPort<T>,
}

impl<T: Clone + Debug> TimeCounter<T> {
    pub fn new(interval: f64) -> Self {
        Self {
            interval,
            time_count: 0,
            in_port: s5e_port::S5ESubscribePort::new(),
            out_port: s5e_port::S5EPublishPort::new(),
        }
    }

    pub fn tick(&mut self, dt: f64) {
        self.time_count += 1;
        if self.time_count > (self.interval / dt) as u32 {
            if let Some(data) = self.in_port.subscribe() {
                self.out_port.publish(data);
            }
            self.time_count = 0;
        }
    }
}

pub struct SensorWithInterval<S>
where
    S: Sensor,
{
    pub sensor: S,
    pub time_counter: TimeCounter<S::OUT>,
    phantom: std::marker::PhantomData<S::IN>,
}

impl<S: Sensor> SensorWithInterval<S> {
    pub fn new(sensor: S, interval: f64) -> Self {
        Self {
            sensor,
            time_counter: TimeCounter::new(interval),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn sensor_tick(&mut self, input: S::IN, dt: f64) {
        self.sensor.sensor_tick(input);
        s5e_port::transfer(self.sensor.sensor_port(), &mut self.time_counter.in_port);
        self.time_counter.tick(dt);
    }

    pub fn port(&self) -> &s5e_port::S5EPublishPort<S::OUT> {
        &self.time_counter.out_port
    }

    pub fn sensor_clear(&mut self) {
        self.sensor.sensor_clear();
        self.time_counter.out_port.clear();
    }
}

//　ほぼフォトダイオードだけど、フォトダイオードと違って放射照度を直接測定できる
pub struct IrradianceSensor {
    pub irradiance: Irradiance,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::IrradianceSensorData>,
    pub alignment: UnitQuaternion<f64>, // z軸で観測する想定のbodyからcomponentへの変換
}

impl IrradianceSensor {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: UnitQuaternion<f64>,
    ) -> Self {
        IrradianceSensor {
            irradiance: Irradiance { value: 0.0 },
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, irradiance: BodyVector) {
        let mut rng_ref = self.rng.borrow_mut();
        let noise_value = self.noise.sample(&mut *rng_ref);
        let irradiance_c = ComponentVector::from_body(irradiance, self.alignment);
        self.irradiance.value = irradiance_c.z + noise_value;
        self.port.publish(s5e_port::IrradianceSensorData {
            irradiance: self.irradiance.value,
        });
    }
}

impl Sensor for IrradianceSensor {
    type IN = BodyVector;
    type OUT = s5e_port::IrradianceSensorData;

    fn sensor_tick(&mut self, irradiance: Self::IN) {
        self.measure(irradiance);
        debug_sensor!("[IrradianceSensor] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::IrradianceSensorData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

pub struct SunSensor {
    pub sun_direction: ComponentDirection,
    pub noise: DirectionNose,
    pub port: s5e_port::S5EPublishPort<s5e_port::LightDirectionData>,
    pub alignment: UnitQuaternion<f64>,
}

impl SunSensor {
    pub fn new(
        angle_noise_stddev: f64,
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        alignment: UnitQuaternion<f64>,
    ) -> Self {
        SunSensor {
            sun_direction: ComponentDirection::default(),
            noise: DirectionNose::new(angle_noise_stddev, rng),
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, sun_position: BodyVector) {
        let component_vector = ComponentVector::from_body(sun_position, self.alignment);
        let sun_direction = ComponentDirection::try_from(component_vector)
            .map(|v| v.add_noise(&self.noise))
            .unwrap_or_else(|error| {
                debug_sensor!(
                    "[SunSensor] Failed to convert BodyVector to ComponentDirection: {}",
                    error
                );
                ComponentDirection::default()
            });
        self.sun_direction = sun_direction.clone();
        self.port.publish(s5e_port::LightDirectionData {
            light_direction: Vector3::from(sun_direction),
        });
    }
}

impl Sensor for SunSensor {
    type IN = BodyVector;
    type OUT = s5e_port::LightDirectionData;

    fn sensor_tick(&mut self, sun_position: Self::IN) {
        self.measure(sun_position);
        debug_sensor!("[SunSensor] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::LightDirectionData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

// z軸が観測可能方向という前提
pub struct LightDetectionSystem {
    // input
    pub light_direction_port: s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,
    pub irradiance_port: s5e_port::S5ESubscribePort<s5e_port::IrradianceSensorData>,
    // output
    pub system_port: s5e_port::S5EPublishPort<s5e_port::LightDetectionSystemData>,
    // parameters
    pub irradiance_threshold: f64,
    pub detectable_angle: f64,
}

impl LightDetectionSystem {
    pub fn new(detectable_angle: f64, irradiance_threshold: f64) -> Self {
        LightDetectionSystem {
            light_direction_port: s5e_port::S5ESubscribePort::new(),
            irradiance_port: s5e_port::S5ESubscribePort::new(),
            system_port: s5e_port::S5EPublishPort::new(),
            irradiance_threshold,
            detectable_angle,
        }
    }
}

impl Sensor for LightDetectionSystem {
    type IN = ();
    type OUT = s5e_port::LightDetectionSystemData;

    fn sensor_tick(&mut self, _: Self::IN) {
        if let Some(irradiance_data) = self.irradiance_port.subscribe() {
            let direction = if let Some(direction_data) = self.light_direction_port.subscribe()
                && direction_data.light_direction.z.acos() < self.detectable_angle
                && irradiance_data.irradiance > self.irradiance_threshold
            {
                Some(direction_data.light_direction)
            } else {
                None
            };
            self.system_port
                .publish(s5e_port::LightDetectionSystemData {
                    light_direction: direction,
                    irradiance: irradiance_data.irradiance,
                });
        }
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::LightDetectionSystemData> {
        &self.system_port
    }

    fn sensor_clear(&mut self) {
        self.system_port.clear();
    }
}

#[derive(Debug, Clone)]
pub struct SunPositionAndIrradiance {
    pub sun_position: BodyVector,
    pub irradiance: Irradiance,
}

pub struct SunSensorWithDetectionLogic {
    pub sun_sensor: SunSensor,
    pub irradiance_sensor: IrradianceSensor,
    pub light_detection_logic: LightDetectionSystem,
}

impl SunSensorWithDetectionLogic {
    pub fn new(
        allignment: UnitQuaternion<f64>,
        angle_noise_stddev: f64,
        irradiance_noise_stddev: f64,
        detectable_angle: f64,
        irradiance_threshold: f64,
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    ) -> Self {
        let sun_sensor = SunSensor::new(angle_noise_stddev, rng.clone(), allignment);

        let irradiance_sensor =
            IrradianceSensor::new(rng.clone(), irradiance_noise_stddev, allignment);
        let light_detection_logic =
            LightDetectionSystem::new(detectable_angle, irradiance_threshold);

        SunSensorWithDetectionLogic {
            sun_sensor,
            irradiance_sensor,
            light_detection_logic,
        }
    }
}

impl Sensor for SunSensorWithDetectionLogic {
    type IN = SunPositionAndIrradiance;
    type OUT = s5e_port::LightDetectionSystemData;

    fn sensor_tick(&mut self, input: Self::IN) {
        let sun_position = input.sun_position;
        let irradiance = input.irradiance;
        let irradiance_vector = BodyVector::from(
            Vector3::from(sun_position.clone())
                .try_normalize(MIN_NORMALIZE_THRESHOLD)
                .map(|x| x * irradiance.value)
                .unwrap_or_else(|| {
                    debug_sensor!(
                        "[SunSensorWithDetectionLogic] Cannot normalize sun position vector: {:?}",
                        sun_position
                    );
                    Vector3::zeros()
                }),
        );
        self.sun_sensor.sensor_tick(sun_position);
        self.irradiance_sensor.sensor_tick(irradiance_vector);

        s5e_port::transfer(
            &self.sun_sensor.port,
            &mut self.light_detection_logic.light_direction_port,
        );
        s5e_port::transfer(
            &self.irradiance_sensor.port,
            &mut self.light_detection_logic.irradiance_port,
        );

        self.light_detection_logic.sensor_tick(());
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::LightDetectionSystemData> {
        self.light_detection_logic.sensor_port()
    }

    fn sensor_clear(&mut self) {
        self.sun_sensor.sensor_clear();
        self.irradiance_sensor.sensor_clear();
        self.light_detection_logic.sensor_clear();
    }
}

pub struct TemperatureSensor {
    pub temperature: f64,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::TemperatureSensorData>,
}

impl TemperatureSensor {
    pub fn new(rng: Rc<RefCell<rand::prelude::ThreadRng>>, noise_std: f64) -> Self {
        TemperatureSensor {
            temperature: 0.0,
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
        }
    }

    fn measure(&mut self, temperature: f64) {
        let mut rng_ref = self.rng.borrow_mut();
        self.temperature = temperature + self.noise.sample(&mut *rng_ref);
        self.port.publish(s5e_port::TemperatureSensorData {
            temperature: self.temperature,
        });
    }
}

impl Sensor for TemperatureSensor {
    type IN = f64;
    type OUT = s5e_port::TemperatureSensorData;

    fn sensor_tick(&mut self, temperature: Self::IN) {
        self.measure(temperature);
        debug_sensor!("[TemperatureSensor] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::TemperatureSensorData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

pub struct Magnetometer {
    pub magnetic_field: ComponentVector,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::MagnetometerData>,
    pub alignment: UnitQuaternion<f64>,
}

impl Magnetometer {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: UnitQuaternion<f64>,
    ) -> Self {
        Magnetometer {
            magnetic_field: ComponentVector::new(),
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, magnetic_field: BodyVector) {
        let mut rng_ref = self.rng.borrow_mut();
        self.magnetic_field = ComponentVector::from_body(
            BodyVector::from(
                Vector3::from(magnetic_field)
                    + Vector3::new(
                        self.noise.sample(&mut *rng_ref),
                        self.noise.sample(&mut *rng_ref),
                        self.noise.sample(&mut *rng_ref),
                    ),
            ),
            self.alignment,
        );
        self.port.publish(s5e_port::MagnetometerData {
            magnetic_field: Vector3::from(self.magnetic_field.clone()),
        });
    }
}

impl Sensor for Magnetometer {
    type IN = BodyVector;
    type OUT = s5e_port::MagnetometerData;

    fn sensor_tick(&mut self, magnetic_field: Self::IN) {
        self.measure(magnetic_field);
        debug_sensor!("[Magnetometer] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::MagnetometerData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

const GYRO_BIAS: [f64; 3] = [0.00, -0.00, 0.000]; // [rad/s]

pub struct GyroSensor {
    pub angular_velocity: ComponentVector,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::GyroSensorData>,
    pub alignment: UnitQuaternion<f64>,
}

impl GyroSensor {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: UnitQuaternion<f64>,
    ) -> Self {
        GyroSensor {
            angular_velocity: ComponentVector::new(),
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, angular_velocity: BodyVector) {
        let mut rng_ref = self.rng.borrow_mut();
        self.angular_velocity = ComponentVector::from_body(
            BodyVector::from(
                Vector3::from(angular_velocity)
                    + Vector3::new(
                        self.noise.sample(&mut *rng_ref),
                        self.noise.sample(&mut *rng_ref),
                        self.noise.sample(&mut *rng_ref),
                    )
                    + Vector3::new(GYRO_BIAS[0], GYRO_BIAS[1], GYRO_BIAS[2]),
            ),
            self.alignment,
        );
        self.port.publish(s5e_port::GyroSensorData {
            angular_velocity: Vector3::from(self.angular_velocity.clone()),
        });
    }
}

impl Sensor for GyroSensor {
    type IN = BodyVector;
    type OUT = s5e_port::GyroSensorData;

    fn sensor_tick(&mut self, angular_velocity: Self::IN) {
        self.measure(angular_velocity);
        debug_sensor!("[GyroSensor] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::GyroSensorData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

pub struct Accelerometer {
    pub acceleration: ComponentVector,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::AccelerometerData>,
    pub alignment: UnitQuaternion<f64>,
}

impl Accelerometer {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: UnitQuaternion<f64>,
    ) -> Self {
        Accelerometer {
            acceleration: ComponentVector::new(),
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, acceleration: BodyVector) {
        let mut rng_ref = self.rng.borrow_mut();
        self.acceleration = ComponentVector::from_body(
            BodyVector::from(
                Vector3::from(acceleration)
                    + Vector3::new(
                        self.noise.sample(&mut *rng_ref),
                        self.noise.sample(&mut *rng_ref),
                        self.noise.sample(&mut *rng_ref),
                    ),
            ),
            self.alignment,
        );
        self.port.publish(s5e_port::AccelerometerData {
            acceleration: Vector3::from(self.acceleration.clone()),
        });
    }
}

impl Sensor for Accelerometer {
    type IN = BodyVector;
    type OUT = s5e_port::AccelerometerData;

    fn sensor_tick(&mut self, acceleration: Self::IN) {
        self.measure(acceleration);
        debug_sensor!("[Accelerometer] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::AccelerometerData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

pub struct StarTracker {
    pub attitude_quaternion: UnitQuaternion<f64>,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::StarTrackerData>,
    pub alignment: UnitQuaternion<f64>,
}

impl StarTracker {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: UnitQuaternion<f64>,
    ) -> Self {
        StarTracker {
            attitude_quaternion: UnitQuaternion::identity(),
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, attitude_quaternion: UnitQuaternion<f64>) {
        let mut rng_ref = self.rng.borrow_mut();
        let noise_quaternion = UnitQuaternion::new(Vector3::new(
            self.noise.sample(&mut *rng_ref),
            self.noise.sample(&mut *rng_ref),
            self.noise.sample(&mut *rng_ref),
        ));
        self.attitude_quaternion = noise_quaternion * self.alignment * attitude_quaternion;
        self.port.publish(s5e_port::StarTrackerData {
            attitude_quaternion: self.attitude_quaternion,
        });
    }
}

impl Sensor for StarTracker {
    type IN = UnitQuaternion<f64>;
    type OUT = s5e_port::StarTrackerData;

    fn sensor_tick(&mut self, attitude_quaternion: Self::IN) {
        self.measure(attitude_quaternion);
        debug_sensor!("[StarTracker] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::StarTrackerData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

#[derive(Clone)]
pub struct TimeSource {
    pub port: s5e_port::S5EPublishPort<s5e_port::TimeData>,
}

impl Default for TimeSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSource {
    pub fn new() -> Self {
        TimeSource {
            port: s5e_port::S5EPublishPort::new(),
        }
    }

    pub fn publish(&mut self, datetime: chrono::NaiveDateTime) {
        self.port.publish(s5e_port::TimeData {
            year: datetime.year(),
            month: datetime.month(),
            day: datetime.day(),
            hour: datetime.hour(),
            minute: datetime.minute(),
            second: datetime.second(),
            nanosecond: datetime.nanosecond(),
        });
    }
}

impl Sensor for TimeSource {
    type IN = chrono::NaiveDateTime;
    type OUT = s5e_port::TimeData;

    fn sensor_tick(&mut self, datetime: Self::IN) {
        self.publish(datetime);
        debug_sensor!("[TimeSource] After publish: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::TimeData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

#[derive(Clone)]
pub struct ECIPositionSensorInput {
    pub position: ECIPosition,
    pub attitude: UnitQuaternion<f64>,
}

pub struct ECIPositionSensor {
    pub position: ECIPosition,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::ECIComponentPositionData>,
    pub alignment: BodyVector,
}

impl ECIPositionSensor {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: BodyVector,
    ) -> Self {
        ECIPositionSensor {
            position: ECIPosition::new(),
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(&mut self, position: ECIPosition, attitude: UnitQuaternion<f64>) {
        let mut rng_ref = self.rng.borrow_mut();
        self.position = ECIPosition::from(
            Vector3::from(position.clone())
                + Vector3::new(
                    self.noise.sample(&mut *rng_ref),
                    self.noise.sample(&mut *rng_ref),
                    self.noise.sample(&mut *rng_ref),
                )
                + attitude.transform_vector(&Vector3::from(self.alignment.clone())),
        );
        self.port.publish(s5e_port::ECIComponentPositionData {
            component_position: Vector3::from(self.position.clone()),
        });
    }
}

impl Sensor for ECIPositionSensor {
    type IN = ECIPositionSensorInput;
    type OUT = s5e_port::ECIComponentPositionData;

    fn sensor_tick(&mut self, data: Self::IN) {
        self.measure(data.position, data.attitude);
        debug_sensor!("[ECIPositionSensor] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::ECIComponentPositionData> {
        &self.port
    }
    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

#[derive(Clone)]
pub struct ECIVelocitySensorInput {
    pub velocity: ECIVelocity,
    pub attitude: UnitQuaternion<f64>,
    pub angular_velocity: BodyVector,
}

pub struct ECIVelocitySensor {
    pub velocity: ECIVelocity,
    pub noise: Normal<f64>,
    pub rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    pub port: s5e_port::S5EPublishPort<s5e_port::ECIComponentVelocityData>,
    pub alignment: BodyVector,
}

impl ECIVelocitySensor {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        noise_std: f64,
        alignment: BodyVector,
    ) -> Self {
        ECIVelocitySensor {
            velocity: ECIVelocity {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            port: s5e_port::S5EPublishPort::new(),
            alignment,
        }
    }

    fn measure(
        &mut self,
        velocity: ECIVelocity,
        attitude: UnitQuaternion<f64>,
        angular_velocity: BodyVector,
    ) {
        let mut rng_ref = self.rng.borrow_mut();
        self.velocity = ECIVelocity::from(
            Vector3::from(velocity.clone())
                + Vector3::new(
                    self.noise.sample(&mut *rng_ref),
                    self.noise.sample(&mut *rng_ref),
                    self.noise.sample(&mut *rng_ref),
                )
                + attitude.transform_vector(
                    &Vector3::from(angular_velocity.clone())
                        .cross(&Vector3::from(self.alignment.clone())),
                ),
        );
        self.port.publish(s5e_port::ECIComponentVelocityData {
            component_velocity: Vector3::from(self.velocity.clone()),
        });
    }
}

impl Sensor for ECIVelocitySensor {
    type IN = ECIVelocitySensorInput;
    type OUT = s5e_port::ECIComponentVelocityData;

    fn sensor_tick(&mut self, data: Self::IN) {
        self.measure(data.velocity, data.attitude, data.angular_velocity);
        debug_sensor!("[ECIVelocitySensor] After measure: {:?}", self.port);
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::ECIComponentVelocityData> {
        &self.port
    }
    fn sensor_clear(&mut self) {
        self.port.clear();
    }
}

#[derive(Clone)]
pub struct ECIGnssSensorInput {
    pub position: ECIPosition,
    pub velocity: ECIVelocity,
    pub attitude: UnitQuaternion<f64>,
    pub angular_velocity: BodyVector,
    pub time: chrono::NaiveDateTime,
}

pub struct ECIGnssSensor {
    pub position_sensor: ECIPositionSensor,
    pub velocity_sensor: ECIVelocitySensor,
    pub timer: TimeSource,
    pub position_port: s5e_port::S5ESubscribePort<s5e_port::ECIComponentPositionData>,
    pub velocity_port: s5e_port::S5ESubscribePort<s5e_port::ECIComponentVelocityData>,
    pub time_port: s5e_port::S5ESubscribePort<s5e_port::TimeData>,
    pub port: s5e_port::S5EPublishPort<s5e_port::ECIGnssData>,
}

impl ECIGnssSensor {
    pub fn new(
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
        position_noise_std: f64,
        velocity_noise_std: f64,
        alignment: BodyVector,
    ) -> Self {
        let position_sensor =
            ECIPositionSensor::new(rng.clone(), position_noise_std, alignment.clone());
        let velocity_sensor = ECIVelocitySensor::new(rng.clone(), velocity_noise_std, alignment);
        ECIGnssSensor {
            position_sensor,
            velocity_sensor,
            timer: TimeSource::new(),
            position_port: s5e_port::S5ESubscribePort::new(),
            velocity_port: s5e_port::S5ESubscribePort::new(),
            time_port: s5e_port::S5ESubscribePort::new(),
            port: s5e_port::S5EPublishPort::new(),
        }
    }
}

impl Sensor for ECIGnssSensor {
    type IN = ECIGnssSensorInput;
    type OUT = s5e_port::ECIGnssData;

    fn sensor_tick(&mut self, data: Self::IN) {
        self.position_sensor.sensor_tick(ECIPositionSensorInput {
            position: data.position,
            attitude: data.attitude,
        });
        self.velocity_sensor.sensor_tick(ECIVelocitySensorInput {
            velocity: data.velocity,
            attitude: data.attitude,
            angular_velocity: data.angular_velocity,
        });
        self.timer.sensor_tick(data.time);

        s5e_port::transfer(&self.position_sensor.port, &mut self.position_port);
        s5e_port::transfer(&self.velocity_sensor.port, &mut self.velocity_port);
        s5e_port::transfer(&self.timer.port, &mut self.time_port);

        if let (Some(position_data), Some(velocity_data), Some(time_data)) = (
            self.position_port.subscribe(),
            self.velocity_port.subscribe(),
            self.time_port.subscribe(),
        ) {
            self.port.publish(s5e_port::ECIGnssData {
                component_position: Vector3::from(position_data.component_position),
                component_velocity: Vector3::from(velocity_data.component_velocity),
                time: time_data.clone(),
            });
        }
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<s5e_port::ECIGnssData> {
        &self.port
    }

    fn sensor_clear(&mut self) {
        self.position_sensor.sensor_clear();
        self.velocity_sensor.sensor_clear();
        self.timer.sensor_clear();
        self.port.clear();
    }
}
