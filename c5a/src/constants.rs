use nalgebra::{Matrix3, Vector3};

pub const DELTA_T: f64 = 0.01; // [s]

pub const SATELLITE_INERTIA: Matrix3<f64> = Matrix3::new(
            0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542,
            0.111208,
        ); // [kg*m^2]

pub const RW_INERTIA: Matrix3<f64> = Matrix3::new(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01); // [kg*m^2]

pub const GYRO_BIAS_DRIFT_STD: f64 = 0.1 * core::f64::consts::PI / 180.0 * 0.01; // [rad/s^2]
pub const MAG_FIELD_NORM_WALK_STD: f64 = 50.0; // [T/s]
pub const MAG_FIELD_DIR_WALK_STD: f64 = 0.01; // [rad/s]
pub const SUN_DIRECTION_WALK_STD: f64 = 0.001; // [rad/s]

// UKF parameters
// alpha: determines the spread of sigma points (0.001 ~ 1.0)
// beta: incorporates prior knowledge of distribution (2.0 is optimal for Gaussian)
// kappa: secondary scaling parameter (3-n or 0 is common, where n is state dimension)
pub const UKF_ALPHA: f64 = 1e-3;
pub const UKF_BETA: f64 = 2.0;
pub const UKF_KAPPA: f64 = 0.0;

pub const SUN_POINTING_TARGET_AXIS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);

pub const FROM_BDOT_ANGULAR_VELOCITY_THRESHOLD: f64 = 1.0 * core::f64::consts::PI / 180.0;
pub const BDOT_TRANSITION_TIMER_THRESHOLD: f64 = 10.0; // [s]
pub const TO_BDOT_ANGULAR_VELOCITY_THRESHOLD: f64 = 5.0e-1;

pub const MTQ_MAX_DIPOLE_MOMENT: f64 = 1.0e10; // [A*m^2]

pub const MTQ_P_GAIN_SCALE: f64 = 1.0e-5;
pub const MTQ_P_GAIN: Matrix3<f64> = Matrix3::new(
    MTQ_P_GAIN_SCALE,
    0.0,
    0.0,
    0.0,
    MTQ_P_GAIN_SCALE,
    0.0,
    0.0,
    0.0,
    MTQ_P_GAIN_SCALE,
);

pub const MTQ_D_GAIN_SCALE: f64 = 1.0e-2;
pub const MTQ_D_GAIN: Matrix3<f64> = Matrix3::new(
    MTQ_D_GAIN_SCALE,
    0.0,
    0.0,
    0.0,
    MTQ_D_GAIN_SCALE,
    0.0,
    0.0,
    0.0,
    MTQ_D_GAIN_SCALE,
);

pub const BDOT_DETUMBLING_GAIN_SCALE: f64 = 1.0e0;
pub const BDOT_DETUMBLING_GAIN: Matrix3<f64> = Matrix3::new(
    BDOT_DETUMBLING_GAIN_SCALE,
    0.0,
    0.0,
    0.0,
    BDOT_DETUMBLING_GAIN_SCALE,
    0.0,
    0.0,
    0.0,
    BDOT_DETUMBLING_GAIN_SCALE,
);

pub const RW_P_GAIN: Matrix3<f64> = Matrix3::new(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1);

pub const RW_D_GAIN: Matrix3<f64> = Matrix3::new(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5);

pub const UNLOADING_START_THRESHOLD: f64 = 5.0; // [Nms]
pub const UNLOADING_END_THRESHOLD: f64 = 1.0; // [Nms]

pub const UNLOADING_GAIN: Matrix3<f64> =
    Matrix3::new(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05);

pub const RW_MAX_ANGULAR_ACCELERATION: f64 = 10.0; // [rad/s^2]

pub const MAGNETOMETER_OBSERVATION_CYCLE: f64 = 1.0; // [s]
pub const MAGNETOMETER_DEMAGNETIZATION_TIME: f64 = 0.1; // [s]
pub const MAGNETOMETER_OBSERVATION_TIME_LIMIT: f64 = 0.5; // [s]
pub const MAGNETOMETER_NOISE_STD: f64 = 10.0; // [nT]

pub const GYRO_NOISE_STD: f64 = 0.1 * core::f64::consts::PI / 180.0; // [rad/s]
pub const GYRO_CUTOFF_HZ: f64 = 0.01; // [Hz]

pub const GNSS_POSITION_NOISE_STD: f64 = 10.0; // [m]
pub const GNSS_VELOCITY_NOISE_STD: f64 = 10.0; // [m/s]

pub const SUN_SENSOR_NOISE_STD: f64 = core::f64::consts::PI / 180.0; // [rad]

pub const STAR_TRACKER_NOISE_STD: f64 = 48.5 * 1e-5; // [rad]

pub const ECLIPTIC_EPSILON_RAD: f64 = 4.0909280e-01;
pub const SUN_MEAN_ANOMALY_COEFF_RAD_CENTURY: [f64; 2] = [6.2399989e+00, 6.2830186e+02];
pub const SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY: [f64; 5] = [
    4.9382346e+00,
    3.3413359e-02,
    3.4906585e-04,
    6.0606129e-03,
    -4.6286132e-05,
];

pub const INERTIAL_SUN_STD: f64 = 1e-3;
pub const IGRF_MAG_FIELD_STD: f64 = 100.0; // [nT]

pub const GME: f64 = 3.986004418e14; // [m^3/s^2] Earth's gravitational constant
