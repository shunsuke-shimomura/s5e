pub const MAGNETOMETER_NOISE_STD: f64 = fsw_s5e::constants::MAGNETOMETER_NOISE_STD; // nT
pub const SUN_SENSOR_DIR_NOISE_STD: f64 = fsw_s5e::constants::SUN_SENSOR_NOISE_STD; // rad
pub const GYRO_NOISE_STD: f64 = fsw_s5e::constants::GYRO_NOISE_STD; // rad/s
pub const GYRO_BIAS_DRIFT_STD: f64 = fsw_s5e::constants::GYRO_BIAS_DRIFT_STD; // rad/sqrt(s)
pub const STAR_TRACKER_NOISE_STD: f64 = fsw_s5e::constants::STAR_TRACKER_NOISE_STD; // rad
pub const MTQ_MAX_DIPOLE_MOMENT: f64 = fsw_s5e::constants::MTQ_MAX_DIPOLE_MOMENT; // A*m^2
pub const SATELLITE_INERTIA: nalgebra::Matrix3<f64> = fsw_s5e::constants::SATELLITE_INERTIA; // kg*m^2
