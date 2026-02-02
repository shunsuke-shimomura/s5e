pub struct SPIdealFsw {
    pub time: fsw_s5e::Time,
    timer: fsw_s5e::estimation::Timer,

    pub gyro_driver: fsw_s5e::driver::sensor::GyroDriver,
    pub magnetometer_driver: fsw_s5e::driver::sensor::MagnetometerDriver,
    pub ss_pz_driver: fsw_s5e::driver::sensor::SunSensorDriver,
    pub ss_py_driver: fsw_s5e::driver::sensor::SunSensorDriver,
    pub ss_mz_driver: fsw_s5e::driver::sensor::SunSensorDriver,
    pub ss_my_driver: fsw_s5e::driver::sensor::SunSensorDriver,

    pub pointing_controller: fsw_s5e::controller::SunPointingController,
}