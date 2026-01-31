use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Default)]
pub struct DebugConfig {
    pub sensor_data: bool,
    pub actuator_data: bool,
}

static DEBUG_CONFIG: OnceLock<DebugConfig> = OnceLock::new();

pub fn init_debug_config(config: DebugConfig) {
    DEBUG_CONFIG.set(config).ok();
}

pub fn get_debug_config() -> &'static DebugConfig {
    DEBUG_CONFIG.get().unwrap_or(&DEFAULT_CONFIG)
}

const DEFAULT_CONFIG: DebugConfig = DebugConfig {
    sensor_data: false,
    actuator_data: false,
};

#[macro_export]
macro_rules! debug_sensor {
    ($($arg:tt)*) => {
        if $crate::get_debug_config().sensor_data {
            let msg = format!($($arg)*);
            println!("[SENSOR] {}", msg);
        }
    };
}

#[macro_export]
macro_rules! debug_actuator {
    ($($arg:tt)*) => {
        if $crate::get_debug_config().actuator_data {
            let msg = format!($($arg)*);
            println!("[ACTUATOR] {}", msg);
        }
    };
}
