use core::f64;
use std::sync::Arc;

pub mod actuator;
pub mod earth;
pub mod moon;
pub mod orbit;
pub mod sensor;
pub mod spice_if;
pub mod sun;
pub mod sat;

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

pub trait SimInputTransfer<SensorOutputPorts> {
    fn transfer_from(&mut self, sensor_output: &SensorOutputPorts);
}

pub trait SimOutputTransfer<ActuatorInputPorts> {
    fn transfer_to(&self, actuator_input: &mut ActuatorInputPorts);
}
