use crate::data;

pub struct Timer {
    time: crate::Time,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Self {
        Timer {
            time: crate::Time::new(),
        }
    }

    pub fn main_loop(&mut self, dt: f64, input: Option<data::TimeSourceData>) -> crate::Time {
        self.time += dt;
        if let Some(time_data) = input {
            self.time.absolute = Some(time_data.datetime);
        }
        self.time
    }
}
