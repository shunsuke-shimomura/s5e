use anyhow::Result;
use chrono::{NaiveDateTime, TimeDelta};
use nalgebra::Vector3;
use sgp4::{Constants, Elements};

use astrodynamics::coordinate::{ECIPosition, ECIVelocity};

use control_system::integrator::rk4::RK4InputPrediction;

#[derive(Debug, Clone)]
pub struct TLE {
    pub line1: String,
    pub line2: String,
}

impl TLE {
    pub fn new(line1: String, line2: String) -> Self {
        Self { line1, line2 }
    }
}

pub struct Trajectory {
    constants: Constants,
    elements: Elements,
}

impl Trajectory {
    pub fn new(elements: Elements) -> Result<Self> {
        Ok(Trajectory {
            constants: Constants::from_elements(&elements)?,
            elements,
        })
    }

    pub fn initial_kinematics(&self, datetime: NaiveDateTime) -> (ECIPosition, ECIVelocity) {
        let duration = self
            .elements
            .datetime_to_minutes_since_epoch(&datetime)
            .expect("nanosec overflow");
        let now = self.constants.propagate(duration).unwrap();
        (
            ECIPosition::from(Vector3::new(
                now.position[0] * 1.0e3,
                now.position[1] * 1.0e3,
                now.position[2] * 1.0e3,
            )),
            ECIVelocity::from(Vector3::new(
                now.velocity[0] * 1.0e3,
                now.velocity[1] * 1.0e3,
                now.velocity[2] * 1.0e3,
            )),
        )
    }

    pub fn propagate(
        &self,
        datetime: NaiveDateTime,
        dt: f64,
    ) -> (
        RK4InputPrediction<ECIPosition, NaiveDateTime, f64>,
        RK4InputPrediction<ECIVelocity, NaiveDateTime, f64>,
    ) {
        let duration = self
            .elements
            .datetime_to_minutes_since_epoch(
                &(datetime + TimeDelta::nanoseconds((dt / 2.0 * 1.0e9) as i64)),
            )
            .expect("nanosec overflow");
        let after_halfdt = self.constants.propagate(duration).unwrap();

        let duration = self
            .elements
            .datetime_to_minutes_since_epoch(
                &(datetime + TimeDelta::nanoseconds((dt * 1.0e9) as i64)),
            )
            .expect("nanosec overflow");
        let after_dt = self.constants.propagate(duration).unwrap();

        (
            RK4InputPrediction {
                after_halfdt: ECIPosition::from(Vector3::new(
                    after_halfdt.position[0] * 1.0e3,
                    after_halfdt.position[1] * 1.0e3,
                    after_halfdt.position[2] * 1.0e3,
                )),
                after_dt: ECIPosition::from(Vector3::new(
                    after_dt.position[0] * 1.0e3,
                    after_dt.position[1] * 1.0e3,
                    after_dt.position[2] * 1.0e3,
                )),
                dt,
                time: datetime,
            },
            RK4InputPrediction {
                after_halfdt: ECIVelocity::from(Vector3::new(
                    after_halfdt.velocity[0] * 1.0e3,
                    after_halfdt.velocity[1] * 1.0e3,
                    after_halfdt.velocity[2] * 1.0e3,
                )),
                after_dt: ECIVelocity::from(Vector3::new(
                    after_dt.velocity[0] * 1.0e3,
                    after_dt.velocity[1] * 1.0e3,
                    after_dt.velocity[2] * 1.0e3,
                )),
                dt,
                time: datetime,
            },
        )
    }
}
