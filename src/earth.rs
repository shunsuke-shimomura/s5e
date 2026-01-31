use chrono::{NaiveDateTime, TimeDelta};

use control_system::integrator::{
    Prediction,
    rk4::{RK4Input, RK4InputPrediction},
};

use astrodynamics::coordinate::{ECIPosition, ECIVelocity};

use crate::{
    spice_if::{self, Radii},
    sun::ShadowSource,
};

pub struct Earth {
    radii: Radii, // in meters
    pub position: RK4Input<ECIPosition, NaiveDateTime, f64>,
    pub velocity: RK4Input<ECIVelocity, NaiveDateTime, f64>,
}

impl Earth {
    pub fn new(datetime: NaiveDateTime) -> Self {
        let et = spice_if::datetime_to_et(datetime);
        let radii = spice_if::get_constant(
            spice_if::CelestialBody::Earth,
            spice_if::CelestialConstantsType::RADII,
        )
        .try_into()
        .expect("Failed to convert radii");
        let (position, velocity) = spice_if::get_state(et, spice_if::CelestialBody::Earth);
        Earth {
            radii,
            position: RK4Input::new(position),
            velocity: RK4Input::new(velocity),
        }
    }

    pub fn tick(&mut self, dt: f64, datetime: NaiveDateTime) {
        let after_half_dt = datetime + TimeDelta::nanoseconds((dt / 2.0 * 1.0e9) as i64);
        let et_half = spice_if::datetime_to_et(after_half_dt);
        let (after_half_dt_pos, after_half_dt_vel) =
            spice_if::get_state(et_half, spice_if::CelestialBody::Earth);
        let after_dt = datetime + TimeDelta::nanoseconds((dt * 1.0e9) as i64);
        let et_dt = spice_if::datetime_to_et(after_dt);
        let (after_dt_pos, after_dt_vel) =
            spice_if::get_state(et_dt, spice_if::CelestialBody::Earth);
        self.position.set(RK4InputPrediction {
            after_halfdt: after_half_dt_pos,
            after_dt: after_dt_pos,
            dt,
            time: datetime,
        });
        self.velocity.set(RK4InputPrediction {
            after_halfdt: after_half_dt_vel,
            after_dt: after_dt_vel,
            dt,
            time: datetime,
        });
    }

    pub fn clear_state(&mut self) {
        self.position.clear();
        self.velocity.clear();
    }

    pub fn shadow_source(&self) -> ShadowSource {
        ShadowSource {
            position: self.position.get_now(),
            radius: self.radii.mean_radius(),
        }
    }
}
