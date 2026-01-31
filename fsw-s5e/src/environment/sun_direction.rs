use astrodynamics::time::{Century, JulianDate};
use nalgebra::{UnitVector3, Vector3};

use crate::{
    constants::{
        ECLIPTIC_EPSILON_RAD, INERTIAL_SUN_STD, SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY,
        SUN_MEAN_ANOMALY_COEFF_RAD_CENTURY,
    },
    data,
};

pub struct InertialSunDirectionCalculation {
    inertial_sun_std: f64,
}

impl Default for InertialSunDirectionCalculation {
    fn default() -> Self {
        Self::new()
    }
}

impl InertialSunDirectionCalculation {
    pub fn new() -> Self {
        Self {
            inertial_sun_std: INERTIAL_SUN_STD,
        }
    }

    pub fn main_loop(&mut self, time: &crate::Time) -> Option<data::InertialSunDirectionData> {
        if let Some(absolute_time) = time.absolute.as_ref() {
            let julian_date = JulianDate::new(
                absolute_time.year,
                absolute_time.month,
                absolute_time.day,
                absolute_time.hour,
                absolute_time.minute,
                absolute_time.second,
                absolute_time.nanosecond,
            );

            let century = Century::from(&julian_date);

            let mean_anomaly = SUN_MEAN_ANOMALY_COEFF_RAD_CENTURY[0]
                + SUN_MEAN_ANOMALY_COEFF_RAD_CENTURY[1] * century.value;

            let ecliptic_longtitude = SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY[0]
                + SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY[1] * mean_anomaly.sin()
                + SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY[2] * (2.0 * mean_anomaly).sin()
                + SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY[3] * century.value
                + SUN_ECLIPTIC_LONGITUDE_COEFF_RAD_CENTURY[4]
                + mean_anomaly;

            let sun_direction_eci = UnitVector3::new_normalize(Vector3::new(
                ecliptic_longtitude.cos(),
                ECLIPTIC_EPSILON_RAD.cos() * ecliptic_longtitude.sin(),
                ECLIPTIC_EPSILON_RAD.sin() * ecliptic_longtitude.sin(),
            ));

            Some(data::InertialSunDirectionData {
                sun_direction_eci,
                std: self.inertial_sun_std,
            })
        } else {
            None
        }
    }
}
