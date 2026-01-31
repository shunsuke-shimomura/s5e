use std::ops::Mul;

use chrono::{Datelike, NaiveDateTime, TimeDelta, Timelike};
use nalgebra::Vector3;

use astrodynamics::coordinate::{ECIPosition, ECIVelocity};

use crate::spice_if::{self, CelestialConstantsType, EphemerisTime, Radii};
use control_system::integrator::{
    Prediction,
    rk4::{RK4Input, RK4InputPrediction},
};

pub const SOLAR_LUMINOSITY_W: Luminosity = Luminosity { value: 3.828e26 }; // Solar luminosity in watts

pub const SOLAR_CONSTANT: f64 = 1366.0; // W/m^2, average solar constant

#[derive(Debug, Clone)]
pub struct Luminosity {
    pub value: f64, // luminosity in watts
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Irradiance {
    pub value: f64, // irradiance in watts per square meter
}

pub struct Sun {
    radii: Radii, // in meters
    solar_luminosity: Luminosity,
    pub position: RK4Input<ECIPosition, NaiveDateTime, f64>,
    pub velocity: RK4Input<ECIVelocity, NaiveDateTime, f64>,
}

impl Sun {
    pub fn new(epoch_datetime: NaiveDateTime) -> Self {
        let et = spice_if::datetime_to_et(epoch_datetime);
        let radii =
            spice_if::get_constant(spice_if::CelestialBody::Sun, CelestialConstantsType::RADII);
        let (position, velocity) = spice_if::get_state(et, spice_if::CelestialBody::Sun);
        Sun {
            radii: radii.try_into().expect("Failed to convert radii"),
            solar_luminosity: SOLAR_LUMINOSITY_W,
            position: RK4Input::new(position),
            velocity: RK4Input::new(velocity),
        }
    }

    pub fn tick(&mut self, dt: f64, datetime: NaiveDateTime) {
        let after_half_dt = datetime + TimeDelta::nanoseconds((dt / 2.0 * 1.0e9) as i64);
        let et_half = spice_if::datetime_to_et(after_half_dt);
        let (after_half_dt_pos, after_half_dt_vel) =
            spice_if::get_state(et_half, spice_if::CelestialBody::Sun);
        let after_dt = datetime + TimeDelta::nanoseconds((dt * 1.0e9) as i64);
        let et_dt = spice_if::datetime_to_et(after_dt);
        let (after_dt_pos, after_dt_vel) = spice_if::get_state(et_dt, spice_if::CelestialBody::Sun);
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

    pub fn light_source(&self) -> LightSource {
        LightSource {
            position: self.position.get_now(),
            radius: self.radii.mean_radius(),
            luminosity: self.solar_luminosity.clone(),
        }
    }
}

pub fn irradiance(light_source: LightSource, observation_position: ECIPosition) -> Irradiance {
    let distance =
        (Vector3::from(light_source.position.clone()) - Vector3::from(observation_position)).norm();
    if distance < 1e-10 {
        // Avoid division by zero
        return Irradiance { value: 0.0 };
    }
    Irradiance {
        value: light_source.luminosity.value / (4.0 * std::f64::consts::PI * distance * distance),
    }
}

#[derive(Debug, Clone)]
pub struct ShadowCoefficient {
    pub value: f64,
}

impl Mul<Irradiance> for ShadowCoefficient {
    type Output = Irradiance;

    fn mul(self, other: Irradiance) -> Self::Output {
        Irradiance {
            value: self.value * other.value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LightSource {
    pub position: ECIPosition,
    pub radius: f64, // m
    pub luminosity: Luminosity,
}

#[derive(Debug, Clone)]
pub struct ShadowSource {
    pub position: ECIPosition,
    pub radius: f64, // m
}

// S2Eから持ってきた関数。正しさは検証できていない
pub fn shadow_coefficient(
    light_source: LightSource,
    shadow_sources: Vec<ShadowSource>,
    observation_position: ECIPosition,
) -> ShadowCoefficient {
    // Constants
    let light_radius_m = light_source.radius; // Radius of the light in meters

    // Vector from observer to light (calculated once)
    let light_pos = light_source.position;
    let r_observation2light = nalgebra::Vector3::new(
        light_pos.x - observation_position.x,
        light_pos.y - observation_position.y,
        light_pos.z - observation_position.z,
    );
    let distance_observation_to_light = r_observation2light.norm();

    // Angular radius of light as seen from observer (calculated once)
    let light_angular_radius = (light_radius_m / distance_observation_to_light).asin();

    // Calculate shadow effect from each shadow source using fold
    let shadow_coefficient =
        shadow_sources
            .iter()
            .fold(1.0, |accumulated_coefficient, shadow_source| {
                // Vector from observer to shadow source
                let r_sc2source = nalgebra::Vector3::new(
                    shadow_source.position.x - observation_position.x,
                    shadow_source.position.y - observation_position.y,
                    shadow_source.position.z - observation_position.z,
                );
                let distance_sat_to_source = r_sc2source.norm();

                // Angular radius of shadow source as seen from observer
                let source_angular_radius = (shadow_source.radius / distance_sat_to_source).asin();

                // Angular distance between light center and shadow source center
                let cos_delta = r_observation2light.dot(&r_sc2source)
                    / (distance_observation_to_light * distance_sat_to_source);
                // Clamp to valid range for acos
                let cos_delta_clamped = cos_delta.clamp(-1.0, 1.0);
                let angular_distance = cos_delta_clamped.acos();

                // Calculate shadow coefficient based on geometric conditions
                let single_shadow_coefficient = if angular_distance
                    < (light_angular_radius - source_angular_radius).abs()
                    && light_angular_radius <= source_angular_radius
                {
                    // Case 1: Total eclipse (umbra) - shadow source completely covers the light source
                    0.0
                } else if angular_distance < (light_angular_radius - source_angular_radius).abs()
                    && light_angular_radius > source_angular_radius
                {
                    // Case 2: Maximum partial eclipse - shadow source is completely inside light source
                    1.0 - (source_angular_radius * source_angular_radius)
                        / (light_angular_radius * light_angular_radius)
                } else if (light_angular_radius - source_angular_radius).abs() <= angular_distance
                    && angular_distance <= (light_angular_radius + source_angular_radius)
                {
                    // Case 3: Partial eclipse (penumbra) - partial overlap
                    // Complex area calculation for overlapping circles

                    // Avoid division by zero
                    if angular_distance < 1e-10 {
                        // When centers are nearly coincident, use simple area ratio
                        let smaller_radius = light_angular_radius.min(source_angular_radius);
                        let larger_radius = light_angular_radius.max(source_angular_radius);
                        1.0 - (smaller_radius * smaller_radius) / (larger_radius * larger_radius)
                    } else {
                        let x = (light_angular_radius * light_angular_radius
                            - source_angular_radius * source_angular_radius
                            + angular_distance * angular_distance)
                            / (2.0 * angular_distance);

                        // Ensure the value under square root is non-negative
                        let discriminant = (light_angular_radius + source_angular_radius
                            - angular_distance)
                            * (angular_distance + light_angular_radius - source_angular_radius)
                            * (angular_distance - light_angular_radius + source_angular_radius)
                            * (angular_distance + light_angular_radius + source_angular_radius);

                        if discriminant < 0.0 {
                            // Degenerate case, use approximation
                            1.0
                        } else {
                            let y = discriminant.sqrt() / (2.0 * angular_distance);

                            // Clamp arguments to acos to valid range
                            let acos_arg1 = (x / light_angular_radius).clamp(-1.0, 1.0);
                            let acos_arg2 =
                                ((angular_distance - x) / source_angular_radius).clamp(-1.0, 1.0);

                            let area_overlap = light_angular_radius
                                * light_angular_radius
                                * acos_arg1.acos()
                                + source_angular_radius * source_angular_radius * acos_arg2.acos()
                                - angular_distance * y;
                            let area_light =
                                std::f64::consts::PI * light_angular_radius * light_angular_radius;

                            1.0 - area_overlap / area_light
                        }
                    }
                } else {
                    // Case 4: No eclipse - no overlap
                    1.0
                };

                // Multiply shadow coefficients (for multiple shadow sources)
                accumulated_coefficient * single_shadow_coefficient
            });

    ShadowCoefficient {
        value: shadow_coefficient,
    }
}

pub fn s2e_sun(datetime: NaiveDateTime) -> (ECIPosition, ECIVelocity) {
    // ユリウス日 (JD, UT) を計算
    let jd = jday(
        datetime.year(),
        datetime.month(),
        datetime.day(),
        datetime.hour(),
        datetime.minute(),
        datetime.second() as f64 + datetime.nanosecond() as f64 * 1e-9,
    );

    // S2E での太陽の位置と速度を取得
    let ephemeris_time = spice::str2et(&format!("jd {:.11}", jd));
    let (position, velocity) = spice_if::get_state(
        EphemerisTime {
            value: ephemeris_time,
        },
        spice_if::CelestialBody::Sun,
    );

    // ECIPosition と ECIVelocity に変換して返す
    (position, velocity)
}

// S2E でのユリウス日 (JD, UT) の計算
/// 計算対象の日付・時刻をユリウス日 (JD, UT) で返す
pub fn jday(year: i32, mon: u32, day: u32, hr: u32, minute: u32, sec: f64) -> f64 {
    let y = year as f64;
    let m = mon as f64;
    let d = day as f64;
    let h = hr as f64;
    let min = minute as f64;
    let s = sec;

    // C++ と同じ式を段階的に分解して計算
    let a = ((m + 9.0) / 12.0).floor(); // floor((mon+9)/12)
    let b = ((7.0 * (y + a)) * 0.25).floor(); // floor((7*(year+a))*0.25)
    let c = ((275.0 * m) / 9.0).floor(); // floor(275*mon/9)

    367.0 * y - b + c + d + 1721013.5 + (((s / 60.0 + min) / 60.0 + h) / 24.0) // 時刻成分を日単位へ
}
