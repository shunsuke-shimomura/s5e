use core::ops::{Add, Div, Mul, Sub};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

#[cfg(feature = "std")]
use std::{cell::RefCell, rc::Rc};

#[cfg(feature = "std")]
use rand_distr::{Distribution, Normal, Uniform};

#[cfg(not(feature = "std"))]
use num_traits::float::Float;

use crate::time::JulianDate;

pub const MIN_NORMALIZE_THRESHOLD: f64 = 1e-10;

// Todo: 歳差運動・章動運動等の考慮, 定数項の出典の調査
#[allow(non_snake_case)]
fn get_dcm_J2000_to_ECEF(julian_date: JulianDate) -> Matrix3<f64> {
    let tut1 = (julian_date.value - 2451545.0) / 36525.0;
    let theta = -6.2e-6 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 67310.54841;
    let gmst_rad = (theta * (360.0 / 86400.0) * (core::f64::consts::PI / 180.0))
        % (2.0 * core::f64::consts::PI);
    Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        gmst_rad.cos(),
        gmst_rad.sin(),
        0.0,
        -gmst_rad.sin(),
        gmst_rad.cos(),
    )
}

#[derive(Debug, Clone)]
pub struct BodyVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<BodyVector> for Vector3<f64> {
    fn from(v: BodyVector) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for BodyVector {
    fn from(v: Vector3<f64>) -> Self {
        BodyVector {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl Add for BodyVector {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        BodyVector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Mul<f64> for BodyVector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        BodyVector {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Div<f64> for BodyVector {
    type Output = Self;

    fn div(self, scalar: f64) -> Self::Output {
        BodyVector {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl Default for BodyVector {
    fn default() -> Self {
        Self::new()
    }
}

impl BodyVector {
    pub fn new() -> Self {
        BodyVector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn from_eci(v: ECIVector, attitude: UnitQuaternion<f64>) -> Self {
        let ret = attitude.transform_vector(&Vector3::from(v));
        BodyVector {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }

    pub fn from_component(v: ComponentVector, alignment: UnitQuaternion<f64>) -> Self {
        let ret = alignment.inverse_transform_vector(&Vector3::from(v));
        BodyVector {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComponentVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for ComponentVector {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentVector {
    pub fn new() -> Self {
        ComponentVector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl From<ComponentVector> for Vector3<f64> {
    fn from(v: ComponentVector) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ComponentVector {
    fn from(v: Vector3<f64>) -> Self {
        ComponentVector {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl ComponentVector {
    // alignment: UnitQuaternionで、Body -> Componentの変換を表す
    pub fn from_body(v: BodyVector, alignment: UnitQuaternion<f64>) -> Self {
        let ret = alignment.transform_vector(&Vector3::from(v));
        ComponentVector {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }
}

#[cfg(feature = "std")]
pub struct DirectionNose {
    theta: Uniform<f64>,
    phi: Uniform<f64>,
    angle: Normal<f64>,
    rng: Rc<RefCell<rand::prelude::ThreadRng>>,
}

#[cfg(feature = "std")]
impl DirectionNose {
    pub fn new(angle: f64, rng: Rc<RefCell<rand::prelude::ThreadRng>>) -> Self {
        DirectionNose {
            theta: Uniform::new(0.0, core::f64::consts::PI),
            phi: Uniform::new(0.0, 2.0 * core::f64::consts::PI),
            angle: Normal::new(0.0, angle).unwrap(),
            rng: Rc::clone(&rng),
        }
    }

    pub fn sample(&self) -> UnitQuaternion<f64> {
        let mut rng_ref = self.rng.borrow_mut();
        let theta = self.theta.sample(&mut *rng_ref);
        let phi = self.phi.sample(&mut *rng_ref);
        let angle = self.angle.sample(&mut *rng_ref);

        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();

        UnitQuaternion::from_quaternion(Quaternion::from_parts(
            (angle / 2.0).cos(),
            Vector3::new(
                x * (angle / 2.0).sin(),
                y * (angle / 2.0).sin(),
                z * (angle / 2.0).sin(),
            ),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct ComponentDirection {
    x: f64,
    y: f64,
    z: f64,
}

impl Default for ComponentDirection {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentDirection {
    pub fn new() -> Self {
        ComponentDirection {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    pub fn x(&self) -> f64 {
        self.x
    }
    pub fn y(&self) -> f64 {
        self.y
    }
    pub fn z(&self) -> f64 {
        self.z
    }

    #[cfg(feature = "std")]
    pub fn add_noise(&self, noise: &DirectionNose) -> Self {
        let quaternion = noise.sample();
        let vec = Vector3::new(self.x, self.y, self.z);
        let rotated_vec = quaternion.transform_vector(&vec);
        ComponentDirection {
            x: rotated_vec.x,
            y: rotated_vec.y,
            z: rotated_vec.z,
        }
    }
}

impl TryFrom<ComponentVector> for ComponentDirection {
    type Error = anyhow::Error;
    fn try_from(v: ComponentVector) -> Result<Self, Self::Error> {
        let vec = Vector3::from(v);
        let direction = vec
            .try_normalize(MIN_NORMALIZE_THRESHOLD)
            .ok_or(anyhow::anyhow!(
                "Cannot normalize ComponentVector to ComponentDirection"
            ))?;
        Ok(ComponentDirection {
            x: direction.x,
            y: direction.y,
            z: direction.z,
        })
    }
}

impl From<ComponentDirection> for Vector3<f64> {
    fn from(v: ComponentDirection) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

#[derive(Debug, Clone)]
pub struct ECIPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ECIPosition {
    pub fn new() -> Self {
        ECIPosition {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Default for ECIPosition {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ECIPosition> for Vector3<f64> {
    fn from(v: ECIPosition) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ECIPosition {
    fn from(v: Vector3<f64>) -> Self {
        ECIPosition {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl Sub for ECIPosition {
    type Output = ECIVector;

    fn sub(self, other: Self) -> Self::Output {
        ECIVector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ECIVelocity {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<ECIVelocity> for Vector3<f64> {
    fn from(v: ECIVelocity) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ECIVelocity {
    fn from(v: Vector3<f64>) -> Self {
        ECIVelocity {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ECIVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ECIVector {
    pub fn new() -> Self {
        ECIVector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Default for ECIVector {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ECIVector> for Vector3<f64> {
    fn from(v: ECIVector) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ECIVector {
    fn from(v: Vector3<f64>) -> Self {
        ECIVector {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl ECIVector {
    pub fn from_ecef(v: ECEFVector, julian_date: JulianDate) -> Self {
        let ret = get_dcm_J2000_to_ECEF(julian_date).try_inverse().unwrap() * Vector3::from(v);
        ECIVector {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }

    pub fn from_body(v: BodyVector, attitude: UnitQuaternion<f64>) -> Self {
        let ret = attitude.inverse_transform_vector(&Vector3::from(v));
        ECIVector {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ECEFPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ECEFPosition {
    pub fn from_eci(v: ECIPosition, julian_date: JulianDate) -> Self {
        let ret = get_dcm_J2000_to_ECEF(julian_date) * Vector3::from(v);
        ECEFPosition {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }
}

impl From<ECEFPosition> for Vector3<f64> {
    fn from(v: ECEFPosition) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ECEFPosition {
    fn from(v: Vector3<f64>) -> Self {
        ECEFPosition {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ECEFVelocity {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ECEFVelocity {
    pub fn from_eci(v: ECIVelocity, x: ECIPosition, julian_date: JulianDate) -> Self {
        let omega = 7.292115146706979e-5;
        let w_exr = Vector3::new(0.0, 0.0, omega).cross(&Vector3::from(x));
        let ret = get_dcm_J2000_to_ECEF(julian_date) * (Vector3::from(v) - w_exr);
        ECEFVelocity {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }
}

impl From<ECEFVelocity> for Vector3<f64> {
    fn from(v: ECEFVelocity) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ECEFVelocity {
    fn from(v: Vector3<f64>) -> Self {
        ECEFVelocity {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ECEFVector {
    x: f64,
    y: f64,
    z: f64,
}

impl From<ECEFVector> for Vector3<f64> {
    fn from(v: ECEFVector) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for ECEFVector {
    fn from(v: Vector3<f64>) -> Self {
        ECEFVector {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl ECEFVector {
    pub fn from_geodetic(v: GeodeticVector, pos: GeodeticPosition) -> Self {
        let lat_rad = pos.latitude * core::f64::consts::PI / 180.0;
        let lon_rad = pos.longitude * core::f64::consts::PI / 180.0;
        let a = 6378137.0;
        let f = 3.352797e-3;
        let rp = a * (1.0 - f);
        let rm2 = a * a * lat_rad.cos() * lat_rad.cos() + rp * rp * lat_rad.sin() * lat_rad.sin();
        let rm = rm2.sqrt();
        let rrm = (a * a * a * a * lat_rad.cos() * lat_rad.cos()
            + rp * rp * rp * rp * lat_rad.sin() * lat_rad.sin())
            / rm2;
        let r = (rrm + 2.0 * pos.altitude * rm + pos.altitude * pos.altitude).sqrt();
        let cth = lat_rad.sin() * (pos.altitude + rp * rp / rm) / r;
        let theta = core::f64::consts::PI - cth.acos();

        let rot1 = Matrix3::new(
            theta.cos(),
            0.0,
            -theta.sin(),
            0.0,
            1.0,
            0.0,
            theta.sin(),
            0.0,
            theta.cos(),
        );

        let rot2 = Matrix3::new(
            lon_rad.cos(),
            -lon_rad.sin(),
            0.0,
            lon_rad.sin(),
            lon_rad.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );

        let ret = rot2 * rot1 * Vector3::from(v);
        ECEFVector {
            x: ret.x,
            y: ret.y,
            z: ret.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeodeticPosition {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}

// Todo: アルゴリズムの検証をする（現状S2Eのコピー）
impl From<ECEFPosition> for GeodeticPosition {
    fn from(value: ECEFPosition) -> Self {
        let a = 6378137.0;
        let f = 3.352797e-3;
        let theta = value.y.atan2(value.x);
        let e2 = f * (2.0 - f);
        let r = (value.x * value.x + value.y * value.y).sqrt();
        let mut lat_tmp_rad = value.z.atan2(r);
        while {
            let lat_tmp_rad_old = lat_tmp_rad;
            let c = 1.0 / (1.0 - e2 * lat_tmp_rad.sin() * lat_tmp_rad.sin()).sqrt();
            lat_tmp_rad = (value.z + a * c * e2 * lat_tmp_rad.sin()).atan2(r);
            (lat_tmp_rad - lat_tmp_rad_old).abs() > 1.0e-10
        } {}
        let c = 1.0 / (1.0 - e2 * lat_tmp_rad.sin() * lat_tmp_rad.sin()).sqrt();
        let latitude = lat_tmp_rad * 180.0 / core::f64::consts::PI;
        let longitude = (theta % (2.0 * core::f64::consts::PI)) * 180.0 / core::f64::consts::PI;
        let altitude = r / lat_tmp_rad.cos() - a * c;
        GeodeticPosition {
            latitude,
            longitude,
            altitude,
        }
    }
}

impl From<GeodeticPosition> for Vector3<f64> {
    fn from(v: GeodeticPosition) -> Self {
        Vector3::new(v.latitude, v.longitude, v.altitude)
    }
}

// latitude: degree, longitude: degree, altitude: m
impl From<Vector3<f64>> for GeodeticPosition {
    fn from(v: Vector3<f64>) -> Self {
        GeodeticPosition {
            latitude: v.x,
            longitude: v.y,
            altitude: v.z,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeodeticVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<GeodeticVector> for Vector3<f64> {
    fn from(v: GeodeticVector) -> Self {
        Vector3::new(v.x, v.y, v.z)
    }
}

impl From<Vector3<f64>> for GeodeticVector {
    fn from(v: Vector3<f64>) -> Self {
        GeodeticVector {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}
