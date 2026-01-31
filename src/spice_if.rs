use std::{fmt::Display, path::Path};

use chrono::NaiveDateTime;
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};
use std::sync::Once;

use astrodynamics::coordinate::{ECEFPosition, ECEFVelocity, ECIPosition, ECIVelocity};

// bodn2c, bodvcd, spkezr,
// sxformは月の座標系と地球座標系の変換

const ECI_COORDINATE_SYSTEM: &str = "J2000";
const ECEF_COORDINATE_SYSTEM: &str = "ITRF93"; // ITRF93 is the latest ITRF standard

static INIT: Once = Once::new();

fn ensure_spice_ready() {
    INIT.call_once(|| {
        println!("Loading SPICE kernels...");
        let kernel_dir = env!("SPICE_KERNEL_DIR");
        println!("Kernel directory: {}", kernel_dir);
        let latest_leapseconds_path = Path::new(kernel_dir).join("latest_leapseconds.tls");
        let de403_masses_path = Path::new(kernel_dir).join("de-403-masses.tpc");
        let gm_de440_path = Path::new(kernel_dir).join("gm_de440.tpc");
        let pck00011_path = Path::new(kernel_dir).join("pck00011.tpc");
        let de442s_path = Path::new(kernel_dir).join("de442s.bsp");
        let earth_latest_high_prec_path = Path::new(kernel_dir).join("earth_latest_high_prec.bsp");
        let _teme_tf_path = Path::new(kernel_dir).join("teme.tf");
        spice::furnsh(latest_leapseconds_path.to_str().unwrap());
        println!("  Loaded latest_leapseconds.tls (Leap Seconds Kernel)");
        spice::furnsh(de403_masses_path.to_str().unwrap());
        println!("  Loaded de-403-masses.tpc (Planetary Constants Kernel)");
        spice::furnsh(gm_de440_path.to_str().unwrap());
        println!("  Loaded gm_de440.tpc (Planetary Constants Kernel)");
        spice::furnsh(pck00011_path.to_str().unwrap());
        println!("  Loaded pck00011.tpc (Planetary Constants Kernel)");
        spice::furnsh(de442s_path.to_str().unwrap());
        println!("  Loaded de442s.bsp (Spacecraft and Planet Kernel)");
        spice::furnsh(earth_latest_high_prec_path.to_str().unwrap());
        println!("  Loaded earth_latest_high_prec.bsp (earth orientation kernel)");
        // ToolkitのUpdateが必要
        // spice::furnsh(teme_tf_path.to_str().unwrap());
        // println!("  Loaded teme.tf (Transformation Kernel)");
    });
}

#[derive(Debug, Clone, Copy)]
pub struct EphemerisTime {
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelestialBody {
    Earth,
    Sun,
    Moon,
    Mercury,
    Venus,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
}

impl Display for CelestialBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CelestialBody::Earth => write!(f, "EARTH"),
            CelestialBody::Sun => write!(f, "SUN"),
            CelestialBody::Moon => write!(f, "MOON"),
            CelestialBody::Mercury => write!(f, "MERCURY"),
            CelestialBody::Venus => write!(f, "VENUS"),
            CelestialBody::Mars => write!(f, "MARS"),
            CelestialBody::Jupiter => write!(f, "JUPITER"),
            CelestialBody::Saturn => write!(f, "SATURN"),
            CelestialBody::Uranus => write!(f, "URANUS"),
            CelestialBody::Neptune => write!(f, "NEPTUNE"),
            CelestialBody::Pluto => write!(f, "PLUTO"),
        }
    }
}

impl CelestialBody {
    pub fn spkezr_string(&self) -> String {
        match self {
            CelestialBody::Earth => "EARTH_BARYCENTER".to_string(),
            CelestialBody::Sun => "SUN".to_string(),
            CelestialBody::Moon => "MOON".to_string(),
            CelestialBody::Mercury => "MERCURY".to_string(),
            CelestialBody::Venus => "VENUS".to_string(),
            CelestialBody::Mars => "MARS_BARYCENTER".to_string(),
            CelestialBody::Jupiter => "JUPITER_BARYCENTER".to_string(),
            CelestialBody::Saturn => "SATURN_BARYCENTER".to_string(),
            CelestialBody::Uranus => "URANUS_BARYCENTER".to_string(),
            CelestialBody::Neptune => "NEPTUNE_BARYCENTER".to_string(),
            CelestialBody::Pluto => "PLUTO_BARYCENTER".to_string(),
        }
    }
}

pub enum CelestialConstantsType {
    GM,
    RADII,
}

impl Display for CelestialConstantsType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CelestialConstantsType::GM => write!(f, "GM"),
            CelestialConstantsType::RADII => write!(f, "RADII"),
        }
    }
}

pub enum CelestialConstants {
    GM(f64),
    Radii(Radii),
}

impl TryFrom<CelestialConstants> for Radii {
    type Error = anyhow::Error;

    fn try_from(value: CelestialConstants) -> Result<Self, Self::Error> {
        match value {
            CelestialConstants::Radii(radii) => Ok(radii),
            _ => Err(anyhow::anyhow!("Cannot convert GM to Radii")),
        }
    }
}

pub struct Radii {
    pub equatorial_semi_major_axis: f64, // m
    pub equatorial_semi_minor_axis: f64, // m
    pub polar_semi_axis: f64,            // m
}

impl Radii {
    pub fn mean_radius(&self) -> f64 {
        // 平均半径を計算
        (self.equatorial_semi_major_axis + self.equatorial_semi_minor_axis + self.polar_semi_axis)
            / 3.0
    }
}

// Time conversion functions
pub fn datetime_to_et(datetime: NaiveDateTime) -> EphemerisTime {
    // SPICEはRFC3339のタイムゾーン部分を認識しないので、ISO形式で変換
    ensure_spice_ready();
    let utc_str = datetime.format("%Y-%m-%dT%H:%M:%S%.6f").to_string();
    let et = spice::str2et(&utc_str);
    EphemerisTime { value: et }
}

// State and position functions
// aberration correctionはNONE
// observerは地球
// spkezrは、指定した天体の位置と速度を取得する関数
// spkezrの戻り値は、位置km、速度km/sの6要素ベクトルと光遅延補正の時間（秒）
pub fn get_state(et: EphemerisTime, target: CelestialBody) -> (ECIPosition, ECIVelocity) {
    ensure_spice_ready();
    let (pos_vel, _) = spice::spkezr(
        &target.spkezr_string(),
        et.value,
        ECI_COORDINATE_SYSTEM,
        "NONE",
        &CelestialBody::Earth.to_string(),
    );

    let position = ECIPosition {
        x: pos_vel[0] * 1.0e3, // Convert from km to m
        y: pos_vel[1] * 1.0e3,
        z: pos_vel[2] * 1.0e3,
    };
    let velocity = ECIVelocity {
        x: pos_vel[3] * 1.0e3, // Convert from km/s to m/s
        y: pos_vel[4] * 1.0e3,
        z: pos_vel[5] * 1.0e3,
    };
    (position, velocity)
}

pub fn get_constant(body: CelestialBody, item: CelestialConstantsType) -> CelestialConstants {
    ensure_spice_ready();
    match item {
        CelestialConstantsType::GM => {
            let gm = spice::bodvrd(&body.to_string(), &item.to_string(), 1);
            CelestialConstants::GM(gm[0])
        }
        CelestialConstantsType::RADII => {
            let radii = spice::bodvrd(&body.to_string(), &item.to_string(), 3);
            CelestialConstants::Radii(Radii {
                equatorial_semi_major_axis: radii[0] * 1.0e3, // Convert from km to m
                equatorial_semi_minor_axis: radii[1] * 1.0e3,
                polar_semi_axis: radii[2] * 1.0e3,
            })
        }
    }
}

fn pos_transform(
    position: Vector3<f64>,
    from_frame: &str,
    to_frame: &str,
    et: EphemerisTime,
) -> Vector3<f64> {
    ensure_spice_ready();
    // 座標系変換行列を取得
    let transform_raw = spice::pxform(from_frame, to_frame, et.value);
    let transform = Matrix3::from(transform_raw);

    // 位置ベクトルを変換
    transform * position
}

fn pos_vel_transform(
    pos_vel: Vector6<f64>,
    from_frame: &str,
    to_frame: &str,
    et: EphemerisTime,
) -> Vector6<f64> {
    ensure_spice_ready();
    let transform = unsafe {
        use std::ffi::CString;
        // sxform_cのための変数準備
        let from_frame = CString::new(from_frame).unwrap();
        let to_frame = CString::new(to_frame).unwrap();
        let mut state_transform = [[0.0_f64; 6]; 6];
        spice::c::sxform_c(
            from_frame.as_ptr() as *mut i8,
            to_frame.as_ptr() as *mut i8,
            et.value,
            state_transform.as_mut_ptr(),
        );
        Matrix6::from(state_transform)
    };

    // 位置と速度を変換
    transform * pos_vel
}

pub fn pos_eci_to_ecef(position: ECIPosition, et: EphemerisTime) -> ECEFPosition {
    // eci座標系からecef座標系への変換
    ECEFPosition::from(pos_transform(
        Vector3::new(position.x, position.y, position.z),
        ECI_COORDINATE_SYSTEM,
        ECEF_COORDINATE_SYSTEM,
        et,
    ))
}

pub fn pos_vel_eci_to_ecef(
    position: ECIPosition,
    velocity: ECIVelocity,
    et: EphemerisTime,
) -> (ECEFPosition, ECEFVelocity) {
    // eci座標系からecef座標系への位置と速度の変換
    let pos_vel = pos_vel_transform(
        Vector6::new(
            position.x, position.y, position.z, velocity.x, velocity.y, velocity.z,
        ),
        ECI_COORDINATE_SYSTEM,
        ECEF_COORDINATE_SYSTEM,
        et,
    );
    // ECEFPosition/Velocityに変換（From traitを使用）
    let ecef_position = ECEFPosition::from(pos_vel.fixed_rows::<3>(0).into_owned());
    let ecef_velocity = ECEFVelocity::from(pos_vel.fixed_rows::<3>(3).into_owned());

    (ecef_position, ecef_velocity)
}

pub fn pos_ecef_to_eci(position: ECEFPosition, et: EphemerisTime) -> ECIPosition {
    // ecef座標系からeci座標系への変換
    ECIPosition::from(pos_transform(
        Vector3::new(position.x, position.y, position.z),
        ECEF_COORDINATE_SYSTEM,
        ECI_COORDINATE_SYSTEM,
        et,
    ))
}

pub fn pos_vel_ecef_to_eci(
    position: ECEFPosition,
    velocity: ECEFVelocity,
    et: EphemerisTime,
) -> (ECIPosition, ECIVelocity) {
    // ecef座標系からeci座標系への位置と速度の変換
    let pos_vel = pos_vel_transform(
        Vector6::new(
            position.x, position.y, position.z, velocity.x, velocity.y, velocity.z,
        ),
        ECEF_COORDINATE_SYSTEM,
        ECI_COORDINATE_SYSTEM,
        et,
    );
    // ECIPosition/Velocityに変換（From traitを使用）
    let eci_position = ECIPosition::from(pos_vel.fixed_rows::<3>(0).into_owned());
    let eci_velocity = ECIVelocity::from(pos_vel.fixed_rows::<3>(3).into_owned());

    (eci_position, eci_velocity)
}

// pub fn pos_teme_to_eci(position: TemePosition, et: EphemerisTime) -> ECIPosition {
//     // TEME座標系からeci座標系への変換
//     ECIPosition::from(
//         pos_transform(
//             Vector3::new(position.x, position.y, position.z),
//             "TEME",
//             ECI_COORDINATE_SYSTEM,
//             et,
//         )
//     )
// }

// pub fn pos_vel_teme_to_eci(
//     position: ECEFPosition,
//     velocity: ECEFVelocity,
//     et: EphemerisTime,
// ) -> (ECIPosition, ECIVelocity) {
//     // TEME座標系からeci座標系への位置と速度の変換
//     let pos_vel = pos_vel_transform(
//         Vector6::new(position.x, position.y, position.z, velocity.x, velocity.y, velocity.z),
//         "TEME",
//         ECI_COORDINATE_SYSTEM,
//         et,
//     );
//     // ECIPosition/Velocityに変換（From traitを使用）
//     let eci_position = ECIPosition::from(pos_vel.fixed_rows::<3>(0).into_owned());
//     let eci_velocity = ECIVelocity::from(pos_vel.fixed_rows::<3>(3).into_owned());

//     (eci_position, eci_velocity)
// }
