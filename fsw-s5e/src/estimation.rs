pub mod attitude_determination;
pub mod direction_estimation;
pub mod orbit_determination;
pub mod timer;

fn orthonormalize_matrix3(matrix: &nalgebra::Matrix3<f64>) -> nalgebra::Rotation3<f64> {
    let u = matrix.column(0).normalize();
    let v = (matrix.column(1) - u * u.dot(&matrix.column(1))).normalize();
    let w = u.cross(&v);

    nalgebra::Rotation3::from_matrix_unchecked(nalgebra::Matrix3::from_columns(&[u, v, w]))
}

pub fn triad_method(
    b_body: &nalgebra::Vector3<f64>,
    s_body: &nalgebra::Vector3<f64>,
    b_inertial: &nalgebra::Vector3<f64>,
    s_inertial: &nalgebra::Vector3<f64>,
) -> nalgebra::UnitQuaternion<f64> {
    let t1_body = b_body.normalize();
    let t2_body = s_body.cross(b_body).normalize();
    let t3_body = t1_body.cross(&t2_body);

    let t1_inertial = b_inertial.normalize();
    let t2_inertial = s_inertial.cross(b_inertial).normalize();
    let t3_inertial = t1_inertial.cross(&t2_inertial);

    let rotation_matrix_raw = nalgebra::Matrix3::from_columns(&[t1_body, t2_body, t3_body])
        * nalgebra::Matrix3::from_columns(&[t1_inertial, t2_inertial, t3_inertial]).transpose();

    let rotation_matrix = orthonormalize_matrix3(&rotation_matrix_raw);

    nalgebra::UnitQuaternion::from_rotation_matrix(&rotation_matrix)
}

pub use attitude_determination::AttitudeDetermination;
pub use direction_estimation::DirectionEstimator;
pub use orbit_determination::OrbitDetermination;
pub use timer::Timer;
