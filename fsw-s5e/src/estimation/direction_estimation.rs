use control_system::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::Direction,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter, input_shift},
    value_structs::EmptyInput,
};
use nalgebra::{Matrix1, Matrix2, Matrix3, SMatrix, SVector, UnitVector3, Vector3};
use std::time::{Duration, Instant};

use crate::{
    constants::{
        GYRO_BIAS_DRIFT_STD, MAG_FIELD_DIR_WALK_STD, MAG_FIELD_NORM_WALK_STD,
        SUN_DIRECTION_WALK_STD, UKF_ALPHA, UKF_BETA, UKF_KAPPA,
    },
    data,
};

#[derive(Default)]
pub struct DirectionEstimatorProfile {
    pub propagation: Duration,
    pub mag_update: Duration,
    pub sun_update: Duration,
    pub output_extraction: Duration,
    pub call_count: u64,
}

impl DirectionEstimatorProfile {
    pub fn print_summary(&self) {
        let total = self.propagation + self.mag_update + self.sun_update + self.output_extraction;
        let pct = |d: Duration| {
            if total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / total.as_secs_f64() * 100.0
            }
        };
        println!(
            "=== DirectionEstimator Profile ({} calls, total {:.3}s) ===",
            self.call_count,
            total.as_secs_f64()
        );
        println!(
            "  propagation:        {:>8.3}ms ({:>5.1}%)",
            self.propagation.as_secs_f64() * 1000.0,
            pct(self.propagation)
        );
        println!(
            "  mag_update:         {:>8.3}ms ({:>5.1}%)",
            self.mag_update.as_secs_f64() * 1000.0,
            pct(self.mag_update)
        );
        println!(
            "  sun_update:         {:>8.3}ms ({:>5.1}%)",
            self.sun_update.as_secs_f64() * 1000.0,
            pct(self.sun_update)
        );
        println!(
            "  output_extraction:  {:>8.3}ms ({:>5.1}%)",
            self.output_extraction.as_secs_f64() * 1000.0,
            pct(self.output_extraction)
        );
    }
}

fn eclipse_direction_estimation_process_noise_covariance() -> nalgebra::SMatrix<f64, 6, 6> {
    let mut matrix = nalgebra::SMatrix::<f64, 6, 6>::zeros();
    matrix
        .fixed_view_mut::<2, 2>(0, 0)
        .copy_from(&(Matrix2::identity() * MAG_FIELD_DIR_WALK_STD * MAG_FIELD_DIR_WALK_STD));
    matrix
        .fixed_view_mut::<1, 1>(2, 2)
        .copy_from(&(Matrix1::identity() * MAG_FIELD_NORM_WALK_STD * MAG_FIELD_NORM_WALK_STD));
    matrix
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(Matrix3::identity() * GYRO_BIAS_DRIFT_STD * GYRO_BIAS_DRIFT_STD));
    matrix
}

fn full_direction_estimation_process_noise_covariance() -> nalgebra::SMatrix<f64, 8, 8> {
    let mut matrix = nalgebra::SMatrix::<f64, 8, 8>::zeros();
    matrix
        .fixed_view_mut::<2, 2>(0, 0)
        .copy_from(&(Matrix2::identity() * MAG_FIELD_DIR_WALK_STD * MAG_FIELD_DIR_WALK_STD));
    matrix
        .fixed_view_mut::<1, 1>(2, 2)
        .copy_from(&(Matrix1::identity() * MAG_FIELD_NORM_WALK_STD * MAG_FIELD_NORM_WALK_STD));
    matrix
        .fixed_view_mut::<2, 2>(3, 3)
        .copy_from(&(Matrix2::identity() * SUN_DIRECTION_WALK_STD * SUN_DIRECTION_WALK_STD));
    matrix
        .fixed_view_mut::<3, 3>(5, 5)
        .copy_from(&(Matrix3::identity() * GYRO_BIAS_DRIFT_STD * GYRO_BIAS_DRIFT_STD));
    matrix
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct MagInitializationInput {
    mag_field: Vector3<f64>,
}

#[derive(EstimationOutputStruct, Clone, Debug)]
struct MagNormInitializationOutput {
    mag_norm: f64,
}

#[derive(EstimationOutputStruct, Clone, Debug)]
struct MagDirectionInitializationOutput {
    mag_direction: Direction,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct DirectionInput {
    angular_velocity: Vector3<f64>,
}

#[derive(Debug, Clone, EstimationState)]
struct FullDirectionState {
    mag_dir: Direction,
    mag_norm: f64,
    sun_direction: Direction,
    gyro_bias: SVector<f64, 3>,
}

struct FullDirectionPropagationModel;

impl PropagationModel for FullDirectionPropagationModel {
    type State = FullDirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = DirectionInput;
    type Time = crate::Time;
    type Dt = f64;
    fn propagate(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        let omega_est = gaussian_input.angular_velocity - state.gyro_bias;
        let axisangle = omega_est * *dt;
        let rot = nalgebra::Rotation3::new(axisangle);
        let new_mag_dir = rot * state.mag_dir.clone();
        let new_sun_direction = rot * state.sun_direction.clone();
        FullDirectionState {
            mag_dir: new_mag_dir,
            mag_norm: state.mag_norm,
            sun_direction: new_sun_direction,
            gyro_bias: state.gyro_bias.clone_owned(),
        }
    }
}

#[derive(Debug, Clone, EstimationOutputStruct)]
struct MagFieldObservationForFull {
    mag_field: Vector3<f64>,
}

struct MagFieldObservationModelForFull;

impl ObservationModel for MagFieldObservationModelForFull {
    type State = FullDirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = crate::Time;
    type Observation = MagFieldObservationForFull;
    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        _gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        let mag_field = state.mag_dir.dir().into_inner() * state.mag_norm;
        MagFieldObservationForFull { mag_field }
    }
}

#[derive(Debug, Clone, EstimationOutputStruct)]
struct SunDirectionObservation {
    sun_direction: Direction,
}

struct SunDirectionObservationModel;

impl ObservationModel for SunDirectionObservationModel {
    type State = FullDirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = crate::Time;
    type Observation = SunDirectionObservation;
    fn predict(
        &self,
        state: &FullDirectionState,
        _deterministic_input: &EmptyInput,
        _gaussian_input: &EmptyInput,
        _time: &crate::Time,
    ) -> SunDirectionObservation {
        SunDirectionObservation {
            sun_direction: state.sun_direction.clone(),
        }
    }
}

struct FullDirectionEstimation {
    ukf: UnscentedKalmanFilter<
        FullDirectionState,
        crate::Time,
        f64,
        FullDirectionPropagationModel,
        EmptyInput,
        DirectionInputGaussian,
        8,
        3,
    >,
    mag_obs: MagFieldObservationModelForFull,
    sun_obs: SunDirectionObservationModel,
}

impl FullDirectionEstimation {
    fn new(
        eclipse_state: &EclipseDirectionState,
        eclipse_covariance: &nalgebra::SMatrix<f64, 6, 6>,
        initial_time: &crate::Time,
        initial_sun_direction: Direction,
        initial_sun_direction_covariance: Matrix2<f64>,
    ) -> Self {
        let propagator = FullDirectionPropagationModel;
        let mag_dir = eclipse_state.mag_dir.clone();
        let mag_dir_covariance = eclipse_covariance.fixed_view::<2, 2>(0, 0).clone_owned();
        let mag_norm = eclipse_state.mag_norm;
        let mag_norm_covariance = eclipse_covariance.fixed_view::<1, 1>(2, 2).clone_owned();
        let gyro_bias = eclipse_state.gyro_bias.clone_owned();
        let gyro_bias_variance = eclipse_covariance.fixed_view::<3, 3>(3, 3).clone_owned();
        let ukf = UnscentedKalmanFilter::new(
            propagator,
            FullDirectionState {
                mag_dir,
                mag_norm,
                sun_direction: initial_sun_direction,
                gyro_bias,
            },
            {
                let mut cov = SMatrix::<f64, 8, 8>::zeros();
                // Magnetic field initial covariance (unknown until observed)
                cov.fixed_view_mut::<2, 2>(0, 0)
                    .copy_from(&mag_dir_covariance);
                cov.fixed_view_mut::<1, 1>(2, 2)
                    .copy_from(&mag_norm_covariance);
                // Sun direction initial covariance
                cov.fixed_view_mut::<2, 2>(3, 3)
                    .copy_from(&initial_sun_direction_covariance);
                // Gyro bias initial covariance (reduced to match attitude_determination)
                cov.fixed_view_mut::<3, 3>(5, 5)
                    .copy_from(&gyro_bias_variance);
                cov
            },
            initial_time,
            UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        );
        let mag_obs = MagFieldObservationModelForFull;
        let sun_obs = SunDirectionObservationModel;
        FullDirectionEstimation {
            ukf,
            mag_obs,
            sun_obs,
        }
    }
}

#[derive(Debug, Clone, EstimationState)]
struct EclipseDirectionState {
    mag_dir: Direction,
    mag_norm: f64,
    gyro_bias: SVector<f64, 3>,
}

struct EclipseDirectionPropagationModel;

impl PropagationModel for EclipseDirectionPropagationModel {
    type State = EclipseDirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = DirectionInput;
    type Time = crate::Time;
    type Dt = f64;
    fn propagate(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        let omega_est = gaussian_input.angular_velocity - state.gyro_bias;
        let axisangle = omega_est * *dt;
        let rot = nalgebra::Rotation3::new(axisangle);
        let new_magnetic_basis = rot * state.mag_dir.clone();
        EclipseDirectionState {
            mag_dir: new_magnetic_basis,
            mag_norm: state.mag_norm,
            gyro_bias: state.gyro_bias.clone_owned(),
        }
    }
}

#[derive(Debug, Clone, EstimationOutputStruct)]
struct MagFieldObservationForEclipse {
    mag_field: Vector3<f64>,
}

struct MagFieldObservationModelForEclipse;

impl ObservationModel for MagFieldObservationModelForEclipse {
    type State = EclipseDirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = crate::Time;
    type Observation = MagFieldObservationForEclipse;
    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        _gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        MagFieldObservationForEclipse {
            mag_field: state.mag_dir.dir().into_inner() * state.mag_norm,
        }
    }
}

pub struct EclipseDirectionEstimation {
    ukf: UnscentedKalmanFilter<
        EclipseDirectionState,
        crate::Time,
        f64,
        EclipseDirectionPropagationModel,
        EmptyInput,
        DirectionInputGaussian,
        6,
        3,
    >,
    mag_obs: MagFieldObservationModelForEclipse,
}

impl EclipseDirectionEstimation {
    fn new(
        initial_time: &crate::Time,
        initial_mag_field: SVector<f64, 3>,
        initial_mag_field_variance: Matrix3<f64>,
    ) -> Self {
        let propagator = EclipseDirectionPropagationModel;
        let mag_init_input = MagInitializationInputGaussian {
            mag_field: initial_mag_field,
            mag_field_covariance: initial_mag_field_variance,
        };
        let mag_norm_func = |input: &MagInitializationInput| MagNormInitializationOutput {
            mag_norm: input.mag_field.norm(),
        };
        let (mag_norm, mag_norm_covariance) = input_shift(
            &mag_init_input,
            mag_norm_func,
            &UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        )
        .unwrap();
        let mag_dir_func = |input: &MagInitializationInput| MagDirectionInitializationOutput {
            mag_direction: Direction::from_dir(UnitVector3::new_normalize(input.mag_field)),
        };
        let (mag_dir, mag_dir_covariance) = input_shift(
            &mag_init_input,
            mag_dir_func,
            &UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        )
        .unwrap();
        let ukf = UnscentedKalmanFilter::new(
            propagator,
            EclipseDirectionState {
                mag_dir: mag_dir.mag_direction,
                mag_norm: mag_norm.mag_norm,
                gyro_bias: SVector::<f64, 3>::zeros(),
            },
            {
                let mut cov = SMatrix::<f64, 6, 6>::zeros();
                // Magnetic field initial covariance (unknown until observed)
                cov.fixed_view_mut::<2, 2>(0, 0)
                    .copy_from(&mag_dir_covariance);
                cov.fixed_view_mut::<1, 1>(2, 2)
                    .copy_from(&mag_norm_covariance);
                // Gyro bias initial covariance (reduced to match attitude_determination)
                cov.fixed_view_mut::<3, 3>(3, 3)
                    .copy_from(&(SMatrix::<f64, 3, 3>::identity() * 1.0e0));
                cov
            },
            initial_time,
            UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        );
        let mag_obs = MagFieldObservationModelForEclipse;
        EclipseDirectionEstimation { ukf, mag_obs }
    }

    fn to_full_mode(
        &self,
        time: &crate::Time,
        sun_direction: Direction,
        sun_direction_variance: Matrix2<f64>,
    ) -> FullDirectionEstimation {
        FullDirectionEstimation::new(
            self.ukf.state(),
            self.ukf.covariance(),
            time,
            sun_direction,
            sun_direction_variance,
        )
    }
}

pub struct DirectionEstimatorInput {
    pub magnetic_field: Option<data::MagnetometerData>,
    pub gyro_data: Option<data::GyroSensorData>,
    pub sun_direction: Option<data::SunSensorData>,
    pub inertial_mag: Option<data::InertialMagneticFieldData>,
    pub inertial_sun: Option<data::InertialSunDirectionData>,
    pub star_tracker: Option<data::StarTrackerData>,
}

pub struct DirectionEstimatorOutput {
    pub sun_direction: Option<data::SunDirectionEstimationData>,
    pub magnetic_field: Option<data::MagneticFieldEstimationData>,
    pub angular_velocity: Option<data::AngularVelocityData>,
}

#[allow(clippy::large_enum_variant)]
enum DirectionEstimatorMode {
    Initial,
    Eclipse(EclipseDirectionEstimation),
    FullDirection(FullDirectionEstimation),
}

pub struct DirectionEstimator {
    mode: DirectionEstimatorMode,
    pub profile: DirectionEstimatorProfile,
}

impl Default for DirectionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectionEstimator {
    pub fn new() -> Self {
        Self {
            mode: DirectionEstimatorMode::Initial,
            profile: DirectionEstimatorProfile::default(),
        }
    }

    pub fn main_loop(
        &mut self,
        time: &crate::Time,
        input: &DirectionEstimatorInput,
    ) -> DirectionEstimatorOutput {
        self.profile.call_count += 1;
        if let Some(gyro_data) = input.gyro_data.as_ref() {
            let propagation_input = DirectionInputGaussian {
                angular_velocity: gyro_data.angular_velocity,
                angular_velocity_covariance: Matrix3::identity() * gyro_data.std * gyro_data.std,
            };
            if let (DirectionEstimatorMode::Initial, Some(mag_data)) =
                (&self.mode, input.magnetic_field.as_ref())
            {
                // Transition to eclipse mode if magnetic field is available
                let eclipse_estimator = EclipseDirectionEstimation::new(
                    time,
                    mag_data.magnetic_field,
                    Matrix3::identity() * mag_data.std * mag_data.std,
                );
                self.mode = DirectionEstimatorMode::Eclipse(eclipse_estimator);
            }
            if let (DirectionEstimatorMode::Eclipse(eclipse_estimator), Some(sun_data)) =
                (&self.mode, input.sun_direction.as_ref())
            {
                // Transition to full direction mode if sun direction is available
                let sun_direction_variance = Matrix2::identity() * sun_data.std * sun_data.std;
                let sun_direction = Direction::from_dir(sun_data.sun_direction);
                let full_estimator =
                    eclipse_estimator.to_full_mode(time, sun_direction, sun_direction_variance);
                self.mode = DirectionEstimatorMode::FullDirection(full_estimator);
            }
            match &mut self.mode {
                DirectionEstimatorMode::Initial => DirectionEstimatorOutput {
                    sun_direction: None,
                    magnetic_field: None,
                    angular_velocity: None,
                },
                DirectionEstimatorMode::Eclipse(eclipse_estimator) => {
                    let t_prof = Instant::now();
                    let process_noise_covariance =
                        Some(eclipse_direction_estimation_process_noise_covariance());
                    // Propagation step
                    eclipse_estimator
                        .ukf
                        .propagate(
                            &EmptyInput,
                            &propagation_input,
                            process_noise_covariance,
                            time,
                        )
                        .unwrap();
                    self.profile.propagation += t_prof.elapsed();
                    // Eclipse mode processing
                    let t_prof = Instant::now();
                    if let Some(mag_data) = input.magnetic_field.as_ref() {
                        let observation = MagFieldObservationForEclipse {
                            mag_field: mag_data.magnetic_field,
                        };
                        eclipse_estimator
                            .ukf
                            .update(
                                &eclipse_estimator.mag_obs,
                                &observation,
                                &EmptyInput,
                                &EmptyInput,
                                time,
                                Matrix3::identity() * mag_data.std * mag_data.std,
                            )
                            .unwrap();
                    }
                    self.profile.mag_update += t_prof.elapsed();
                    let t_prof = Instant::now();
                    let state = eclipse_estimator.ukf.state();
                    let angular_velocity_est = gyro_data.angular_velocity - state.gyro_bias;
                    let angular_velocity_covariance = eclipse_estimator
                        .ukf
                        .covariance()
                        .fixed_view::<3, 3>(3, 3)
                        .clone_owned()
                        + Matrix3::identity() * gyro_data.std * gyro_data.std;
                    let angular_velocity_output = data::AngularVelocityData {
                        angular_velocity: angular_velocity_est,
                        angular_velocity_variance: angular_velocity_covariance,
                    };
                    let output = DirectionEstimatorOutput {
                        sun_direction: None,
                        magnetic_field: Some(data::MagneticFieldEstimationData {
                            magnetic_field_direction: state.mag_dir.dir(),
                            magnetic_field_direction_variance: eclipse_estimator
                                .ukf
                                .covariance()
                                .fixed_view::<2, 2>(0, 0)
                                .clone_owned(),
                            magnetic_field_norm: state.mag_norm,
                            magnetic_field_norm_variance: eclipse_estimator
                                .ukf
                                .covariance()
                                .fixed_view::<1, 1>(2, 2)
                                .clone_owned(),
                        }),
                        angular_velocity: Some(angular_velocity_output),
                    };
                    self.profile.output_extraction += t_prof.elapsed();
                    output
                }
                DirectionEstimatorMode::FullDirection(full_estimator) => {
                    // Propagation step
                    let t_prof = Instant::now();
                    let process_noise_covariance =
                        Some(full_direction_estimation_process_noise_covariance());
                    full_estimator
                        .ukf
                        .propagate(
                            &EmptyInput,
                            &propagation_input,
                            process_noise_covariance,
                            time,
                        )
                        .unwrap();
                    self.profile.propagation += t_prof.elapsed();
                    // Full direction mode processing
                    let t_prof = Instant::now();
                    if let Some(mag_data) = input.magnetic_field.as_ref() {
                        let observation = MagFieldObservationForFull {
                            mag_field: mag_data.magnetic_field,
                        };
                        full_estimator
                            .ukf
                            .update(
                                &full_estimator.mag_obs,
                                &observation,
                                &EmptyInput,
                                &EmptyInput,
                                time,
                                Matrix3::identity() * mag_data.std * mag_data.std,
                            )
                            .unwrap();
                    }
                    self.profile.mag_update += t_prof.elapsed();
                    let t_prof = Instant::now();
                    if let Some(sun_data) = input.sun_direction.as_ref() {
                        let sun_direction = Direction::from_dir(sun_data.sun_direction);
                        let observation = SunDirectionObservation { sun_direction };
                        full_estimator
                            .ukf
                            .update(
                                &full_estimator.sun_obs,
                                &observation,
                                &EmptyInput,
                                &EmptyInput,
                                time,
                                Matrix2::identity() * sun_data.std * sun_data.std,
                            )
                            .unwrap();
                    }
                    self.profile.sun_update += t_prof.elapsed();
                    let t_prof = Instant::now();
                    let state = full_estimator.ukf.state();
                    let angular_velocity_est = gyro_data.angular_velocity - state.gyro_bias;
                    let angular_velocity_covariance = full_estimator
                        .ukf
                        .covariance()
                        .fixed_view::<3, 3>(5, 5)
                        .clone_owned()
                        + Matrix3::identity() * gyro_data.std * gyro_data.std;
                    let angular_velocity_output = data::AngularVelocityData {
                        angular_velocity: angular_velocity_est,
                        angular_velocity_variance: angular_velocity_covariance,
                    };
                    let sun_direction_output = data::SunDirectionEstimationData {
                        sun_direction: state.sun_direction.dir(),
                        sun_direction_variance: full_estimator
                            .ukf
                            .covariance()
                            .fixed_view::<2, 2>(3, 3)
                            .clone_owned(),
                    };
                    let magnetic_field_output = data::MagneticFieldEstimationData {
                        magnetic_field_direction: state.mag_dir.dir(),
                        magnetic_field_direction_variance: full_estimator
                            .ukf
                            .covariance()
                            .fixed_view::<2, 2>(0, 0)
                            .clone_owned(),
                        magnetic_field_norm: state.mag_norm,
                        magnetic_field_norm_variance: full_estimator
                            .ukf
                            .covariance()
                            .fixed_view::<1, 1>(2, 2)
                            .clone_owned(),
                    };
                    let output = DirectionEstimatorOutput {
                        sun_direction: Some(sun_direction_output),
                        magnetic_field: Some(magnetic_field_output),
                        angular_velocity: Some(angular_velocity_output),
                    };
                    self.profile.output_extraction += t_prof.elapsed();
                    output
                }
            }
        } else {
            // No gyro data; cannot propagate
            DirectionEstimatorOutput {
                sun_direction: None,
                magnetic_field: None,
                angular_velocity: None,
            }
        }
    }
}

#[cfg(test)]
mod tests_util {
    use control_system::components::{Direction, GaussianNominalType};

    #[test]
    fn rot_test() {
        use nalgebra::{Rotation3, UnitQuaternion, Vector3};

        let axisangle = Vector3::new(0.1f64, 0.2, 0.3);
        let v = Vector3::new(1.0, 2.0, 3.0);

        let q = UnitQuaternion::new(axisangle);
        let rot = Rotation3::new(axisangle);

        let vq = q * v;
        let vr = rot * v;

        // 誤差込みで一致を見る
        assert!(vq.relative_eq(&vr, 1.0e-12, 1.0e-12));
    }
    #[test]
    fn rot_float_error_test() {
        use control_system::components::GaussianValueType;
        println!();
        let v = nalgebra::Vector3::x_axis();
        let rot_criteria =
            nalgebra::Rotation3::face_towards(&v.into_inner(), &nalgebra::Vector3::y());
        let dir_criteria = Direction::from_dir(v);
        let criteria_q = nalgebra::UnitQuaternion::from_rotation_matrix(&rot_criteria);
        let axisangle_nominal =
            nalgebra::SVector::<f64, 3>::new(0.0, 0.0, core::f64::consts::FRAC_PI_2);
        let delta_axisangle =
            nalgebra::SVector::<f64, 3>::new(0.0, core::f64::consts::FRAC_PI_2, 0.0);

        let q_nominal = nalgebra::UnitQuaternion::new(axisangle_nominal);
        let v_rot_nominal_from_direct = q_nominal * v.into_inner();
        let q_rot_nominal = q_nominal * criteria_q;
        let rot_nominal = q_rot_nominal.to_rotation_matrix();
        let rot_dir = q_nominal.to_rotation_matrix() * dir_criteria.clone();
        println!("Rot nominal: {:?}", rot_nominal);
        let nominal_basis_2d = rot_nominal.matrix().fixed_view::<3, 2>(0, 0).clone_owned();
        println!("{:?}", nominal_basis_2d);
        let v_rot_nominal_from_q = rot_nominal.matrix().column(2).clone_owned();

        assert!((v_rot_nominal_from_direct - v_rot_nominal_from_q).norm() < 1.0e-12);

        let axisangle_pos = axisangle_nominal + delta_axisangle;
        let q_pos = nalgebra::UnitQuaternion::new(axisangle_pos);
        let q_rot_pos = q_pos * criteria_q;
        let rot_pos = q_rot_pos.to_rotation_matrix();
        let axisangle_pos_from_nominal = q_rot_pos.error(&q_rot_nominal);
        let angle_projection_pos =
            nominal_basis_2d * nominal_basis_2d.transpose() * axisangle_pos_from_nominal;
        let q_projection_pos = nalgebra::UnitQuaternion::new(angle_projection_pos);
        let q_rot_pos_projected = q_projection_pos * q_rot_nominal;
        let rot_pos_projected = q_rot_pos_projected.to_rotation_matrix();
        let rot_dir_pos = q_pos.to_rotation_matrix() * dir_criteria.clone();
        let dir_error_pos = rot_dir_pos.error(&rot_dir);
        println!();
        println!("{:?}", axisangle_pos_from_nominal);
        println!("{:?}", angle_projection_pos);
        println!("{:?}", rot_pos.matrix().column(2).clone_owned());
        println!("{:?}", rot_pos_projected.matrix().column(2).clone_owned());
        println!(
            "{:?}",
            rot_pos.matrix().column(2) - rot_pos_projected.matrix().column(2)
        );
        println!("Rot dir pos: {:?}", rot_dir_pos.dir());
        println!("dir error pos: {:?}", dir_error_pos);
        let reconst_dir_pos = rot_dir_pos.merge_sigma(&dir_error_pos);
        println!("Reconst dir pos: {:?}", reconst_dir_pos.dir());

        let axisangle_neg = axisangle_nominal - delta_axisangle;
        let q_neg = nalgebra::UnitQuaternion::new(axisangle_neg);
        let q_rot_neg = q_neg * criteria_q;
        let axisangle_neg_from_nominal = q_rot_neg.error(&q_rot_nominal);
        let angle_2d_neg = nominal_basis_2d.transpose() * axisangle_neg_from_nominal;
        let rot_dir_neg = q_neg.to_rotation_matrix() * dir_criteria.clone();
        let dir_error_neg = rot_dir_neg.error(&rot_dir);
        println!();
        println!("{:?}", axisangle_neg_from_nominal);
        println!("{:?}", angle_2d_neg);
        println!("Rot dir neg: {:?}", rot_dir_neg.dir());
        println!("dir error neg: {:?}", dir_error_neg);
    }
}

#[cfg(test)]
mod tests_eclipse {
    use super::*;
    use nalgebra::Matrix3;
    use rand::distributions::Distribution;
    use statrs::distribution::MultivariateNormal;

    // ========== Initialization Tests ==========
    mod initialization {
        use super::*;

        #[test]
        fn test_ukf_mode_initialization() {
            let estimator = DirectionEstimator::new();

            // Check initial state
            assert!(matches!(estimator.mode, DirectionEstimatorMode::Initial));
        }

        #[test]
        fn test_eclipse_mode_initialization() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 50.0, -30.0);
            let initial_mag_field_variance = Matrix3::identity() * 10.0;

            let estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );

            // Check initial state values
            assert!(
                (estimator
                    .ukf
                    .state()
                    .mag_dir
                    .dir()
                    .angle(&initial_mag_field))
                .abs()
                    < 1e-5,
                "Initial magnetic field should match input, initialized: {:?}, input: {:?}, error: {:?}",
                estimator.ukf.state().mag_dir.dir(),
                initial_mag_field.normalize(),
                estimator
                    .ukf
                    .state()
                    .mag_dir
                    .dir()
                    .angle(&initial_mag_field),
            );
            assert!(
                estimator.ukf.state().gyro_bias.norm() < 1e-10,
                "Initial gyro bias should be zero"
            );

            // Check covariance is positive definite
            let cov = estimator.ukf.covariance();
            let eigenvalues = cov.symmetric_eigenvalues();
            for eig in eigenvalues.iter() {
                assert!(eig > &0.0, "Initial covariance should be positive definite");
            }
        }

        #[test]
        fn test_transition_to_eclipse_mode() {
            let mut estimator = DirectionEstimator::new();
            let time = crate::Time::from_seconds(0.0);

            let mag_data = data::MagnetometerData {
                magnetic_field: nalgebra::SVector::<f64, 3>::new(100.0, 50.0, -30.0),
                std: 1.0,
            };

            let gyro_data = data::GyroSensorData {
                angular_velocity: nalgebra::SVector::<f64, 3>::zeros(),
                std: 0.01,
            };

            let input = DirectionEstimatorInput {
                magnetic_field: Some(mag_data),
                gyro_data: Some(gyro_data),
                sun_direction: None,
                inertial_mag: None,
                inertial_sun: None,
                star_tracker: None,
            };

            let _ = estimator.main_loop(&time, &input);

            assert!(
                matches!(estimator.mode, DirectionEstimatorMode::Eclipse(_)),
                "Should transition to Eclipse mode when magnetic field is available"
            );
        }
    }

    // ========== Propagation Tests ==========
    mod propagation {
        use astrodynamics::coordinate::BodyVector;
        use control_system::{
            components::GaussianValueType,
            integrator::{
                Prediction, TimeIntegrator,
                rk4::{RK4Phase, RK4Solver},
            },
        };
        use nalgebra::{Quaternion, UnitQuaternion, UnitVector3, Vector2};

        use crate::constants::GYRO_NOISE_STD;

        use super::*;
        #[test]
        fn test_propagation_pos_and_neg() {
            let propagation_model = EclipseDirectionPropagationModel;
            let initial_mag_dir = Direction::from_dir(UnitVector3::new_normalize(
                nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
            ));
            let initial_state = EclipseDirectionState {
                mag_dir: initial_mag_dir.clone(),
                mag_norm: 100.0,
                gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
            };
            let omega = nalgebra::SVector::<f64, 3>::new(0.0, 0.0, 1.0);
            let dt = 0.1;

            let g_input_nominal = DirectionInput {
                angular_velocity: omega,
            };
            let new_state_nominal = propagation_model.propagate(
                &initial_state,
                &EmptyInput,
                &g_input_nominal,
                &crate::Time::from_seconds(0.0),
                &dt,
            );
            let nominal_error = new_state_nominal.mag_dir.error(&initial_state.mag_dir);
            println!("Nominal error: {:?}", nominal_error);

            let g_input_pos = DirectionInput {
                angular_velocity: omega + Vector3::new(1.0, 0.0, 0.0),
            };
            let new_state_pos = propagation_model.propagate(
                &initial_state,
                &EmptyInput,
                &g_input_pos,
                &crate::Time::from_seconds(0.0),
                &dt,
            );
            let pos_error = new_state_pos.mag_dir.error(&initial_state.mag_dir);

            let g_input_neg = DirectionInput {
                angular_velocity: omega - Vector3::new(1.0, 0.0, 0.0),
            };
            let new_state_neg = propagation_model.propagate(
                &initial_state,
                &EmptyInput,
                &g_input_neg,
                &crate::Time::from_seconds(0.0),
                &dt,
            );
            let neg_error = new_state_neg.mag_dir.error(&initial_state.mag_dir);

            println!("Pos error: {:?}", pos_error);
            println!("Neg error: {:?}", neg_error);
        }

        #[test]
        fn test_propagation_model_with_rk4_isotropic() {
            let mut time = 0.0;
            let dt = 0.01;
            let inertia = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
            let b_inertial = nalgebra::SVector::<f64, 3>::new(
                9894.151141285403,
                -17712.724132008952,
                -1358.4275738223103,
            );
            let initial_q = Quaternion::new(
                -0.18473353039839202,
                0.7308976443576298,
                -0.5467873085093476,
                0.3642606147693126,
            );
            let initial_b = UnitQuaternion::from_quaternion(initial_q) * b_inertial;
            let mut omega = RK4Solver::new(BodyVector::from(nalgebra::SVector::<f64, 3>::new(
                0.01, 0.005, 0.007,
            )));
            let mut attitude = RK4Solver::new(initial_q);
            let mut state = EclipseDirectionState {
                mag_dir: Direction::from_dir(UnitVector3::new_normalize(initial_b)),
                mag_norm: initial_b.norm(),
                gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
            };
            for _i in 0..100 {
                time += dt;
                let omega_f = |_, _, angular_velocity: BodyVector| {
                    let input_torque = Vector3::zeros();

                    let gyroscopic_torque = Vector3::from(angular_velocity.clone())
                        .cross(&(inertia * Vector3::from(angular_velocity.clone())));

                    let angular_acceleration =
                        inertia.try_inverse().unwrap() * (input_torque - gyroscopic_torque);

                    BodyVector::from(angular_acceleration)
                };
                omega.propagate(omega_f, dt, time);

                let q_f = |phase: RK4Phase, _, q| {
                    let omega_body = Vector3::from(omega.get(phase).unwrap());
                    let omega_quat = Quaternion::from_parts(0.0, omega_body);
                    0.5 * omega_quat * q
                };
                attitude.propagate(q_f, dt, time);
                omega.clear();
                attitude.clear();

                // Compare with small angle approximation
                let propagation_model = EclipseDirectionPropagationModel;
                let g_input = DirectionInput {
                    angular_velocity: omega.get_now().into(),
                };
                state = propagation_model.propagate(
                    &state,
                    &EmptyInput,
                    &g_input,
                    &crate::Time::from_seconds(time),
                    &dt,
                );
            }

            let rk4_mag_field = UnitQuaternion::from_quaternion(attitude.get_now()) * b_inertial;

            let ukf_mag_field_dir = state.mag_dir.dir();
            let ukf_mag_field_norm = state.mag_norm;

            println!();
            println!("Propagated mag dir  : {}", ukf_mag_field_dir.into_inner());
            println!("Propagated mag norm : {}", ukf_mag_field_norm);
            println!("RK4 mag field dir   : {}", rk4_mag_field.normalize());
            println!("RK4 mag field norm  : {}", rk4_mag_field.norm());
            assert!(
                ukf_mag_field_dir.angle(&rk4_mag_field).abs() < 1e-6,
                "RK4 mag field and UKF propagated mag field should match: RK4 error = {}",
                ukf_mag_field_dir.angle(&rk4_mag_field).abs(),
            );
        }

        #[test]
        fn test_ukf_propagation() {
            use control_system::components::GaussianNominalType;
            use control_system::components::GaussianValueType;
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let initial_mag_field_variance = Matrix3::identity() * 1.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );
            let state_mag_dir_nominal = estimator.ukf.state().mag_dir.clone();
            let cov_before = estimator.ukf.covariance().trace();

            // Propagate with zero angular velocity
            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::new(0.1, 0.2, 0.3),
                angular_velocity_covariance: Matrix3::identity() * 0.01,
            };
            let process_noise_covariance =
                Some(eclipse_direction_estimation_process_noise_covariance());
            let dt = 0.1;
            let time = initial_time + dt;
            let mut rng = rand::thread_rng();
            let omega_mvn = MultivariateNormal::new_from_nalgebra(
                omega.angular_velocity,
                omega.angular_velocity_covariance,
            )
            .unwrap();
            let state_mag_dir_error_mvn = MultivariateNormal::new_from_nalgebra(
                Vector2::zeros(),
                estimator
                    .ukf
                    .covariance()
                    .fixed_view::<2, 2>(0, 0)
                    .clone_owned(),
            )
            .unwrap();
            let state_gyro_bias_mvn = MultivariateNormal::new_from_nalgebra(
                estimator.ukf.state().gyro_bias.clone_owned(),
                estimator
                    .ukf
                    .covariance()
                    .fixed_view::<3, 3>(3, 3)
                    .clone_owned(),
            )
            .unwrap();
            let shifted_mag_dir_nominal = {
                let omega_est = omega.angular_velocity - estimator.ukf.state().gyro_bias;
                let axisangle = omega_est * dt;
                let rot = nalgebra::Rotation3::new(axisangle);
                rot * estimator.ukf.state().mag_dir.clone()
            };
            let mut shifted_mag_dir_list = Vec::new();
            // Monte Carlo expected value
            for _ in 0..1000 {
                let angular_velocity_sample = omega_mvn.sample(&mut rng);
                let state_mag_dir_error_sample = state_mag_dir_error_mvn.sample(&mut rng);

                let state_mag_dir_sample =
                    state_mag_dir_nominal.merge_sigma(&state_mag_dir_error_sample);

                let state_gyro_bias_sample = state_gyro_bias_mvn.sample(&mut rng);
                let omega_est = angular_velocity_sample - state_gyro_bias_sample;
                let axisangle = omega_est * dt;
                let rot = nalgebra::Rotation3::new(axisangle);
                let new_magnetic_dir = rot * state_mag_dir_sample;
                shifted_mag_dir_list.push(new_magnetic_dir);
            }
            let expected_mag_dir = {
                let mean_vec = shifted_mag_dir_list
                    .iter()
                    .fold(Vector2::zeros(), |acc, dir| {
                        acc + dir.error(&shifted_mag_dir_nominal)
                    })
                    / (shifted_mag_dir_list.len() as f64);
                shifted_mag_dir_nominal.merge_sigma(&mean_vec)
            };
            estimator
                .ukf
                .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                .unwrap();

            let state_after = estimator.ukf.state();
            let cov_after = estimator.ukf.covariance().trace();
            println!();
            println!(
                "Expected mag dir after propagation: {:?}, UKF mag field: {:?}, error: {}",
                expected_mag_dir.dir(),
                state_after.mag_dir.dir(),
                (state_after.mag_dir.dir().angle(&expected_mag_dir.dir())).abs()
            );
            println!(
                "Nominal propagated mag dir: {:?}",
                shifted_mag_dir_nominal.dir()
            );

            // Covariance should increase due to process noise
            assert!(
                (state_after.mag_dir.dir().angle(&expected_mag_dir.dir())).abs() < 1e-2,
                "Magnetic field expected: {:?}, got: {:?}, error: {}",
                expected_mag_dir.dir(),
                state_after.mag_dir.dir(),
                (state_after.mag_dir.dir().angle(&expected_mag_dir.dir())).abs()
            );
            assert!(
                cov_after >= cov_before,
                "Covariance should increase: {} -> {}",
                cov_before,
                cov_after
            );
        }

        #[test]
        fn test_propagation_increases_covariance() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 50.0, -30.0);
            let initial_mag_field_variance = Matrix3::identity() * 1.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );

            let cov_before = estimator.ukf.covariance().trace();

            // Propagate with non-zero angular velocity and process noise
            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::zeros(),
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(eclipse_direction_estimation_process_noise_covariance());
            let time = initial_time + 0.1;
            estimator
                .ukf
                .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                .unwrap();

            let cov_after = estimator.ukf.covariance().trace();

            // Covariance trace should increase due to process noise
            assert!(
                cov_after > cov_before,
                "Covariance should increase after propagation: {} -> {}",
                cov_before,
                cov_after
            );
        }

        #[test]
        fn test_magnetic_field_dir_rotation_consistency_with_rk4() {
            let inertia = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let initial_mag_field_variance = Matrix3::identity() * 1.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );
            let total_rotation = std::f64::consts::PI / 2.0; // 90 degrees total
            let dt = 0.01;
            let total_steps = 100;
            let angular_velocity = nalgebra::SVector::<f64, 3>::new(
                0.0,
                0.0,
                total_rotation / (dt * total_steps as f64),
            ); // rad/s around z-axis
            println!(
                "Angular velocity for 90 deg rotation in {} steps: {}",
                total_steps, angular_velocity
            );

            let initial_q = Quaternion::identity();
            let mut omega = RK4Solver::new(BodyVector::from(angular_velocity));
            let mut attitude = RK4Solver::new(initial_q);

            let steps = 100;

            let mut time = initial_time;
            for _ in 0..steps {
                time += dt;
                let omega_f = |_, _, angular_velocity: BodyVector| {
                    let input_torque = Vector3::zeros();

                    let gyroscopic_torque = Vector3::from(angular_velocity.clone())
                        .cross(&(inertia * Vector3::from(angular_velocity.clone())));

                    let angular_acceleration =
                        inertia.try_inverse().unwrap() * (input_torque - gyroscopic_torque);

                    BodyVector::from(angular_acceleration)
                };
                omega.propagate(omega_f, dt, time);

                let q_f = |phase: RK4Phase, _, q| {
                    let omega_body = Vector3::from(omega.get(phase).unwrap());
                    let omega_quat = Quaternion::from_parts(0.0, omega_body);
                    0.5 * omega_quat * q
                };
                attitude.propagate(q_f, dt, time);
                omega.clear();
                attitude.clear();

                let omega_input = DirectionInputGaussian {
                    angular_velocity,
                    angular_velocity_covariance: Matrix3::identity()
                        * GYRO_NOISE_STD
                        * GYRO_NOISE_STD,
                };

                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega_input, None, &time)
                    .unwrap();
            }

            let state = estimator.ukf.state();
            // After 90 degrees rotation around z-axis, [100,0,0] should become [0,100,0]
            let state_dir = state.mag_dir.dir();
            let expected_dir = UnitVector3::new_normalize(
                UnitQuaternion::from_quaternion(attitude.get_now()) * initial_mag_field,
            );
            println!();
            println!(
                "Final magnetic field direction: {:?}, expected: {:?}",
                state_dir, expected_dir
            );
            println!("Direction error (rad): {}", state_dir.angle(&expected_dir));
            assert!(
                (state_dir.angle(&expected_dir)).abs() < 1e-2,
                "Magnetic field rotation failed: expected {:?}, got {:?}",
                expected_dir,
                state_dir
            );
        }
    }

    // ========== Update Tests ==========
    mod update {
        use crate::constants::MAGNETOMETER_NOISE_STD;

        use super::*;
        impl EclipseDirectionEstimation {
            fn new_for_direct(
                initial_time: &crate::Time,
                initial_mag_dir: UnitVector3<f64>,
                initial_mag_dir_variance: Matrix2<f64>,
                initial_mag_norm: f64,
                initial_mag_norm_variance: f64,
            ) -> Self {
                let initial_state = EclipseDirectionState {
                    mag_dir: Direction::from_dir(initial_mag_dir),
                    mag_norm: initial_mag_norm,
                    gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
                };
                let mut initial_covariance = SMatrix::<f64, 6, 6>::zeros();
                initial_covariance
                    .fixed_view_mut::<2, 2>(0, 0)
                    .copy_from(&initial_mag_dir_variance);
                initial_covariance
                    .fixed_view_mut::<1, 1>(2, 2)
                    .fill(initial_mag_norm_variance);
                initial_covariance
                    .fixed_view_mut::<3, 3>(3, 3)
                    .copy_from(&(Matrix3::identity() * 0.1)); // Initial gyro bias variance

                let propagation_model = EclipseDirectionPropagationModel;

                let ukf = UnscentedKalmanFilter::new(
                    propagation_model,
                    initial_state,
                    initial_covariance,
                    initial_time,
                    UKFParameters::default(),
                );

                EclipseDirectionEstimation {
                    ukf,
                    mag_obs: MagFieldObservationModelForEclipse,
                }
            }
        }

        #[test]
        fn test_ukf_magnetometer_update_reduces_error_and_covariance() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(0.0, 100.0, 0.0);
            let mut estimator = EclipseDirectionEstimation::new_for_direct(
                &initial_time,
                UnitVector3::new_normalize(initial_mag_field),
                Matrix2::identity() * 1.0,
                initial_mag_field.norm(),
                1.0,
            );

            let time = initial_time;

            let cov_before = *estimator.ukf.covariance();

            // Apply magnetometer update
            let b_true = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let b_measured = MagFieldObservationForEclipse { mag_field: b_true };
            let observation_model = MagFieldObservationModelForEclipse;

            estimator
                .ukf
                .update(
                    &observation_model,
                    &b_measured,
                    &EmptyInput,
                    &EmptyInput,
                    &time,
                    Matrix3::identity() * MAGNETOMETER_NOISE_STD * MAGNETOMETER_NOISE_STD,
                )
                .unwrap();

            let cov_after = estimator.ukf.covariance();

            // Magnetic field error and covariance should decrease
            println!("cov before update: {}", cov_before);
            println!("cov after update: {}", cov_after);
            println!(
                "error after update: {}",
                estimator.ukf.state().mag_dir.dir().angle(&b_true)
            );
            // 球面上に沿った分布になるので、sinθ ≈ θ として計算されてしまうので、それを前提に評価
            assert!(
                (estimator.ukf.state().mag_dir.dir().angle(&b_true)
                    - (core::f64::consts::PI / 2.0 - 1.0))
                    .abs()
                    < 1e-2,
                "Magnetic field dir estimate should be close to true value after update: {:?} vs {:?}",
                estimator.ukf.state().mag_dir.dir(),
                UnitVector3::new_normalize(b_true)
            );
            assert!(
                (estimator.ukf.state().mag_norm - b_true.norm()).abs() < 1e-2,
                "Magnetic field norm estimate should be close to true value after update: {} vs {}",
                estimator.ukf.state().mag_norm,
                b_true.norm()
            );
            assert!(
                cov_after.fixed_view::<3, 3>(0, 0).trace()
                    < cov_before.fixed_view::<3, 3>(0, 0).trace(),
                "Magnetic field covariance should decrease after update: {} -> {}",
                cov_before,
                cov_after
            );
        }
    }

    // ========== Convergence Tests ==========
    mod convergence {
        use crate::constants::MAGNETOMETER_NOISE_STD;

        use super::*;

        #[test]
        fn test_ukf_converges_to_true_directions() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(1.0e3, 0.0, 0.0);
            let initial_mag_field_variance = Matrix3::identity() * 10.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );

            // True magnetic field (body frame)
            let inertial_b = nalgebra::SVector::<f64, 3>::new(0.0, 0.8, -0.0) * 1e3;
            let mut true_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);

            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::new(0.1, 0.2, 0.3),
                angular_velocity_covariance: Matrix3::identity() * 0.001,
            };
            let process_noise_covariance =
                Some(eclipse_direction_estimation_process_noise_covariance());
            let mut time = initial_time;
            let dt = 0.1;

            // Run multiple propagation and update cycles
            for _ in 0..1000 {
                time += dt;

                let delta_theta = omega.angular_velocity * dt;
                true_q = nalgebra::UnitQuaternion::from_scaled_axis(delta_theta) * true_q;

                let true_b = true_q * inertial_b;

                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                    .unwrap();

                // Apply magnetometer update
                let b_measured = MagFieldObservationForEclipse { mag_field: true_b };
                let observation_model = MagFieldObservationModelForEclipse;

                estimator
                    .ukf
                    .update(
                        &observation_model,
                        &b_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time,
                        Matrix3::identity() * MAGNETOMETER_NOISE_STD * MAGNETOMETER_NOISE_STD,
                    )
                    .unwrap();
            }

            let final_state = estimator.ukf.state();

            println!();
            println!("estimated bias: {:?}", final_state.gyro_bias);

            // Check magnetic field convergence (Eclipse mode: mag-only, slower convergence)
            let true_b = true_q * inertial_b;
            let true_b_dir = UnitVector3::new_normalize(true_b);
            let dir_error = (final_state.mag_dir.dir().angle(&true_b_dir)).abs();
            let norm_error = (final_state.mag_norm - true_b.norm()).abs();
            assert!(
                dir_error < 1e-2,
                "Magnetic field estimate should converge: error = {}",
                dir_error
            );
            assert!(
                norm_error < 1.0,
                "Magnetic field norm estimate should converge: error = {}",
                norm_error
            );
            assert!(
                estimator.ukf.covariance().fixed_view::<2, 2>(0, 0).trace() < 1e-2,
                "Magnetic field covariance should be small after convergence"
            );
        }

        #[test]
        fn test_gyro_bias_estimation_converges() {
            let initial_time = crate::Time::from_seconds(0.0);

            // Inertial frame magnetic field (fixed)
            let inertial_b = nalgebra::SVector::<f64, 3>::new(0.3, 0.1, -0.2) * 1.0e3;

            // Initial attitude
            let initial_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
            let mut true_q = initial_q;

            let true_bias = nalgebra::SVector::<f64, 3>::new(0.05, -0.03, 0.01);
            let true_omega = nalgebra::SVector::<f64, 3>::new(0.1, 0.2, 0.05);
            let omega = DirectionInputGaussian {
                angular_velocity: true_omega + true_bias,
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(eclipse_direction_estimation_process_noise_covariance());
            let mut time = initial_time;
            let dt = 0.02;

            let initial_mag_field = initial_q * inertial_b;
            let initial_mag_field_variance = Matrix3::identity() * 1.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );

            // Run multiple propagation and update cycles
            for _ in 0..1000 {
                time += dt;

                // Update true attitude
                let delta_theta = true_omega * dt;
                let delta_q = nalgebra::UnitQuaternion::from_scaled_axis(delta_theta);
                true_q = delta_q * true_q;

                // Transform inertial magnetic field to body frame
                let true_b_body = true_q * inertial_b;

                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                    .unwrap();

                // Apply magnetometer update
                let b_measured = MagFieldObservationForEclipse {
                    mag_field: true_b_body,
                };
                let observation_model = MagFieldObservationModelForEclipse;

                estimator
                    .ukf
                    .update(
                        &observation_model,
                        &b_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time,
                        Matrix3::identity() * GYRO_BIAS_DRIFT_STD * GYRO_BIAS_DRIFT_STD,
                    )
                    .unwrap();
            }

            let estimated_bias = estimator.ukf.state().gyro_bias;
            println!();
            println!("Estimated gyro bias: {:?}", estimated_bias);
            println!("True gyro bias:      {:?}", true_bias);
            let bias_error = (estimated_bias - true_bias).norm();

            assert!(
                bias_error < 0.002,
                "Gyro bias error should be small: {} (estimated: {:?}, true: {:?})",
                bias_error,
                estimated_bias,
                true_bias
            );
        }
    }

    // ========== Robustness Tests ==========
    mod robustness {
        use super::*;

        #[test]
        fn test_covariance_remains_positive_definite() {
            let initial_time = crate::Time::from_seconds(0.0);

            // True magnetic field (body frame)
            let inertial_b = nalgebra::SVector::<f64, 3>::new(0.3, 0.1, -0.2) * 1.0e3;
            let true_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
            let mut true_b = true_q * inertial_b;

            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::new(0.1, 0.0, 0.0),
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(eclipse_direction_estimation_process_noise_covariance());

            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(1000.0, 0.0, 0.0);
            let initial_mag_field_variance = Matrix3::identity() * 1.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );
            let mut time = initial_time;
            let dt = 0.02;

            // Run many propagation and update cycles
            for i in 0..100 {
                time += dt;

                let delta_theta = omega.angular_velocity * dt;

                true_b = nalgebra::UnitQuaternion::from_scaled_axis(delta_theta) * true_b;

                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                    .unwrap();

                // Occasional measurements
                if i % 5 == 0 {
                    let b_measured = MagFieldObservationForEclipse { mag_field: true_b };
                    let observation_model = MagFieldObservationModelForEclipse;

                    estimator
                        .ukf
                        .update(
                            &observation_model,
                            &b_measured,
                            &EmptyInput,
                            &EmptyInput,
                            &time,
                            Matrix3::identity() * 0.001,
                        )
                        .unwrap();
                };

                let cov = estimator.ukf.covariance();

                // Check symmetry
                let symmetry_error = (cov - cov.transpose()).norm();
                assert!(
                    symmetry_error < 1e-10,
                    "Covariance should be symmetric at iteration {}",
                    i
                );

                // Check positive definiteness
                let eigenvalues = cov.symmetric_eigenvalues();
                for (j, eig) in eigenvalues.iter().enumerate() {
                    assert!(
                        eig > &0.0,
                        "Covariance eigenvalue {} should be positive at iteration {}: {}",
                        j,
                        i,
                        eig
                    );
                }
            }
        }

        #[test]
        fn test_handles_measurement_outliers() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 50.0, -30.0);
            let initial_mag_field_variance = Matrix3::identity() * 1.0;
            let mut estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );

            // Apply outlier measurement (very different from expected)
            let outlier_measurement = nalgebra::SVector::<f64, 3>::new(-1000.0, 2000.0, -500.0);
            let b_measured = MagFieldObservationForEclipse {
                mag_field: outlier_measurement,
            };

            // Use large measurement noise to reduce impact
            estimator
                .ukf
                .update(
                    &MagFieldObservationModelForEclipse,
                    &b_measured,
                    &EmptyInput,
                    &EmptyInput,
                    &initial_time,
                    Matrix3::identity() * 10000.0, // Large measurement uncertainty
                )
                .unwrap();

            // State should not deviate too much due to large measurement noise
            let state = estimator.ukf.state();
            let initial_b_dir = UnitVector3::new_normalize(initial_mag_field);
            let deviation = state.mag_dir.dir().angle(&initial_b_dir).abs();
            assert!(
                deviation < 0.5,
                "State should not deviate too much from initial with high measurement noise: {}",
                deviation
            );
        }
    }
}

#[cfg(test)]
mod tests_full {
    use super::*;
    use nalgebra::{Matrix3, UnitVector3, Vector3};

    impl FullDirectionEstimation {
        fn new_for_test(
            initial_time: &crate::Time,
            mag_dir: Direction,
            mag_dir_variance: Matrix2<f64>,
            mag_norm: f64,
            mag_norm_variance: f64,
            sun_direction: Direction,
            sun_direction_variance: Matrix2<f64>,
        ) -> Self {
            let state = FullDirectionState {
                mag_dir,
                mag_norm,
                sun_direction,
                gyro_bias: SVector::<f64, 3>::zeros(),
            };
            let ukf = UnscentedKalmanFilter::new(
                FullDirectionPropagationModel,
                state,
                {
                    let mut cov = SMatrix::<f64, 8, 8>::zeros();
                    // Magnetic field initial covariance (unknown until observed)
                    cov.fixed_view_mut::<2, 2>(0, 0)
                        .copy_from(&mag_dir_variance);
                    cov[(2, 2)] = mag_norm_variance;
                    // Sun direction initial covariance (from parameter)
                    cov.fixed_view_mut::<2, 2>(3, 3)
                        .copy_from(&sun_direction_variance);
                    // Gyro bias initial covariance (reduced to match attitude_determination)
                    cov.fixed_view_mut::<3, 3>(5, 5)
                        .copy_from(&(SMatrix::<f64, 3, 3>::identity() * 1.0e1));
                    cov
                },
                initial_time,
                UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
            );
            FullDirectionEstimation {
                ukf,
                mag_obs: MagFieldObservationModelForFull,
                sun_obs: SunDirectionObservationModel,
            }
        }
    }

    impl FullDirectionEstimation {
        fn new_for_direct(
            initial_time: &crate::Time,
            initial_mag_dir: UnitVector3<f64>,
            initial_mag_dir_variance: Matrix2<f64>,
            initial_mag_norm: f64,
            initial_mag_norm_variance: f64,
            initial_sun_direction: UnitVector3<f64>,
            initial_sun_direction_variance: Matrix2<f64>,
        ) -> Self {
            let state = FullDirectionState {
                mag_dir: Direction::from_dir(initial_mag_dir),
                mag_norm: initial_mag_norm,
                sun_direction: Direction::from_dir(initial_sun_direction),
                gyro_bias: SVector::<f64, 3>::zeros(),
            };
            let ukf = UnscentedKalmanFilter::new(
                FullDirectionPropagationModel,
                state,
                {
                    let mut cov = SMatrix::<f64, 8, 8>::zeros();
                    // Magnetic field initial covariance (unknown until observed)
                    cov.fixed_view_mut::<2, 2>(0, 0)
                        .copy_from(&initial_mag_dir_variance);
                    cov[(2, 2)] = initial_mag_norm_variance;
                    // Sun direction initial covariance (from parameter)
                    cov.fixed_view_mut::<2, 2>(3, 3)
                        .copy_from(&initial_sun_direction_variance);
                    // Gyro bias initial covariance (reduced to match attitude_determination)
                    cov.fixed_view_mut::<3, 3>(5, 5)
                        .copy_from(&(SMatrix::<f64, 3, 3>::identity() * 1.0e1));
                    cov
                },
                initial_time,
                UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
            );
            FullDirectionEstimation {
                ukf,
                mag_obs: MagFieldObservationModelForFull,
                sun_obs: SunDirectionObservationModel,
            }
        }
    }

    // ========== Initialization Tests ==========
    mod initialization {
        use super::*;

        #[test]
        fn test_full_direction_state_initialization() {
            let initial_time = crate::Time::from_seconds(0.0);
            let mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 50.0, -30.0);
            let initial_mag_direction = UnitVector3::new_normalize(mag_field);
            let initial_mag_dir_variance = Matrix2::identity() * 1.0;
            let initial_mag_norm = mag_field.norm();
            let initial_mag_norm_variance = 1.0;
            let sun_direction = Direction::from_dir(UnitVector3::new_normalize(Vector3::y()));
            let sun_direction_variance = Matrix2::identity() * 0.01;

            let estimator = FullDirectionEstimation::new_for_test(
                &initial_time,
                Direction::from_dir(initial_mag_direction),
                initial_mag_dir_variance,
                initial_mag_norm,
                initial_mag_norm_variance,
                sun_direction.clone(),
                sun_direction_variance,
            );

            assert!(
                (estimator.ukf.state().mag_dir.dir().into_inner()
                    - initial_mag_direction.into_inner())
                .norm()
                    < 1e-10,
                "Magnetic field direction should match initialization value"
            );
            assert!(
                (estimator.ukf.state().mag_norm - initial_mag_norm).abs() < 1e-10,
                "Magnetic field norm should match initialization value"
            );

            let sun_error = estimator
                .ukf
                .state()
                .sun_direction
                .dir()
                .angle(&sun_direction.dir());
            assert!(
                sun_error < 1e-10,
                "Sun direction should match initialization value"
            );
        }

        #[test]
        fn test_mode_transition_from_eclipse() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let eclipse_estimator =
                EclipseDirectionEstimation::new(&initial_time, initial_mag, Matrix3::identity());

            let eclipse_cov = eclipse_estimator.ukf.covariance();

            let sun_direction = Direction::from_dir(UnitVector3::new_normalize(Vector3::z()));
            let sun_variance = Matrix2::identity() * 0.1;

            let full_estimator =
                eclipse_estimator.to_full_mode(&initial_time, sun_direction, sun_variance);

            // Check state transfer
            assert!(
                (full_estimator.ukf.state().mag_dir.dir().into_inner() - initial_mag.normalize())
                    .norm()
                    < 1e-10,
                "Magnetic field direction should transfer from eclipse mode"
            );
            assert!(
                (full_estimator.ukf.state().mag_norm - initial_mag.norm()).abs() < 1e-1,
                "Magnetic field norm should transfer from eclipse mode, got {}, expected {}",
                full_estimator.ukf.state().mag_norm,
                initial_mag.norm()
            );

            // Check covariance transfer
            let mag_cov = full_estimator.ukf.covariance().fixed_view::<3, 3>(0, 0);
            let bias_cov = full_estimator.ukf.covariance().fixed_view::<3, 3>(5, 5);
            assert!(
                (mag_cov - eclipse_cov.fixed_view::<3, 3>(0, 0)).norm() < 1e-10,
                "Magnetic field covariance should transfer from eclipse mode"
            );
            assert!(
                (bias_cov - eclipse_cov.fixed_view::<3, 3>(3, 3)).norm() < 1e-10,
                "Gyro bias covariance should transfer from eclipse mode"
            );
        }
    }

    mod propagation {
        use astrodynamics::coordinate::BodyVector;
        use control_system::integrator::{
            Prediction, TimeIntegrator,
            rk4::{RK4Phase, RK4Solver},
        };
        use nalgebra::{Quaternion, UnitQuaternion, Vector2};
        use rand::prelude::Distribution;
        use statrs::distribution::MultivariateNormal;

        use crate::constants::{MAGNETOMETER_NOISE_STD, SUN_SENSOR_NOISE_STD};

        use super::*;
        #[test]
        fn test_ukf_propagation_increases_covariance() {
            let initial_time = crate::Time::from_seconds(0.0);
            let mag_dir = Direction::from_dir(Vector3::x_axis());
            let mag_dir_variance = Matrix2::identity() * 1.0;
            let mag_norm = 100.0;
            let mag_norm_variance = 1.0;
            let sun_direction = Direction::from_dir(UnitVector3::new_normalize(Vector3::x()));
            let sun_direction_variance = Matrix2::identity() * 1.0e-3;
            let mut estimator = FullDirectionEstimation::new_for_test(
                &initial_time,
                mag_dir,
                mag_dir_variance,
                mag_norm,
                mag_norm_variance,
                sun_direction.clone(),
                sun_direction_variance,
            );

            let cov_before = estimator.ukf.covariance().trace();

            // Propagate with zero angular velocity
            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::zeros(),
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(full_direction_estimation_process_noise_covariance());
            let time = initial_time + 0.1;
            estimator
                .ukf
                .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                .unwrap();

            let cov_after = estimator.ukf.covariance().trace();

            // Covariance should increase due to process noise
            assert!(
                cov_after >= cov_before,
                "Covariance should increase due to process noise: {} -> {}",
                cov_before,
                cov_after
            );
        }
        #[test]
        fn test_propagation_model_with_rk4_isotropic() {
            let mut time = 0.0;
            let dt = 0.01;
            let inertia = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
            let b_inertial = nalgebra::SVector::<f64, 3>::new(
                9894.151141285403,
                -17712.724132008952,
                -1358.4275738223103,
            );
            let s_inertial = UnitVector3::new_normalize(nalgebra::SVector::<f64, 3>::new(
                -0.9145143032379116,
                -0.4033661855532052,
                0.03563167563588264,
            ));
            let initial_q = Quaternion::new(
                -0.18473353039839202,
                0.7308976443576298,
                -0.5467873085093476,
                0.3642606147693126,
            );
            let initial_b = UnitQuaternion::from_quaternion(initial_q) * b_inertial;
            let initial_s = UnitQuaternion::from_quaternion(initial_q) * s_inertial;
            let mut omega = RK4Solver::new(BodyVector::from(nalgebra::SVector::<f64, 3>::new(
                0.01, 0.005, 0.007,
            )));
            let mut attitude = RK4Solver::new(initial_q);
            let mut state = FullDirectionState {
                mag_dir: Direction::from_dir(UnitVector3::new_normalize(initial_b)),
                mag_norm: initial_b.norm(),
                sun_direction: Direction::from_dir(initial_s),
                gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
            };
            for _i in 0..100 {
                time += dt;
                let omega_f = |_, _, angular_velocity: BodyVector| {
                    let input_torque = Vector3::zeros();

                    let gyroscopic_torque = Vector3::from(angular_velocity.clone())
                        .cross(&(inertia * Vector3::from(angular_velocity.clone())));

                    let angular_acceleration =
                        inertia.try_inverse().unwrap() * (input_torque - gyroscopic_torque);

                    BodyVector::from(angular_acceleration)
                };
                omega.propagate(omega_f, dt, time);

                let q_f = |phase: RK4Phase, _, q| {
                    let omega_body = Vector3::from(omega.get(phase).unwrap());
                    let omega_quat = Quaternion::from_parts(0.0, omega_body);
                    0.5 * omega_quat * q
                };
                attitude.propagate(q_f, dt, time);
                omega.clear();
                attitude.clear();

                // Compare with small angle approximation
                let propagation_model = FullDirectionPropagationModel;
                let g_input = DirectionInput {
                    angular_velocity: omega.get_now().into(),
                };
                state = propagation_model.propagate(
                    &state,
                    &EmptyInput,
                    &g_input,
                    &crate::Time::from_seconds(time),
                    &dt,
                );
            }
            let rk4_final_b = UnitQuaternion::from_quaternion(attitude.get_now()) * b_inertial;
            let rk4_final_s = UnitQuaternion::from_quaternion(attitude.get_now()) * s_inertial;
            let state_b_dir = state.mag_dir.dir();
            let state_b_norm = state.mag_norm;
            let state_s_dir = state.sun_direction.dir();

            let b_dir_error = (state_b_dir.angle(&UnitVector3::new_normalize(rk4_final_b))).abs();
            let b_norm_error = (state_b_norm - rk4_final_b.norm()).abs();
            let s_dir_error = (state_s_dir.angle(&rk4_final_s)).abs();
            assert!(
                b_dir_error < 1e-5,
                "Magnetic field direction propagation error too large: {}",
                b_dir_error
            );
            assert!(
                b_norm_error < 1e-1,
                "Magnetic field norm propagation error too large: {}",
                b_norm_error
            );
            assert!(
                s_dir_error < 1e-5,
                "Sun direction propagation error too large: {}",
                s_dir_error
            );
        }

        #[test]
        fn test_ukf_propagation() {
            use control_system::components::GaussianNominalType;
            use control_system::components::GaussianValueType;
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let initial_mag_field_variance =
                Matrix3::identity() * MAGNETOMETER_NOISE_STD * MAGNETOMETER_NOISE_STD;
            let initial_sun_direction =
                Direction::from_dir(UnitVector3::new_normalize(Vector3::y()));
            let initial_sun_direction_variance =
                Matrix2::identity() * SUN_SENSOR_NOISE_STD * SUN_SENSOR_NOISE_STD;
            let eclipse_estimator = EclipseDirectionEstimation::new(
                &initial_time,
                initial_mag_field,
                initial_mag_field_variance,
            );
            let mut estimator = eclipse_estimator.to_full_mode(
                &initial_time,
                initial_sun_direction,
                initial_sun_direction_variance,
            );
            let state_mag_dir_nominal = estimator.ukf.state().mag_dir.clone();
            let state_sun_dir_nominal = estimator.ukf.state().sun_direction.clone();
            let cov_before = estimator.ukf.covariance().trace();

            // Propagate with zero angular velocity
            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::new(0.1, 0.2, 0.3),
                angular_velocity_covariance: Matrix3::identity() * 0.01,
            };
            let process_noise_covariance =
                Some(full_direction_estimation_process_noise_covariance());
            let dt = 0.1;
            let time = initial_time + dt;
            let mut rng = rand::thread_rng();
            let omega_mvn = MultivariateNormal::new_from_nalgebra(
                omega.angular_velocity,
                omega.angular_velocity_covariance,
            )
            .unwrap();
            let state_mag_dir_error_mvn = MultivariateNormal::new_from_nalgebra(
                Vector2::zeros(),
                estimator
                    .ukf
                    .covariance()
                    .fixed_view::<2, 2>(0, 0)
                    .clone_owned(),
            )
            .unwrap();
            let state_sun_dir_error_mvn = MultivariateNormal::new_from_nalgebra(
                Vector2::zeros(),
                estimator
                    .ukf
                    .covariance()
                    .fixed_view::<2, 2>(3, 3)
                    .clone_owned(),
            )
            .unwrap();
            let state_gyro_bias_mvn = MultivariateNormal::new_from_nalgebra(
                estimator.ukf.state().gyro_bias.clone_owned(),
                estimator
                    .ukf
                    .covariance()
                    .fixed_view::<3, 3>(3, 3)
                    .clone_owned(),
            )
            .unwrap();
            let shifted_mag_dir_nominal = {
                let omega_est = omega.angular_velocity - estimator.ukf.state().gyro_bias;
                let axisangle = omega_est * dt;
                let rot = nalgebra::Rotation3::new(axisangle);
                rot * estimator.ukf.state().mag_dir.clone()
            };
            let mut shifted_mag_dir_list = Vec::new();
            let mut shifted_sun_dir_list = Vec::new();
            // Monte Carlo expected value
            for _ in 0..1000 {
                let angular_velocity_sample = omega_mvn.sample(&mut rng);
                let state_mag_dir_error_sample = state_mag_dir_error_mvn.sample(&mut rng);
                let state_sun_dir_error_sample = state_sun_dir_error_mvn.sample(&mut rng);

                let state_mag_dir_sample =
                    state_mag_dir_nominal.merge_sigma(&state_mag_dir_error_sample);
                let state_sun_dir_sample =
                    state_sun_dir_nominal.merge_sigma(&state_sun_dir_error_sample);

                let state_gyro_bias_sample = state_gyro_bias_mvn.sample(&mut rng);
                let omega_est = angular_velocity_sample - state_gyro_bias_sample;
                let axisangle = omega_est * dt;
                let rot = nalgebra::Rotation3::new(axisangle);
                let new_magnetic_dir = rot * state_mag_dir_sample;
                let new_sun_dir = rot * state_sun_dir_sample;
                shifted_mag_dir_list.push(new_magnetic_dir);
                shifted_sun_dir_list.push(new_sun_dir);
            }
            let expected_mag_dir = {
                let mean_vec = shifted_mag_dir_list
                    .iter()
                    .fold(Vector2::zeros(), |acc, dir| {
                        acc + dir.error(&shifted_mag_dir_nominal)
                    })
                    / (shifted_mag_dir_list.len() as f64);
                shifted_mag_dir_nominal.merge_sigma(&mean_vec)
            };
            let expected_sun_dir = {
                let mean_vec = shifted_sun_dir_list
                    .iter()
                    .fold(Vector2::zeros(), |acc, dir| {
                        acc + dir.error(&state_sun_dir_nominal)
                    })
                    / (shifted_sun_dir_list.len() as f64);
                state_sun_dir_nominal.merge_sigma(&mean_vec)
            };
            estimator
                .ukf
                .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                .unwrap();

            let state_after = estimator.ukf.state();
            let cov_after = estimator.ukf.covariance().trace();
            println!();
            println!(
                "Expected mag dir after propagation: {:?}, UKF mag field: {:?}, error: {}",
                expected_mag_dir.dir(),
                state_after.mag_dir.dir(),
                (state_after.mag_dir.dir().angle(&expected_mag_dir.dir())).abs()
            );
            println!(
                "Expected sun dir after propagation: {:?}, UKF sun dir: {:?}, error: {}",
                expected_sun_dir.dir(),
                state_after.sun_direction.dir(),
                (state_after
                    .sun_direction
                    .dir()
                    .angle(&expected_sun_dir.dir()))
                .abs()
            );
            println!(
                "Nominal propagated mag dir: {:?}",
                shifted_mag_dir_nominal.dir()
            );

            // Covariance should increase due to process noise
            assert!(
                (state_after.mag_dir.dir().angle(&expected_mag_dir.dir())).abs() < 1e-2,
                "Magnetic field expected: {:?}, got: {:?}, error: {}",
                expected_mag_dir.dir(),
                state_after.mag_dir.dir(),
                (state_after.mag_dir.dir().angle(&expected_mag_dir.dir())).abs()
            );
            assert!(
                (state_after
                    .sun_direction
                    .dir()
                    .angle(&expected_sun_dir.dir()))
                .abs()
                    < 1e-2,
                "Sun direction expected: {:?}, got: {:?}, error: {}",
                expected_sun_dir.dir(),
                state_after.sun_direction.dir(),
                (state_after
                    .sun_direction
                    .dir()
                    .angle(&expected_sun_dir.dir()))
                .abs()
            );
            assert!(
                cov_after >= cov_before,
                "Covariance should increase: {} -> {}",
                cov_before,
                cov_after
            );
        }
    }

    mod update {
        use super::*;

        #[test]
        fn test_ukf_magnetometer_update_reduces_error_and_covariance() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(0.0, 100.0, 0.0);
            let initial_sun_direction = Vector3::y_axis();
            let mut estimator = FullDirectionEstimation::new_for_direct(
                &initial_time,
                UnitVector3::new_normalize(initial_mag_field),
                Matrix2::identity() * 1.0,
                initial_mag_field.norm(),
                1.0,
                initial_sun_direction,
                Matrix2::identity() * 1.0,
            );

            let time = initial_time;

            let cov_before = *estimator.ukf.covariance();

            // Apply magnetometer update
            let b_true = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let b_measured = MagFieldObservationForFull { mag_field: b_true };
            let observation_model = MagFieldObservationModelForFull;

            estimator
                .ukf
                .update(
                    &observation_model,
                    &b_measured,
                    &EmptyInput,
                    &EmptyInput,
                    &time,
                    Matrix3::identity() * 0.000001,
                )
                .unwrap();

            let cov_after = estimator.ukf.covariance();

            // 球面上に沿った分布から三次元ユークリッド空間を予測することになるので、sinθ ≈ θ として計算されてしまう。それを前提に評価
            assert!(
                (estimator.ukf.state().mag_dir.dir().angle(&b_true)
                    - (core::f64::consts::PI / 2.0 - 1.0))
                    .abs()
                    < 1e-2,
                "Magnetic field dir estimate should be close to true value after update: {:?} vs {:?}",
                estimator.ukf.state().mag_dir.dir(),
                UnitVector3::new_normalize(b_true)
            );
            assert!(
                (estimator.ukf.state().mag_norm - b_true.norm()).abs() < 1e-2,
                "Magnetic field norm estimate should be close to true value after update: {} vs {}",
                estimator.ukf.state().mag_norm,
                b_true.norm()
            );
            assert!(
                cov_after.fixed_view::<3, 3>(0, 0).trace()
                    < cov_before.fixed_view::<3, 3>(0, 0).trace(),
                "Magnetic field covariance should decrease after update: {} -> {}",
                cov_before,
                cov_after
            );
        }

        #[test]
        fn test_ukf_sun_sensor_update_reduces_error_and_covariance() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_mag_field = nalgebra::SVector::<f64, 3>::new(0.0, 100.0, 0.0);
            let initial_sun_direction = Vector3::x_axis();
            let mut estimator = FullDirectionEstimation::new_for_direct(
                &initial_time,
                UnitVector3::new_normalize(initial_mag_field),
                Matrix2::identity() * 1.0,
                initial_mag_field.norm(),
                1.0,
                initial_sun_direction,
                Matrix2::identity() * 1.0,
            );

            let time = initial_time;

            let cov_before = *estimator.ukf.covariance();

            // Apply magnetometer update
            let s_true = Vector3::x_axis();
            let s_measured = SunDirectionObservation {
                sun_direction: Direction::from_dir(s_true),
            };
            let observation_model = SunDirectionObservationModel;

            estimator
                .ukf
                .update(
                    &observation_model,
                    &s_measured,
                    &EmptyInput,
                    &EmptyInput,
                    &time,
                    Matrix2::identity() * 0.000001,
                )
                .unwrap();

            let cov_after = estimator.ukf.covariance();

            assert!(
                estimator
                    .ukf
                    .state()
                    .sun_direction
                    .dir()
                    .angle(&s_true)
                    .abs()
                    < 1e-2,
                "Sun direction estimate should be close to true value after update: {:?} vs {:?}, error: {}",
                estimator.ukf.state().sun_direction.dir(),
                s_true,
                estimator.ukf.state().sun_direction.dir().angle(&s_true)
            );
            assert!(
                cov_after.fixed_view::<2, 2>(3, 3).trace()
                    < cov_before.fixed_view::<2, 2>(3, 3).trace(),
                "Sun direction covariance should decrease after update: {} -> {}",
                cov_before,
                cov_after
            );
        }
    }

    mod convergence {
        use super::*;

        #[test]
        fn test_gyro_bias_estimation_converges() {
            // Inertial frame vectors (fixed)
            let inertial_b = nalgebra::SVector::<f64, 3>::new(0.3, 0.1, -0.2) * 1e2;
            let inertial_s = UnitVector3::new_normalize(Vector3::y());

            // Initial attitude
            let initial_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
            let initial_s_body = initial_q * inertial_s;
            let initial_b_body = initial_q * inertial_b;

            let initial_time = crate::Time::from_seconds(0.0);
            let mag_dir_variance = Matrix2::identity() * 1.0;
            let mag_norm_variance = 1.0;
            let sun_direction_variance = Matrix2::identity();
            let mut estimator = FullDirectionEstimation::new_for_test(
                &initial_time,
                Direction::from_dir(UnitVector3::new_normalize(initial_b_body)),
                mag_dir_variance,
                inertial_b.norm(),
                mag_norm_variance,
                Direction::from_dir(initial_s_body),
                sun_direction_variance,
            );

            let true_bias = nalgebra::SVector::<f64, 3>::new(0.05, -0.03, 0.02);
            let true_omega = nalgebra::SVector::<f64, 3>::new(0.1, 0.0, 0.0);
            let omega = DirectionInputGaussian {
                angular_velocity: true_omega + true_bias,
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(full_direction_estimation_process_noise_covariance());
            let mut time = initial_time;
            let dt = 0.02;

            let mut true_q = initial_q;

            // Run multiple propagation and update cycles
            for _ in 0..300 {
                time += dt;

                // Update true attitude
                let delta_theta = true_omega * dt;
                let delta_q = nalgebra::UnitQuaternion::from_scaled_axis(delta_theta);
                true_q = delta_q * true_q;

                // Body frame measurements: rotate inertial vectors by inverse of attitude
                let true_b_body = true_q * inertial_b;
                let true_s_body = true_q * inertial_s;

                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                    .unwrap();

                // Apply magnetometer update
                let b_measured = MagFieldObservationForFull {
                    mag_field: true_b_body,
                };
                let observation_model = MagFieldObservationModelForFull;

                estimator
                    .ukf
                    .update(
                        &observation_model,
                        &b_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time,
                        Matrix3::identity() * 0.001,
                    )
                    .unwrap();

                let s_measured = SunDirectionObservation {
                    sun_direction: Direction::from_dir(true_s_body),
                };
                let sun_observation_model = SunDirectionObservationModel;
                estimator
                    .ukf
                    .update(
                        &sun_observation_model,
                        &s_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time,
                        Matrix2::identity() * 0.001,
                    )
                    .unwrap();
            }

            let estimated_bias = estimator.ukf.state().gyro_bias;
            let bias_error = (estimated_bias - true_bias).norm();

            println!(
                "Estimated gyro bias: {:?}, True gyro bias: {:?}, Error norm: {}",
                estimated_bias, true_bias, bias_error
            );

            assert!(
                bias_error < 1e-3,
                "Gyro bias error should be small: {} (estimated: {:?}, true: {:?})",
                bias_error,
                estimated_bias,
                true_bias
            );
        }

        #[test]
        fn test_ukf_converges_to_true_directions() {
            // Inertial frame references
            let inertial_b = nalgebra::SVector::<f64, 3>::new(0.3, 0.1, -0.2) * 1e2;
            let inertial_s = Direction::from_dir(UnitVector3::new_normalize(Vector3::y()));
            let initial_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
            let mut true_q = initial_q;

            // Initial body frame measurements
            let initial_b_body = true_q * inertial_b;
            let initial_s_body = true_q * inertial_s.dir();

            let initial_time = crate::Time::from_seconds(0.0);
            let mag_dir_variance = Matrix2::identity() * 1.0;
            let mag_norm_variance = 1.0;
            let sun_direction_variance = Matrix2::identity();
            let mut estimator = FullDirectionEstimation::new_for_test(
                &initial_time,
                Direction::from_dir(UnitVector3::new_normalize(initial_b_body)),
                mag_dir_variance,
                inertial_b.norm(),
                mag_norm_variance,
                Direction::from_dir(initial_s_body),
                sun_direction_variance,
            );

            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::new(0.1, 0.0, 0.0),
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(full_direction_estimation_process_noise_covariance());
            let mut time = initial_time;
            let dt = 0.02;

            // Run multiple propagation and update cycles
            for _ in 0..1000 {
                time += dt;

                let delta_theta = omega.angular_velocity * dt;
                let delta_q = nalgebra::UnitQuaternion::from_scaled_axis(delta_theta);

                // Update true attitude
                true_q = delta_q * true_q;

                // Transform inertial vectors to body frame
                let true_b_body = true_q * inertial_b;
                let true_s_body = true_q * inertial_s.dir();

                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                    .unwrap();

                // Apply magnetometer update
                let b_measured = MagFieldObservationForFull {
                    mag_field: true_b_body,
                };
                let observation_model = MagFieldObservationModelForFull;

                estimator
                    .ukf
                    .update(
                        &observation_model,
                        &b_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time,
                        Matrix3::identity() * 0.001,
                    )
                    .unwrap();

                let s_measured = SunDirectionObservation {
                    sun_direction: Direction::from_dir(true_s_body),
                };
                let sun_observation_model = SunDirectionObservationModel;
                estimator
                    .ukf
                    .update(
                        &sun_observation_model,
                        &s_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time,
                        Matrix2::identity() * 0.001,
                    )
                    .unwrap();
            }

            let final_state = estimator.ukf.state();

            // Final true measurements for comparison
            let final_true_b = true_q * inertial_b;
            let final_true_s = true_q * inertial_s.dir();

            // Check magnetic field convergence
            let b_dir_error = (final_state
                .mag_dir
                .dir()
                .angle(&UnitVector3::new_normalize(final_true_b)))
            .abs();
            let b_norm_error = (final_state.mag_norm - final_true_b.norm()).abs();
            assert!(
                b_dir_error < 1e-3,
                "Magnetic field direction estimate should converge: error = {}",
                b_dir_error
            );
            assert!(
                b_norm_error < 1e-1,
                "Magnetic field norm estimate should converge: error = {}",
                b_norm_error
            );
            assert!(
                estimator.ukf.covariance().fixed_view::<3, 3>(0, 0).trace() < 1e-2,
                "Magnetic field covariance should be small after convergence"
            );
            // Check sun direction convergence
            let s_error =
                (final_state.sun_direction.dir().into_inner() - final_true_s.into_inner()).norm();
            assert!(
                s_error < 1e-3,
                "Sun direction estimate should converge: error = {}",
                s_error
            );
            assert!(
                estimator.ukf.covariance().fixed_view::<2, 2>(3, 3).trace() < 1e-2,
                "Sun direction covariance should be small after convergence"
            );
            let bias_error = final_state.gyro_bias.norm();
            assert!(
                bias_error < 1e-3,
                "Gyro bias estimate should converge to zero: error = {}",
                bias_error
            );
            assert!(
                estimator.ukf.covariance().fixed_view::<3, 3>(5, 5).trace() < 1e-2,
                "Gyro bias covariance should be small after convergence"
            );
        }
    }

    mod robustness {
        use super::*;
        #[test]
        fn test_covariance_remains_positive_definite() {
            let initial_time = crate::Time::from_seconds(0.0);
            let inertial_b = nalgebra::SVector::<f64, 3>::new(0.3, 0.1, -0.2) * 1e2;
            let inertial_s = Direction::from_dir(UnitVector3::new_normalize(Vector3::y()));
            let mag_dir_variance = Matrix2::identity() * 1.0;
            let mag_norm_variance = 1.0;
            let sun_direction_variance = Matrix2::identity();
            let mut estimator = FullDirectionEstimation::new_for_test(
                &initial_time,
                Direction::from_dir(UnitVector3::new_normalize(inertial_b)),
                mag_dir_variance,
                inertial_b.norm(),
                mag_norm_variance,
                inertial_s.clone(),
                sun_direction_variance,
            );

            // True magnetic field (body frame)
            let true_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
            let mut true_b = true_q * inertial_b;
            let mut true_s = true_q * inertial_s.dir();
            let omega = DirectionInputGaussian {
                angular_velocity: nalgebra::SVector::<f64, 3>::new(0.1, 0.0, 0.0),
                angular_velocity_covariance: Matrix3::identity() * 0.1,
            };
            let process_noise_covariance =
                Some(full_direction_estimation_process_noise_covariance());
            let mut time = initial_time;
            let dt = 0.02;
            // Run many propagation and update cycles
            for i in 0..100 {
                time += dt;
                let delta_theta = omega.angular_velocity * dt;
                let delta_q = nalgebra::UnitQuaternion::from_scaled_axis(delta_theta);
                true_b = delta_q * true_b;
                true_s = delta_q * true_s;
                estimator
                    .ukf
                    .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                    .unwrap();
                // Occasional measurements
                if i % 5 == 0 {
                    let b_measured = MagFieldObservationForFull { mag_field: true_b };
                    let observation_model = MagFieldObservationModelForFull;
                    estimator
                        .ukf
                        .update(
                            &observation_model,
                            &b_measured,
                            &EmptyInput,
                            &EmptyInput,
                            &time,
                            Matrix3::identity() * 0.001,
                        )
                        .unwrap();
                    let s_measured = SunDirectionObservation {
                        sun_direction: Direction::from_dir(true_s),
                    };
                    let sun_observation_model = SunDirectionObservationModel;
                    estimator
                        .ukf
                        .update(
                            &sun_observation_model,
                            &s_measured,
                            &EmptyInput,
                            &EmptyInput,
                            &time,
                            Matrix2::identity() * 0.001,
                        )
                        .unwrap();
                };
                let cov = estimator.ukf.covariance();
                // Check symmetry
                let symmetry_error = (cov - cov.transpose()).norm();
                assert!(
                    symmetry_error < 1e-10,
                    "Covariance should be symmetric at iteration {}",
                    i
                );
                // Check positive definiteness
                let eigenvalues = cov.symmetric_eigenvalues();
                for (j, eig) in eigenvalues.iter().enumerate() {
                    assert!(
                        eig > &0.0,
                        "Covariance eigenvalue {} should be positive at iteration {}: {}",
                        j,
                        i,
                        eig
                    );
                }
            }
        }
    }

    mod quaternion {
        use nalgebra::UnitQuaternion;

        use crate::estimation::triad_method;

        use super::*;

        #[derive(Debug, Clone, EstimationGaussianInput)]
        struct InertialVectorInput {
            inertial_sun_direction: Direction,
            inertial_mag_field: SVector<f64, 3>,
        }

        #[derive(Debug, Clone, EstimationOutputStruct)]
        struct AttitudeQuaternionObservation {
            attitude: UnitQuaternion<f64>,
        }

        struct AttitudeQuaternionObservationModel;

        impl ObservationModel for AttitudeQuaternionObservationModel {
            type State = FullDirectionState;
            type DeterministicInput = EmptyInput;
            type GaussianInput = InertialVectorInput;
            type Time = crate::Time;
            type Observation = AttitudeQuaternionObservation;
            fn predict(
                &self,
                state: &Self::State,
                _deterministic_input: &Self::DeterministicInput,
                gaussian_input: &Self::GaussianInput,
                _time: &Self::Time,
            ) -> Self::Observation {
                let b_body = state.mag_dir.dir().into_inner();
                let s_body = state.sun_direction.dir().into_inner();
                let b_inertial = gaussian_input.inertial_mag_field;
                let s_inertial = gaussian_input.inertial_sun_direction.dir().into_inner();
                let attitude = triad_method(&b_body, &s_body, &b_inertial, &s_inertial);
                AttitudeQuaternionObservation { attitude }
            }
        }

        #[test]
        fn test_ukf_attitude_quaternion_update_reduces_error_and_covariance() {
            let initial_time = crate::Time::from_seconds(0.0);
            let initial_q = UnitQuaternion::identity();
            let inertial_b = nalgebra::SVector::<f64, 3>::new(100.0, 0.0, 0.0);
            let initial_b = initial_q * inertial_b;
            let inertial_s = Vector3::x_axis();
            let initial_s = initial_q * inertial_s;
            let mut estimator = FullDirectionEstimation::new_for_direct(
                &initial_time,
                UnitVector3::new_normalize(initial_b),
                Matrix2::identity() * 1.0,
                initial_b.norm(),
                1.0,
                initial_s,
                Matrix2::identity() * 1.0,
            );

            let cov_before = *estimator.ukf.covariance();

            // Apply attitude quaternion update
            let time = initial_time;
            let q_true = UnitQuaternion::new(Vector3::new(0.1, -0.2, 0.3));
            let s_true = q_true * inertial_s;
            let b_true = q_true * inertial_b;
            let q_measured = AttitudeQuaternionObservation { attitude: q_true };
            let observation_model = AttitudeQuaternionObservationModel;
            let observation_input = InertialVectorInputGaussian {
                inertial_mag_field: inertial_b,
                inertial_mag_field_covariance: Matrix3::identity() * 0.001,
                inertial_sun_direction: Direction::from_dir(inertial_s),
                inertial_sun_direction_covariance: Matrix2::identity() * 0.001,
            };

            estimator
                .ukf
                .update(
                    &observation_model,
                    &q_measured,
                    &EmptyInput,
                    &observation_input,
                    &time,
                    Matrix3::identity() * 0.0001,
                )
                .unwrap();

            let cov_after = estimator.ukf.covariance();

            // Quaternion観測が方向ベクトルに影響を与えないことを確認
            assert!(
                (estimator
                    .ukf
                    .state()
                    .mag_dir
                    .dir()
                    .angle(&UnitVector3::new_normalize(b_true)))
                .abs()
                    == initial_b.angle(&UnitVector3::new_normalize(b_true)).abs(),
                "Magnetic field direction estimate should not update: {:?} vs {:?}, error: {}, initial error: {}",
                estimator.ukf.state().mag_dir.dir(),
                UnitVector3::new_normalize(b_true),
                estimator
                    .ukf
                    .state()
                    .mag_dir
                    .dir()
                    .angle(&UnitVector3::new_normalize(b_true)),
                initial_b.angle(&UnitVector3::new_normalize(b_true))
            );

            assert!(
                estimator
                    .ukf
                    .state()
                    .sun_direction
                    .dir()
                    .angle(&s_true)
                    .abs()
                    == initial_s.angle(&s_true).abs(),
                "Sun direction estimate should not update: {:?} vs {:?}, error: {}, initial error: {}",
                estimator.ukf.state().sun_direction.dir(),
                s_true,
                estimator.ukf.state().sun_direction.dir().angle(&s_true),
                initial_s.angle(&s_true)
            );
            // Quaternion観測は不確かさを減らさないはず
            assert!(
                cov_after.fixed_view::<2, 2>(3, 3).trace()
                    <= cov_before.fixed_view::<2, 2>(3, 3).trace(),
                "Sun direction covariance should decrease after update: {} -> {}",
                cov_before,
                cov_after
            );
        }
    }
}
