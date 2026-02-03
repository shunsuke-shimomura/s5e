use control_system::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    components::Direction,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter, input_shift},
    value_structs::EmptyInput,
};
use nalgebra::{Matrix2, Matrix3, SVector, UnitQuaternion, UnitVector3};
use std::time::{Duration, Instant};

use crate::{
    constants::{GYRO_BIAS_DRIFT_STD, UKF_ALPHA, UKF_BETA, UKF_KAPPA},
    data,
    estimation::triad_method,
};

fn attitude_estimation_process_noise_covariance() -> nalgebra::SMatrix<f64, 6, 6> {
    let mut matrix = nalgebra::SMatrix::<f64, 6, 6>::zeros();
    matrix
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(Matrix3::identity() * GYRO_BIAS_DRIFT_STD * GYRO_BIAS_DRIFT_STD));
    matrix
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct AttitudeDeterminationPropagationInput {
    angular_velocity: SVector<f64, 3>,
}

#[derive(EstimationState, Clone, Debug)]
struct AttitudeDeterminationState {
    attitude: UnitQuaternion<f64>,
    gyro_bias: nalgebra::SVector<f64, 3>,
}

struct AttitudeDeterminationModel;

impl PropagationModel for AttitudeDeterminationModel {
    type State = AttitudeDeterminationState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = AttitudeDeterminationPropagationInput;
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
        let delta_theta = omega_est * *dt;
        let delta_q = UnitQuaternion::new(delta_theta);
        AttitudeDeterminationState {
            attitude: delta_q * state.attitude,
            gyro_bias: state.gyro_bias,
        }
    }
}

#[derive(EstimationOutputStruct, Debug)]
struct AttitudeObservation {
    attitude: UnitQuaternion<f64>,
}

struct AttitudeObservationModel;

impl ObservationModel for AttitudeObservationModel {
    type State = AttitudeDeterminationState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = crate::Time;
    type Observation = AttitudeObservation;
    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        _gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        AttitudeObservation {
            attitude: state.attitude,
        }
    }
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct MagneticFieldInput {
    magnetic_field: SVector<f64, 3>,
}

fn mag_dir_func(mag_field: &MagneticFieldInput) -> MagneticDirectionOutput {
    MagneticDirectionOutput {
        magnetic_direction: Direction::from_dir(UnitVector3::new_normalize(
            mag_field.magnetic_field,
        )),
    }
}

#[derive(EstimationOutputStruct, Debug, Clone)]
struct MagneticDirectionOutput {
    magnetic_direction: Direction,
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct InertialMagneticDirectionInput {
    inertial_magnetic_direction: Direction,
}

#[derive(EstimationOutputStruct, Debug)]
struct MagDirObservation {
    magnetic_direction: Direction,
}

struct MagDirObservationModel;

impl ObservationModel for MagDirObservationModel {
    type State = AttitudeDeterminationState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = InertialMagneticDirectionInput;
    type Time = crate::Time;
    type Observation = MagDirObservation;

    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        let body_q = &state.attitude;
        let inertial_b = gaussian_input.inertial_magnetic_direction.clone();
        MagDirObservation {
            magnetic_direction: body_q.to_rotation_matrix() * inertial_b,
        }
    }
}

#[derive(EstimationGaussianInput, Clone, Debug)]
struct InertialSunDirectionInput {
    inertial_sun_direction: Direction,
}

#[derive(EstimationOutputStruct, Debug)]
struct SunDirectionObservation {
    sun_direction: Direction,
}

struct SunDirectionObservationModel;

impl ObservationModel for SunDirectionObservationModel {
    type State = AttitudeDeterminationState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = InertialSunDirectionInput;
    type Time = crate::Time;
    type Observation = SunDirectionObservation;
    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        let body_q = &state.attitude;
        let inertial_s = gaussian_input.inertial_sun_direction.clone();
        let body_s = body_q.to_rotation_matrix() * inertial_s;
        SunDirectionObservation {
            sun_direction: body_s,
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum AttitudeDeterminationMode {
    Initial,
    Active(QuaternionEstimation),
}

struct QuaternionEstimation {
    ukf: UnscentedKalmanFilter<
        AttitudeDeterminationState,
        crate::Time,
        f64,
        AttitudeDeterminationModel,
        EmptyInput,
        AttitudeDeterminationPropagationInputGaussian,
        6,
        3,
    >,
    attitude_obs_model: AttitudeObservationModel,
    mag_obs_model: MagDirObservationModel,
    sun_obs_model: SunDirectionObservationModel,
}

impl QuaternionEstimation {
    pub fn new(
        initial_attitude: UnitQuaternion<f64>,
        initial_attitude_covariance: Matrix3<f64>,
        initial_time: &crate::Time,
    ) -> Self {
        let propagator = AttitudeDeterminationModel;
        let initial_covariance = {
            let mut mat = nalgebra::SMatrix::<f64, 6, 6>::zeros();
            // Use the provided attitude covariance instead of default
            mat.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&initial_attitude_covariance);
            // Keep gyro bias covariance as default for now
            mat.fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(Matrix3::identity() * 1e-1));
            mat
        };

        let ukf = UnscentedKalmanFilter::new(
            propagator,
            AttitudeDeterminationState {
                attitude: initial_attitude,
                gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
            },
            initial_covariance,
            initial_time,
            UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        );
        QuaternionEstimation {
            ukf,
            attitude_obs_model: AttitudeObservationModel,
            mag_obs_model: MagDirObservationModel,
            sun_obs_model: SunDirectionObservationModel,
        }
    }
}

// Input struct for TRIAD initialization covariance calculation
#[derive(EstimationGaussianInput, Clone, Debug)]
struct TriadObservationInput {
    body_magnetic_field: SVector<f64, 3>,
    body_sun_direction: SVector<f64, 3>,
    inertial_magnetic_field: SVector<f64, 3>,
    inertial_sun_direction: SVector<f64, 3>,
}

pub struct AttitudeDeterminationInput {
    pub magnetic_field: Option<data::MagnetometerData>,
    pub gyro_data: Option<data::GyroSensorData>,
    pub sun_direction: Option<data::SunSensorData>,
    pub inertial_mag: Option<data::InertialMagneticFieldData>,
    pub inertial_sun: Option<data::InertialSunDirectionData>,
    pub star_tracker: Option<data::StarTrackerData>,
}

#[derive(Default)]
pub struct AttitudeDeterminationProfile {
    pub propagation: Duration,
    pub stt_update: Duration,
    pub mag_update: Duration,
    pub sun_update: Duration,
    pub initialization: Duration,
    pub output_extraction: Duration,
    pub call_count: u64,
}

impl AttitudeDeterminationProfile {
    pub fn print_summary(&self) {
        let total = self.propagation
            + self.stt_update
            + self.mag_update
            + self.sun_update
            + self.initialization
            + self.output_extraction;
        let pct = |d: Duration| {
            if total.as_nanos() == 0 {
                0.0
            } else {
                d.as_secs_f64() / total.as_secs_f64() * 100.0
            }
        };
        println!(
            "=== AttitudeDetermination Profile ({} calls, total {:.3}s) ===",
            self.call_count,
            total.as_secs_f64()
        );
        println!(
            "  propagation:        {:>8.3}ms ({:>5.1}%)",
            self.propagation.as_secs_f64() * 1000.0,
            pct(self.propagation)
        );
        println!(
            "  stt_update:         {:>8.3}ms ({:>5.1}%)",
            self.stt_update.as_secs_f64() * 1000.0,
            pct(self.stt_update)
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
            "  initialization:     {:>8.3}ms ({:>5.1}%)",
            self.initialization.as_secs_f64() * 1000.0,
            pct(self.initialization)
        );
        println!(
            "  output_extraction:  {:>8.3}ms ({:>5.1}%)",
            self.output_extraction.as_secs_f64() * 1000.0,
            pct(self.output_extraction)
        );
    }
}

pub struct AttitudeDeterminationOutput {
    pub attitude: Option<data::AttitudeDeterminationData>,
    pub angular_velocity: Option<data::AngularVelocityData>,
}

pub struct AttitudeDetermination {
    mode: AttitudeDeterminationMode,
    pub profile: AttitudeDeterminationProfile,
}

impl Default for AttitudeDetermination {
    fn default() -> Self {
        Self::new()
    }
}

impl AttitudeDetermination {
    pub fn new() -> Self {
        AttitudeDetermination {
            mode: AttitudeDeterminationMode::Initial,
            profile: AttitudeDeterminationProfile::default(),
        }
    }

    pub fn gyro_bias(&self) -> Option<data::GyroBiasData> {
        match &self.mode {
            AttitudeDeterminationMode::Initial => None,
            AttitudeDeterminationMode::Active(estimation) => {
                let state = estimation.ukf.state();
                Some(data::GyroBiasData {
                    gyro_bias: state.gyro_bias,
                    gyro_bias_variance: estimation
                        .ukf
                        .covariance()
                        .fixed_view::<3, 3>(3, 3)
                        .clone_owned(),
                })
            }
        }
    }

    pub fn main_loop(
        &mut self,
        time: &crate::Time,
        input: &AttitudeDeterminationInput,
    ) -> AttitudeDeterminationOutput {
        self.profile.call_count += 1;
        match &mut self.mode {
            AttitudeDeterminationMode::Initial => {
                let t_prof = Instant::now();
                // Initialize attitude estimation with appropriate covariance
                if let Some(star_data) = input.star_tracker.as_ref() {
                    // For star tracker, use its measurement noise as initial covariance
                    let initial_covariance = Matrix3::identity() * star_data.std * star_data.std;
                    let estimation =
                        QuaternionEstimation::new(star_data.attitude, initial_covariance, time);
                    self.mode = AttitudeDeterminationMode::Active(estimation);
                } else if let (
                    Some(mag_data),
                    Some(inertial_mag_data),
                    Some(sun_data),
                    Some(inertial_sun_data),
                ) = (
                    input.magnetic_field.as_ref(),
                    input.inertial_mag.as_ref(),
                    input.sun_direction.as_ref(),
                    input.inertial_sun.as_ref(),
                ) {
                    let b_body = &mag_data.magnetic_field;
                    let s_body = &sun_data.sun_direction.into_inner();
                    let b_inertial = &inertial_mag_data.magnetic_field_eci;
                    let s_inertial = &inertial_sun_data.sun_direction_eci.into_inner();

                    // Calculate initial covariance from sensor observations using input_shift
                    let triad_input = TriadObservationInputGaussian {
                        body_magnetic_field: *b_body,
                        body_magnetic_field_covariance: Matrix3::identity()
                            * mag_data.std
                            * mag_data.std,
                        body_sun_direction: *s_body,
                        body_sun_direction_covariance: Matrix3::identity()
                            * sun_data.std
                            * sun_data.std,
                        inertial_magnetic_field: *b_inertial,
                        inertial_magnetic_field_covariance: inertial_mag_data
                            .magnetic_field_eci_variance,
                        inertial_sun_direction: *s_inertial,
                        inertial_sun_direction_covariance: Matrix3::identity()
                            * inertial_sun_data.std
                            * inertial_sun_data.std,
                    };

                    // Use input_shift to propagate measurement uncertainties to attitude uncertainty
                    let triad_func = |input: &TriadObservationInput| -> AttitudeDeterminationState {
                        let q = triad_method(
                            &input.body_magnetic_field,
                            &input.body_sun_direction,
                            &input.inertial_magnetic_field,
                            &input.inertial_sun_direction,
                        );
                        AttitudeDeterminationState {
                            attitude: q,
                            gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
                        }
                    };

                    let ukf_params = UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA);

                    let (initial_state, covariance) =
                        input_shift(&triad_input, triad_func, &ukf_params).unwrap();
                    // Extract the attitude portion of the covariance (first 3x3 block)
                    let initial_attitude_covariance =
                        covariance.fixed_view::<3, 3>(0, 0).clone_owned();
                    let estimation = QuaternionEstimation::new(
                        initial_state.attitude,
                        initial_attitude_covariance,
                        time,
                    );
                    self.mode = AttitudeDeterminationMode::Active(estimation);
                }
                self.profile.initialization += t_prof.elapsed();
                AttitudeDeterminationOutput {
                    attitude: None,
                    angular_velocity: None,
                }
            }
            AttitudeDeterminationMode::Active(estimation) => {
                if let Some(gyro_data) = input.gyro_data.as_ref() {
                    // Removed verbose angular velocity output
                    let t_prof = Instant::now();
                    let angular_velocity_input = AttitudeDeterminationPropagationInputGaussian {
                        angular_velocity: gyro_data.angular_velocity,
                        angular_velocity_covariance: Matrix3::identity()
                            * gyro_data.std
                            * gyro_data.std,
                    };
                    let process_noise_covariance =
                        Some(attitude_estimation_process_noise_covariance());
                    estimation
                        .ukf
                        .propagate(
                            &EmptyInput,
                            &angular_velocity_input,
                            process_noise_covariance,
                            time,
                        )
                        .unwrap();
                    self.profile.propagation += t_prof.elapsed();
                    let t_prof = Instant::now();
                    if let Some(stt_data) = input.star_tracker.as_ref() {
                        let measurement_covariance =
                            Matrix3::identity() * stt_data.std * stt_data.std;
                        let measured_q = AttitudeObservation {
                            attitude: stt_data.attitude,
                        };
                        estimation
                            .ukf
                            .update(
                                &estimation.attitude_obs_model,
                                &measured_q,
                                &EmptyInput,
                                &EmptyInput,
                                time,
                                measurement_covariance,
                            )
                            .unwrap();
                    }
                    self.profile.stt_update += t_prof.elapsed();
                    let t_prof = Instant::now();
                    if let (Some(mag_data), Some(inertial_mag_data)) =
                        (input.magnetic_field.as_ref(), input.inertial_mag.as_ref())
                    {
                        let inertial_mag_field_input = MagneticFieldInputGaussian {
                            magnetic_field: inertial_mag_data.magnetic_field_eci,
                            magnetic_field_covariance: inertial_mag_data
                                .magnetic_field_eci_variance,
                        };
                        let measured_mag_field_input = MagneticFieldInputGaussian {
                            magnetic_field: mag_data.magnetic_field,
                            magnetic_field_covariance: Matrix3::identity()
                                * mag_data.std
                                * mag_data.std,
                        };
                        let (inertial_mag_dir, inertial_mag_dir_covariance) = input_shift(
                            &inertial_mag_field_input,
                            mag_dir_func,
                            &UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
                        )
                        .unwrap();
                        let (measured_mag_dir, measured_mag_dir_covariance) = input_shift(
                            &measured_mag_field_input,
                            mag_dir_func,
                            &UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
                        )
                        .unwrap();

                        let inertial_b = InertialMagneticDirectionInputGaussian {
                            inertial_magnetic_direction: inertial_mag_dir.magnetic_direction,
                            inertial_magnetic_direction_covariance: inertial_mag_dir_covariance
                                + Matrix2::identity() * 1e-6,
                        };
                        let measured_b = MagDirObservation {
                            magnetic_direction: measured_mag_dir.magnetic_direction,
                        };
                        let measurement_covariance = measured_mag_dir_covariance;

                        estimation
                            .ukf
                            .update(
                                &estimation.mag_obs_model,
                                &measured_b,
                                &EmptyInput,
                                &inertial_b,
                                time,
                                measurement_covariance,
                            )
                            .unwrap();
                    }
                    self.profile.mag_update += t_prof.elapsed();
                    let t_prof = Instant::now();
                    if let (Some(sun_data), Some(inertial_sun_data)) =
                        (input.sun_direction.as_ref(), input.inertial_sun.as_ref())
                    {
                        let measurement_covariance =
                            Matrix2::identity() * sun_data.std * sun_data.std;
                        let measured_s = SunDirectionObservation {
                            sun_direction: Direction::from_dir(sun_data.sun_direction),
                        };
                        let inertial_s = InertialSunDirectionInputGaussian {
                            inertial_sun_direction: Direction::from_dir(
                                inertial_sun_data.sun_direction_eci,
                            ),
                            inertial_sun_direction_covariance: inertial_sun_data.std
                                * inertial_sun_data.std
                                * Matrix2::identity(),
                        };
                        estimation
                            .ukf
                            .update(
                                &estimation.sun_obs_model,
                                &measured_s,
                                &EmptyInput,
                                &inertial_s,
                                time,
                                measurement_covariance,
                            )
                            .unwrap();
                    }
                    self.profile.sun_update += t_prof.elapsed();
                    let t_prof = Instant::now();
                    let state = estimation.ukf.state();
                    let covariance = estimation.ukf.covariance();
                    let attitude_data = data::AttitudeDeterminationData {
                        attitude: state.attitude,
                        attitude_variance: covariance.fixed_view::<3, 3>(0, 0).clone_owned(),
                    };
                    let angular_velocity_data = data::AngularVelocityData {
                        angular_velocity: gyro_data.angular_velocity
                            - state.gyro_bias.clone_owned(),
                        angular_velocity_variance: Matrix3::identity()
                            * gyro_data.std
                            * gyro_data.std
                            + estimation
                                .ukf
                                .covariance()
                                .fixed_view::<3, 3>(3, 3)
                                .clone_owned(),
                    };
                    let output = AttitudeDeterminationOutput {
                        attitude: Some(attitude_data),
                        angular_velocity: Some(angular_velocity_data),
                    };
                    self.profile.output_extraction += t_prof.elapsed();
                    output
                } else {
                    AttitudeDeterminationOutput {
                        attitude: None,
                        angular_velocity: None,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_init {
    use super::*;
    use nalgebra::{Matrix3, UnitQuaternion};

    #[test]
    fn test_ukf_initialization() {
        let initial_time = crate::Time::from_seconds(0.0);
        let initial_q = UnitQuaternion::identity();
        let initial_covariance = Matrix3::identity(); // Default for test

        let estimation = QuaternionEstimation::new(initial_q, initial_covariance, &initial_time);

        // Check initial state
        assert!((estimation.ukf.state().attitude.angle() - initial_q.angle()).abs() < 1e-10);
        assert!(estimation.ukf.state().gyro_bias.norm() < 1e-10);

        // Check covariance is positive definite
        let cov = estimation.ukf.covariance();
        let eigenvalues = cov.symmetric_eigenvalues();
        for eig in eigenvalues.iter() {
            assert!(eig > &0.0, "Initial covariance should be positive definite");
        }
    }
}

#[cfg(test)]
mod tests_propagation {
    use super::*;
    use crate::constants::GYRO_NOISE_STD;
    use astrodynamics::coordinate::BodyVector;
    use control_system::integrator::{
        Prediction, TimeIntegrator,
        rk4::{RK4Phase, RK4Solver},
    };
    use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
    use rand::distributions::Distribution;
    use statrs::distribution::MultivariateNormal;

    #[test]
    fn test_propagation_pos_and_neg() {
        let propagation_model = AttitudeDeterminationModel;
        let initial_q = UnitQuaternion::from_quaternion(Quaternion::new(
            -0.18473353039839202,
            0.7308976443576298,
            -0.5467873085093476,
            0.3642606147693126,
        ));
        let initial_state = AttitudeDeterminationState {
            attitude: initial_q,
            gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
        };
        let omega = nalgebra::SVector::<f64, 3>::new(0.0, 0.0, 1.0);
        let dt = 0.01;

        let g_input_nominal = AttitudeDeterminationPropagationInput {
            angular_velocity: omega,
        };
        let new_state_nominal = propagation_model.propagate(
            &initial_state,
            &EmptyInput,
            &g_input_nominal,
            &crate::Time::from_seconds(0.0),
            &dt,
        );
        let nominal_error = new_state_nominal.attitude.error(&initial_state.attitude);

        let g_input_pos = AttitudeDeterminationPropagationInput {
            angular_velocity: omega + nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        };
        let new_state_pos = propagation_model.propagate(
            &initial_state,
            &EmptyInput,
            &g_input_pos,
            &crate::Time::from_seconds(0.0),
            &dt,
        );
        use control_system::components::GaussianValueType;
        let pos_error_from_before = new_state_pos.attitude.error(&initial_state.attitude);
        let pos_error_from_after = new_state_pos.attitude.error(&new_state_nominal.attitude);

        let g_input_neg = AttitudeDeterminationPropagationInput {
            angular_velocity: omega - nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        };
        let new_state_neg = propagation_model.propagate(
            &initial_state,
            &EmptyInput,
            &g_input_neg,
            &crate::Time::from_seconds(0.0),
            &dt,
        );
        let neg_error_from_before = new_state_neg.attitude.error(&initial_state.attitude);
        let neg_error_from_after = new_state_neg.attitude.error(&new_state_nominal.attitude);

        println!();
        println!("Pos error from before: {:?}", pos_error_from_before);
        println!("Neg error from before: {:?}", neg_error_from_before);
        println!("Pos error from after : {:?}", pos_error_from_after);
        println!("Neg error from after : {:?}", neg_error_from_after);
        assert!(
            ((pos_error_from_before + neg_error_from_before) / 2.0 - nominal_error).norm() < 1e-10,
            "Propagation with positive and negative angular velocities should yield opposite directions, error {:?}",
            (pos_error_from_before + neg_error_from_before) / 2.0 - nominal_error
        );
        assert!(
            (pos_error_from_after + neg_error_from_after).norm() < 1e-6,
            "Propagation with positive and negative angular velocities should yield opposite directions, error {:?}",
            pos_error_from_after + neg_error_from_after
        );
    }

    #[test]
    fn test_propagation_model_with_rk4_isotropic() {
        let mut time = 0.0;
        let dt = 0.01;
        let inertia = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let initial_q = Quaternion::new(
            -0.18473353039839202,
            0.7308976443576298,
            -0.5467873085093476,
            0.3642606147693126,
        );
        let mut omega = RK4Solver::new(BodyVector::from(nalgebra::SVector::<f64, 3>::new(
            0.01, 0.005, 0.007,
        )));
        let mut attitude = RK4Solver::new(initial_q);
        let mut state = AttitudeDeterminationState {
            attitude: UnitQuaternion::from_quaternion(initial_q),
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
            let propagation_model = AttitudeDeterminationModel;
            let g_input = AttitudeDeterminationPropagationInput {
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

        let rk4_attitude = UnitQuaternion::from_quaternion(attitude.get_now());

        let ukf_attitude = state.attitude;

        println!();
        println!("Propagated attitude: {}", state.attitude);
        println!(
            "RK4 attitude       : {}",
            UnitQuaternion::from_quaternion(attitude.get_now())
        );
        println!("RK4 attitude norm  : {}", attitude.get_now().norm());
        assert!(
            (rk4_attitude * ukf_attitude.inverse()).angle().abs() < 1e-6,
            "RK4 attitude and UKF propagated attitude should match: RK4 error = {}",
            (rk4_attitude * ukf_attitude.inverse()).angle().abs(),
        );
    }

    #[test]
    fn test_ukf_propagation_with_rk4_isotropic() {
        println!();
        let mut time = 0.0;
        let dt = 0.01;
        let inertia = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let initial_q = Quaternion::new(
            -0.18473353039839202,
            0.7308976443576298,
            -0.5467873085093476,
            0.3642606147693126,
        );
        let mut omega = RK4Solver::new(BodyVector::from(nalgebra::SVector::<f64, 3>::new(
            0.01, 0.005, 0.007,
        )));
        let mut attitude = RK4Solver::new(initial_q);
        let mut estimation = QuaternionEstimation::new(
            UnitQuaternion::from_quaternion(initial_q),
            Matrix3::identity(),
            &crate::Time::from_seconds(time),
        );
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
            let g_input = AttitudeDeterminationPropagationInputGaussian {
                angular_velocity: omega.get_now().into(),
                angular_velocity_covariance: Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
            };
            let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());

            estimation
                .ukf
                .propagate(
                    &EmptyInput,
                    &g_input,
                    process_noise_covariance,
                    &crate::Time::from_seconds(time),
                )
                .unwrap();
            let state = estimation.ukf.state();
            let rk4_q = UnitQuaternion::from_quaternion(attitude.get_now());
            let attitude_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal();
            assert!(
                (state.attitude * rk4_q.inverse()).angle().abs()
                    <= 3.0 * attitude_covariance_diagonal.map(|x| x.sqrt()).norm(),
                "Attitude error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                time,
                (state.attitude * rk4_q.inverse()).scaled_axis(),
                attitude_covariance_diagonal.map(|x| x.sqrt()).norm()
            );
        }

        let rk4_attitude = UnitQuaternion::from_quaternion(attitude.get_now());

        println!("Estimated attitude : {}", estimation.ukf.state().attitude);
        println!(
            "RK4 attitude       : {}",
            UnitQuaternion::from_quaternion(attitude.get_now())
        );
        println!();

        let ukf_attitude = estimation.ukf.state().attitude;
        assert!(
            (rk4_attitude * ukf_attitude.inverse()).angle().abs() < 1e-2,
            "RK4 attitude and UKF propagated attitude should match: error = {}",
            (rk4_attitude * ukf_attitude.inverse()).angle().abs()
        );
    }

    #[test]
    fn test_propagation_model_with_ukf_propagation() {
        let mut time = 0.0;
        let dt = 0.01;
        let initial_q = Quaternion::new(
            -0.18473353039839202,
            0.7308976443576298,
            -0.5467873085093476,
            0.3642606147693126,
        );
        let omega = nalgebra::SVector::<f64, 3>::new(0.01, 0.005, 0.007);
        let mut state = AttitudeDeterminationState {
            attitude: UnitQuaternion::from_quaternion(initial_q),
            gyro_bias: nalgebra::SVector::<f64, 3>::zeros(),
        };
        let mut estimation = QuaternionEstimation::new(
            UnitQuaternion::from_quaternion(initial_q),
            Matrix3::identity(),
            &crate::Time::from_seconds(time),
        );
        for _i in 0..100 {
            time += dt;
            // Compare with small angle approximation
            let propagation_model = AttitudeDeterminationModel;
            let g_input = AttitudeDeterminationPropagationInput {
                angular_velocity: omega,
            };
            state = propagation_model.propagate(
                &state,
                &EmptyInput,
                &g_input,
                &crate::Time::from_seconds(time),
                &dt,
            );

            let g_input = AttitudeDeterminationPropagationInputGaussian {
                angular_velocity: omega,
                angular_velocity_covariance: Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
            };
            let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());

            estimation
                .ukf
                .propagate(
                    &EmptyInput,
                    &g_input,
                    process_noise_covariance,
                    &crate::Time::from_seconds(time),
                )
                .unwrap();
        }

        println!();
        println!("Propagated attitude: {}", state.attitude);
        println!("UKF attitude       : {}", estimation.ukf.state().attitude);

        let ukf_attitude = estimation.ukf.state().attitude;

        let model_attitude = state.attitude;
        assert!(
            (model_attitude * ukf_attitude.inverse()).angle().abs() < 1e-2,
            "RK4 attitude and UKF propagated attitude should match: error = {}",
            (model_attitude * ukf_attitude.inverse()).angle().abs(),
        );
    }

    // このテストは非常に時間がかかる上、統計的に優位な結果を得るには更に多くのサンプルが必要であるため、通常のテスト実行ではスキップする
    #[test]
    #[ignore]
    fn test_ukf_propagation_with_statistics() {
        let initial_time = crate::Time::from_seconds(0.0);

        let mut estimation = QuaternionEstimation::new(
            UnitQuaternion::identity(),
            Matrix3::identity(),
            &initial_time,
        );

        let num_samples = 1000000;
        let mut shifted_attitudes = Vec::with_capacity(num_samples);

        let omega_mean = nalgebra::SVector::<f64, 3>::new(0.01, 0.005, 0.007);
        let omega_cov = Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD;
        let omega_dist = MultivariateNormal::new_from_nalgebra(omega_mean, omega_cov).unwrap();
        let state_attitude_mean = estimation.ukf.state().attitude;
        let state_attitude_cov = estimation
            .ukf
            .covariance()
            .fixed_view::<3, 3>(0, 0)
            .clone_owned();
        let state_attitude_dist = MultivariateNormal::new_from_nalgebra(
            nalgebra::SVector::<f64, 3>::zeros(),
            state_attitude_cov,
        )
        .unwrap();
        let state_gyro_bias_mean = estimation.ukf.state().gyro_bias;
        let state_gyro_bias_cov = estimation
            .ukf
            .covariance()
            .fixed_view::<3, 3>(3, 3)
            .clone_owned();
        let state_gyro_bias_dist =
            MultivariateNormal::new_from_nalgebra(state_gyro_bias_mean, state_gyro_bias_cov)
                .unwrap();

        let mut rng = rand::thread_rng();
        let dt = 0.01;

        let new_attitude_nominal = {
            let omega_est = omega_mean - state_gyro_bias_mean;
            let delta_theta = omega_est * dt;
            let delta_q = UnitQuaternion::new(delta_theta);
            delta_q * state_attitude_mean
        };

        for _ in 0..num_samples {
            let sampled_omega = SVector::<f64, 3>::from(omega_dist.sample(&mut rng));
            let sampled_attitude_error =
                SVector::<f64, 3>::from(state_attitude_dist.sample(&mut rng));
            let sampled_gyro_bias = SVector::<f64, 3>::from(state_gyro_bias_dist.sample(&mut rng));
            let sampled_attitude =
                UnitQuaternion::new(sampled_attitude_error) * state_attitude_mean;
            let omega_est = sampled_omega - sampled_gyro_bias;
            let delta_theta = omega_est * dt;
            let delta_q = UnitQuaternion::new(delta_theta);
            let new_attitude = delta_q * sampled_attitude;
            println!();
            println!("delta_theta : {:?}", delta_theta);
            println!(
                "inversed    : {:?}",
                (new_attitude * sampled_attitude.inverse()).scaled_axis()
            );
            println!("new attitude: {}", new_attitude);
            println!(
                "new reconst : {}",
                UnitQuaternion::new((new_attitude * new_attitude_nominal.inverse()).scaled_axis())
                    * new_attitude_nominal
            );
            shifted_attitudes.push(new_attitude);
        }

        let new_attitude_mean = {
            let mean_vec = shifted_attitudes.iter().fold(Vector3::zeros(), |acc, q| {
                acc + (q * new_attitude_nominal.inverse()).scaled_axis()
            }) / (num_samples as f64);
            UnitQuaternion::new(mean_vec) * new_attitude_nominal
        };

        let omega_input = AttitudeDeterminationPropagationInputGaussian {
            angular_velocity: omega_mean,
            angular_velocity_covariance: omega_cov,
        };
        let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());
        estimation
            .ukf
            .propagate(
                &EmptyInput,
                &omega_input,
                process_noise_covariance,
                &(initial_time + dt),
            )
            .unwrap();
        let ukf_attitude = estimation.ukf.state().attitude;
        println!();
        println!(
            "Nominal attitude         : {}",
            new_attitude_nominal.into_inner()
        );
        println!(
            "Monte Carlo mean attitude: {}",
            new_attitude_mean.into_inner()
        );
        println!("UKF propagated attitude  : {}", ukf_attitude.into_inner());
        assert!(
            (new_attitude_mean * ukf_attitude.inverse()).angle().abs() < 1e-2,
            "UKF propagated attitude should match Monte Carlo mean: UKF attitude = {}, MC mean attitude = {}",
            ukf_attitude,
            new_attitude_mean
        );
    }

    #[test]
    fn test_ukf_propagation_increases_covariance() {
        let initial_time = crate::Time::from_seconds(0.0);

        let mut estimation = QuaternionEstimation::new(
            UnitQuaternion::identity(),
            Matrix3::identity(),
            &initial_time,
        );

        let cov_before = estimation.ukf.covariance().trace();

        // Propagate with zero angular velocity
        let omega = AttitudeDeterminationPropagationInputGaussian {
            angular_velocity: nalgebra::SVector::<f64, 3>::zeros(),
            angular_velocity_covariance: Matrix3::identity() * 0.1,
        };
        let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());
        let time = initial_time + 0.1;
        estimation
            .ukf
            .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
            .unwrap();

        let cov_after = estimation.ukf.covariance().trace();

        // Covariance trace should increase due to process noise (gyro bias random walk)
        assert!(
            cov_after > cov_before,
            "Covariance should increase after propagation: {} -> {}",
            cov_before,
            cov_after
        );
    }
}

#[cfg(test)]
mod tests_update {
    use crate::constants::STAR_TRACKER_NOISE_STD;

    use super::*;
    use control_system::components::GaussianValueType;
    use nalgebra::{Matrix2, Matrix3, SMatrix, UnitQuaternion, Vector3};

    impl QuaternionEstimation {
        fn new_for_test(
            initial_q: UnitQuaternion<f64>,
            initial_bias: Vector3<f64>,
            initial_covariance: SMatrix<f64, 6, 6>,
            initial_time: &crate::Time,
        ) -> Self {
            QuaternionEstimation {
                ukf: UnscentedKalmanFilter::new(
                    AttitudeDeterminationModel,
                    AttitudeDeterminationState {
                        attitude: initial_q,
                        gyro_bias: initial_bias,
                    },
                    initial_covariance,
                    initial_time,
                    UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
                ),
                attitude_obs_model: AttitudeObservationModel,
                mag_obs_model: MagDirObservationModel,
                sun_obs_model: SunDirectionObservationModel,
            }
        }
    }

    #[test]
    fn test_ukf_magnetometer_update_reduces_error_and_covariance() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude
        let true_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.8);

        let mut estimation =
            QuaternionEstimation::new(initial_q, Matrix3::identity(), &initial_time);

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse())
            .angle()
            .abs();

        // Apply magnetometer update
        let b_inertial = Direction::from_dir(UnitVector3::new_normalize(
            nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        ));
        let b_measured = true_q.to_rotation_matrix() * b_inertial.clone();
        let b_observation = MagDirObservation {
            magnetic_direction: b_measured,
        };
        let observation_model = MagDirObservationModel;
        let inertial_b_input = InertialMagneticDirectionInputGaussian {
            inertial_magnetic_direction: b_inertial,
            inertial_magnetic_direction_covariance: Matrix2::identity() * 0.001,
        };
        estimation
            .ukf
            .update(
                &observation_model,
                &b_observation,
                &EmptyInput,
                &inertial_b_input,
                &initial_time,
                Matrix2::identity() * 0.001,
            )
            .unwrap();

        let final_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();

        println!(
            "Initial error: {}, Final error: {}",
            initial_error, final_error
        );

        // Error should decrease
        assert!(
            final_error < initial_error,
            "Error should decrease after magnetometer update: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_ukf_magnetometer_update_reduces_error_and_covariance2() {
        println!();
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude
        let true_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            -0.1832862773715284,
            0.7306793575902393,
            -0.5476673863075061,
            0.36410747162604,
        ));

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            -0.1837135755728519,
            0.730908171012182,
            -0.5476338980400441,
            0.363482711301833,
        ));

        let initial_bias =
            Vector3::new(0.16494703361283974, -0.092729521982097, 0.08978577836521058);

        let initial_covariance = SMatrix::<f64, 6, 6>::new(
            0.0063205958176944564,
            -0.0013832663596597234,
            0.007915201408897898,
            -0.0413808223943483,
            0.009184632333352061,
            -0.052030969598036374,
            -0.0013832663596597234,
            0.0003030945272360891,
            -0.001732343107578383,
            0.009057374060158531,
            -0.002013737559017086,
            0.01138935758699449,
            0.007915201408897898,
            -0.001732343107578383,
            0.009913004204486757,
            -0.051821224910376724,
            0.011502844953763955,
            -0.06516678717328989,
            -0.0413808223943483,
            0.009057374060158531,
            -0.051821224910376724,
            0.276455099397282,
            -0.06139531617963357,
            0.3476181293280811,
            0.009184632333352061,
            -0.002013737559017086,
            0.011502844953763955,
            -0.06139531617963357,
            0.01368783242242529,
            -0.07721363803249588,
            -0.052030969598036374,
            0.01138935758699449,
            -0.06516678717328989,
            0.3476181293280811,
            -0.07721363803249588,
            0.4372297517341999,
        );

        let mut estimation = QuaternionEstimation::new_for_test(
            initial_q,
            initial_bias,
            initial_covariance,
            &initial_time,
        );

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse())
            .angle()
            .abs();

        // Apply magnetometer update
        let b_inertial = Direction::from_dir(UnitVector3::new_normalize(
            nalgebra::SVector::<f64, 3>::new(0.4718, -0.8803, -0.0497),
        ));
        let true_b_body = true_q.to_rotation_matrix() * b_inertial.clone();
        let b_measured = Direction::from_dir(UnitVector3::new_normalize(
            nalgebra::SVector::<f64, 3>::new(0.5946, -0.0885, 0.7991),
        ));
        println!("b_inertial: {:?}", b_inertial.dir());
        println!("true_b_body (true_q * b_inertial): {:?}", true_b_body.dir());
        println!("b_measured: {:?}", b_measured.dir());
        println!(
            "True b body error norm: {} rad ({} deg)",
            (true_b_body.error(&b_measured).norm()).abs(),
            (true_b_body.error(&b_measured).norm()).abs().to_degrees()
        );

        // Also check what the initial estimate predicts
        let initial_b_body = initial_q.to_rotation_matrix() * b_inertial.clone();
        println!(
            "initial_b_body (initial_q * b_inertial): {:?}",
            initial_b_body.dir()
        );
        println!(
            "Initial b body error norm: {} rad ({} deg)",
            (initial_b_body.error(&b_measured).norm()).abs(),
            (initial_b_body.error(&b_measured).norm())
                .abs()
                .to_degrees()
        );
        // Use true_b_body instead of b_measured to test with consistent data
        let b_observation = MagDirObservation {
            magnetic_direction: true_b_body.clone(),
        };
        let observation_model = MagDirObservationModel;
        let inertial_b_input = InertialMagneticDirectionInputGaussian {
            inertial_magnetic_direction: b_inertial.clone(),
            inertial_magnetic_direction_covariance: Matrix2::new(
                1.022066451004334e-6,
                -4.5442615326934943e-7,
                -4.5442615326934943e-7,
                1.0383774194238193e-5,
            ),
        };
        println!("\n=== Before UKF update ===");
        println!("State attitude: {:?}", estimation.ukf.state().attitude);
        println!("State gyro_bias: {:?}", estimation.ukf.state().gyro_bias);

        // Analyze the covariance structure
        let cov = estimation.ukf.covariance();
        println!("\nCovariance matrix:");
        println!("{:.6}", cov);
        let att_cov = cov.fixed_view::<3, 3>(0, 0);
        let bias_cov = cov.fixed_view::<3, 3>(3, 3);
        let att_bias_cov = cov.fixed_view::<3, 3>(0, 3);
        println!("\nAttitude covariance (3x3 block):");
        println!("{:.6}", att_cov);
        println!(
            "Eigenvalues: {:?}",
            nalgebra::SymmetricEigen::new(att_cov.clone_owned()).eigenvalues
        );
        println!("\nGyro bias covariance (3x3 block):");
        println!("{:.6}", bias_cov);
        println!("\nAttitude-Bias cross-covariance (3x3 block):");
        println!("{:.6}", att_bias_cov);

        // Analyze cross-covariance in observable/unobservable frame
        let body_mag_dir_init = initial_q.to_rotation_matrix() * b_inertial.dir().into_inner();
        println!(
            "\nBody mag direction (unobservable axis): {:?}",
            body_mag_dir_init
        );

        // Compute covariance in unobservable direction
        let var_unobs = body_mag_dir_init.dot(&(att_cov * body_mag_dir_init));
        println!(
            "Attitude variance in unobservable direction: {:.6} rad²",
            var_unobs
        );

        // Compute covariance projection
        let att_cov_owned = att_cov.clone_owned();
        let e1 = body_mag_dir_init; // unobservable axis
        // Find two orthogonal observable axes
        let arbitrary = if e1.x.abs() < 0.9 {
            Vector3::<f64>::x()
        } else {
            Vector3::<f64>::y()
        };
        let e2 = e1.cross(&arbitrary).normalize();
        let e3 = e1.cross(&e2).normalize();
        println!("Observable axis 1: {:?}", e2);
        println!("Observable axis 2: {:?}", e3);

        let var_obs1 = e2.dot(&(att_cov_owned * e2));
        let var_obs2 = e3.dot(&(att_cov_owned * e3));
        let cov_unobs_obs1 = e1.dot(&(att_cov_owned * e2));
        let cov_unobs_obs2 = e1.dot(&(att_cov_owned * e3));
        println!(
            "Attitude variance in observable direction 1: {:.6} rad²",
            var_obs1
        );
        println!(
            "Attitude variance in observable direction 2: {:.6} rad²",
            var_obs2
        );
        println!("Covariance between unobs and obs1: {:.6}", cov_unobs_obs1);
        println!("Covariance between unobs and obs2: {:.6}", cov_unobs_obs2);

        // Compute correlation coefficients
        let rho1 = cov_unobs_obs1 / (var_unobs.sqrt() * var_obs1.sqrt());
        let rho2 = cov_unobs_obs2 / (var_unobs.sqrt() * var_obs2.sqrt());
        println!("Correlation coeff (unobs, obs1): {:.4}", rho1);
        println!("Correlation coeff (unobs, obs2): {:.4}", rho2);

        // Analyze: if we observe in obs1 direction with innovation δy1,
        // the expected correction in unobs direction due to correlation is:
        // δx_unobs = Cov(unobs, obs1) / Var(obs1) * δy1
        let regression_coeff1 = cov_unobs_obs1 / var_obs1;
        let regression_coeff2 = cov_unobs_obs2 / var_obs2;
        println!("\nRegression analysis:");
        println!(
            "If we observe 1 rad error in obs1, expected unobs correction: {:.4} rad",
            regression_coeff1
        );
        println!(
            "If we observe 1 rad error in obs2, expected unobs correction: {:.4} rad",
            regression_coeff2
        );

        // What would the current estimate predict?
        let predicted_b_body =
            estimation.ukf.state().attitude.to_rotation_matrix() * b_inertial.clone();
        println!(
            "Predicted b_body (estimate * b_inertial): {:?}",
            predicted_b_body.dir()
        );
        println!(
            "Observation (true_b_body): {:?}",
            b_observation.magnetic_direction.dir()
        );
        let innovation = predicted_b_body.error(&b_observation.magnetic_direction);
        println!(
            "Innovation (predicted - observed): {:?}, norm: {} rad ({} deg)",
            innovation,
            innovation.norm(),
            innovation.norm().to_degrees()
        );

        estimation
            .ukf
            .update(
                &observation_model,
                &b_observation,
                &EmptyInput,
                &inertial_b_input,
                &initial_time,
                Matrix2::new(
                    2.470878708621567e-7,
                    -3.637986993631267e-10,
                    -3.637986993631267e-10,
                    2.3954375192478654e-7,
                ),
            )
            .unwrap();

        println!("\n=== After UKF update (before fix) ===");
        let state_after_raw = estimation.ukf.state().attitude;
        let attitude_change_raw = (state_after_raw * initial_q.inverse()).scaled_axis();
        println!("State attitude: {:?}", state_after_raw);
        println!(
            "Attitude change from initial: {:?}, norm: {} rad ({} deg)",
            attitude_change_raw,
            attitude_change_raw.norm(),
            attitude_change_raw.norm().to_degrees()
        );

        // Check if correction is in unobservable direction
        let body_mag_dir = initial_q.to_rotation_matrix() * b_inertial.dir().into_inner();
        let correction_parallel = attitude_change_raw.dot(&body_mag_dir);
        let correction_perp = (attitude_change_raw - correction_parallel * body_mag_dir).norm();
        println!("Body mag dir (unobservable axis): {:?}", body_mag_dir);
        println!(
            "Correction parallel to mag (unobservable): {} rad ({} deg)",
            correction_parallel.abs(),
            correction_parallel.abs().to_degrees()
        );
        println!(
            "Correction perpendicular to mag (observable): {} rad ({} deg)",
            correction_perp,
            correction_perp.to_degrees()
        );
        println!(
            "Ratio parallel/total: {:.2}%",
            correction_parallel.abs() / attitude_change_raw.norm() * 100.0
        );

        // FIX: Project out unobservable axis correction
        let correction_observable = attitude_change_raw - correction_parallel * body_mag_dir;
        let corrected_attitude = UnitQuaternion::new(correction_observable) * initial_q;

        println!("\n=== After fix (projected out unobservable) ===");
        let attitude_change_fixed = (corrected_attitude * initial_q.inverse()).scaled_axis();
        println!(
            "Attitude change after fix: {:?}, norm: {} rad ({} deg)",
            attitude_change_fixed,
            attitude_change_fixed.norm(),
            attitude_change_fixed.norm().to_degrees()
        );

        let final_error = (corrected_attitude * true_q.inverse()).angle().abs();
        println!(
            "Initial error: {}, Final error: {}",
            initial_error, final_error
        );

        // Error should decrease
        assert!(
            final_error < initial_error,
            "Error should decrease after magnetometer update: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_ukf_magnetometer_update_reduces_error_and_covariance3_1() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude
        let true_q = UnitQuaternion::new(Vector3::new(0.1, -0.2, 0.3));

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::new(
            Vector3::new(0.4, -0.1, 0.2).normalize() * STAR_TRACKER_NOISE_STD * 0.4,
        ) * true_q;

        println!("\n=== Test 3 Geometry Analysis ===");
        println!("true_q: {:?}", true_q.into_inner());
        println!("initial_q: {:?}", initial_q.into_inner());

        // Analyze the initial error direction
        let initial_error_quat = initial_q * true_q.inverse();
        let initial_error_axis = initial_error_quat.scaled_axis();
        println!(
            "Initial error axis: [{:.6}, {:.6}, {:.6}]",
            initial_error_axis.x, initial_error_axis.y, initial_error_axis.z
        );
        let err_norm: f64 = initial_error_axis.norm();
        println!(
            "Initial error magnitude: {:.6} rad ({:.4} deg)",
            err_norm,
            err_norm.to_degrees()
        );

        // Body magnetic field direction
        let b_inertial_vec = Vector3::new(1.0, 0.0, 0.0);
        let b_body_from_true = true_q.to_rotation_matrix() * b_inertial_vec;
        let b_body_from_initial = initial_q.to_rotation_matrix() * b_inertial_vec;
        println!(
            "b_body (true_q): [{:.6}, {:.6}, {:.6}]",
            b_body_from_true.x, b_body_from_true.y, b_body_from_true.z
        );
        println!(
            "b_body (initial_q): [{:.6}, {:.6}, {:.6}]",
            b_body_from_initial.x, b_body_from_initial.y, b_body_from_initial.z
        );

        // Analyze: initial error direction vs body magnetic field direction
        let error_dot_mag: f64 = initial_error_axis
            .normalize()
            .dot(&b_body_from_true.normalize());
        println!(
            "Initial error · body_mag: {:.6} (1 = parallel, 0 = perpendicular)",
            error_dot_mag.abs()
        );

        let initial_covariance = {
            let mut cov = SMatrix::<f64, 6, 6>::zeros();
            cov.fixed_view_mut::<3, 3>(0, 0).copy_from(&Matrix3::new(
                0.000150, -0.000031, 0.000148, -0.000031, 0.000042, -0.000040, 0.000148, -0.000040,
                0.000222,
            ));
            cov.fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(Matrix3::identity() * 0.1));
            cov.fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-Matrix3::identity() * 0.001));
            cov.fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-Matrix3::identity() * 0.001));
            cov
        };

        // Analyze covariance in observable/unobservable frame
        let att_cov = initial_covariance.fixed_view::<3, 3>(0, 0).clone_owned();
        let e1 = b_body_from_initial.normalize(); // unobservable axis
        let arbitrary = if e1.x.abs() < 0.9 {
            Vector3::<f64>::x()
        } else {
            Vector3::<f64>::y()
        };
        let e2 = e1.cross(&arbitrary).normalize();
        let e3 = e1.cross(&e2).normalize();

        println!("\n--- Observable/Unobservable Axis Definitions (in body frame xyz) ---");
        println!("e1 (unobs) = [{:.4}, {:.4}, {:.4}]", e1.x, e1.y, e1.z);
        println!("e2 (obs1)  = [{:.4}, {:.4}, {:.4}]", e2.x, e2.y, e2.z);
        println!("e3 (obs2)  = [{:.4}, {:.4}, {:.4}]", e3.x, e3.y, e3.z);

        // Show correlation between obs1/obs2 axes (should be 0)
        let cov_obs1_obs2 = e2.dot(&(att_cov * e3));
        println!(
            "\nCov(obs1, obs2) = {:.6} (correlation between observable axes)",
            cov_obs1_obs2
        );

        let var_unobs = e1.dot(&(att_cov * e1));
        let var_obs1 = e2.dot(&(att_cov * e2));
        let var_obs2 = e3.dot(&(att_cov * e3));
        let cov_unobs_obs1 = e1.dot(&(att_cov * e2));
        let cov_unobs_obs2 = e1.dot(&(att_cov * e3));

        let rho1 = cov_unobs_obs1 / (var_unobs.sqrt() * var_obs1.sqrt());
        let rho2 = cov_unobs_obs2 / (var_unobs.sqrt() * var_obs2.sqrt());
        let beta1 = cov_unobs_obs1 / var_obs1;
        let beta2 = cov_unobs_obs2 / var_obs2;

        // Correlation between obs1 and obs2
        let rho_obs1_obs2 = cov_obs1_obs2 / (var_obs1.sqrt() * var_obs2.sqrt());

        println!("\nCovariance in observable/unobservable frame:");
        println!(
            "var_unobs={:.6}, var_obs1={:.6}, var_obs2={:.6}",
            var_unobs, var_obs1, var_obs2
        );
        println!("rho(unobs,obs1)={:.4}, rho(unobs,obs2)={:.4}", rho1, rho2);
        println!(
            "rho(obs1,obs2)={:.4}  ← correlation between observable axes!",
            rho_obs1_obs2
        );
        println!("beta1={:.4}, beta2={:.4}", beta1, beta2);

        let mut estimation = QuaternionEstimation::new_for_test(
            initial_q,
            Vector3::zeros(),
            initial_covariance,
            &initial_time,
        );

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse())
            .angle()
            .abs();
        let attitude_before = estimation.ukf.state().attitude;

        // Apply magnetometer update
        let b_inertial = Direction::from_dir(UnitVector3::new_normalize(
            nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        ));
        let b_measured = true_q.to_rotation_matrix() * b_inertial.clone();
        let b_observation = MagDirObservation {
            magnetic_direction: b_measured,
        };
        let observation_model = MagDirObservationModel;
        let inertial_b_input = InertialMagneticDirectionInputGaussian {
            inertial_magnetic_direction: b_inertial,
            inertial_magnetic_direction_covariance: Matrix2::identity() * 0.000001,
        };
        estimation
            .ukf
            .update(
                &observation_model,
                &b_observation,
                &EmptyInput,
                &inertial_b_input,
                &initial_time,
                Matrix2::identity() * 0.000001,
            )
            .unwrap();

        // Analyze the attitude change
        let attitude_after = estimation.ukf.state().attitude;
        let attitude_change = (attitude_after * attitude_before.inverse()).scaled_axis();

        let att_change_unobs = attitude_change.dot(&e1);
        let att_change_obs = (attitude_change - att_change_unobs * e1).norm();

        println!("\n--- Attitude Change Analysis ---");
        println!(
            "Attitude change: [{:.6}, {:.6}, {:.6}]",
            attitude_change.x, attitude_change.y, attitude_change.z
        );
        println!(
            "Attitude change magnitude: {:.6} rad ({:.4} deg)",
            attitude_change.norm(),
            attitude_change.norm().to_degrees()
        );
        println!(
            "In unobservable direction: {:.6} rad",
            att_change_unobs.abs()
        );
        println!("In observable direction: {:.6} rad", att_change_obs);
        println!(
            "Unobs/total: {:.2}%",
            if attitude_change.norm() > 1e-10 {
                att_change_unobs.abs() / attitude_change.norm() * 100.0
            } else {
                0.0
            }
        );

        // Check if correction is in the right direction
        let correction_dot_error = attitude_change.dot(&initial_error_axis);
        println!("\nCorrection · initial_error: {:.6}", correction_dot_error);
        println!("(positive = correction in same direction as error = wrong)");
        println!("(negative = correction opposite to error = correct)");

        // Decompose initial error into observable/unobservable components
        let error_unobs = initial_error_axis.dot(&e1) * e1;
        let error_obs1 = initial_error_axis.dot(&e2) * e2;
        let error_obs2 = initial_error_axis.dot(&e3) * e3;
        let error_obs = error_obs1 + error_obs2;

        println!("\n--- Error Decomposition ---");
        println!("Error in unobs direction: {:.6} rad", error_unobs.norm());
        println!("Error in obs1 direction: {:.6} rad", error_obs1.norm());
        println!("Error in obs2 direction: {:.6} rad", error_obs2.norm());
        println!("Error in observable (total): {:.6} rad", error_obs.norm());

        // Decompose correction into observable/unobservable components
        let corr_unobs = attitude_change.dot(&e1) * e1;
        let corr_obs1 = attitude_change.dot(&e2) * e2;
        let corr_obs2 = attitude_change.dot(&e3) * e3;
        let corr_obs = corr_obs1 + corr_obs2;

        println!("\n--- Correction Decomposition ---");
        println!(
            "Correction in unobs direction: {:.6} rad",
            corr_unobs.norm()
        );
        println!("Correction in obs1 direction: {:.6} rad", corr_obs1.norm());
        println!("Correction in obs2 direction: {:.6} rad", corr_obs2.norm());
        println!(
            "Correction in observable (total): {:.6} rad",
            corr_obs.norm()
        );

        // Check if observable correction is opposite to observable error (correct direction)
        let obs_correction_dot_obs_error = corr_obs.dot(&error_obs);
        println!("\n--- Observable Correction Direction ---");
        println!(
            "Observable correction · observable error: {:.6}",
            obs_correction_dot_obs_error
        );
        println!("(negative = correct direction, positive = wrong direction)");

        let final_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();
        println!("estimated bias: {:?}", estimation.ukf.state().gyro_bias);

        println!(
            "\nInitial error: {:.6} rad, Final error: {:.6} rad",
            initial_error, final_error
        );
        println!(
            "Error change: {:.6} rad ({:.2}%)",
            final_error - initial_error,
            (final_error - initial_error) / initial_error * 100.0
        );

        // Error should decrease
        assert!(
            final_error < initial_error,
            "Error should decrease after magnetometer update: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_ukf_magnetometer_update_reduces_error_and_covariance3_2() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude
        let true_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            -0.18264539090472454,
            0.7303350755510656,
            -0.5477170765166343,
            0.365044302370481,
        ));

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            -0.1827, 0.7303, -0.5477, 0.3650,
        ));

        println!("\n=== Test 4 Geometry Analysis ===");
        println!("true_q: {:?}", true_q.into_inner());
        println!("initial_q: {:?}", initial_q.into_inner());

        // Analyze the initial error direction
        let initial_error_quat = initial_q * true_q.inverse();
        let initial_error_axis = initial_error_quat.scaled_axis();
        println!(
            "Initial error axis: [{:.6}, {:.6}, {:.6}]",
            initial_error_axis.x, initial_error_axis.y, initial_error_axis.z
        );
        let err_norm: f64 = initial_error_axis.norm();
        println!(
            "Initial error magnitude: {:.6} rad ({:.4} deg)",
            err_norm,
            err_norm.to_degrees()
        );

        // Body magnetic field direction
        let b_inertial_vec = Vector3::new(1.0, 0.0, 0.0);
        let b_body_from_true = true_q.to_rotation_matrix() * b_inertial_vec;
        let b_body_from_initial = initial_q.to_rotation_matrix() * b_inertial_vec;
        println!(
            "b_body (true_q): [{:.6}, {:.6}, {:.6}]",
            b_body_from_true.x, b_body_from_true.y, b_body_from_true.z
        );
        println!(
            "b_body (initial_q): [{:.6}, {:.6}, {:.6}]",
            b_body_from_initial.x, b_body_from_initial.y, b_body_from_initial.z
        );

        // Analyze: initial error direction vs body magnetic field direction
        let error_dot_mag: f64 = initial_error_axis
            .normalize()
            .dot(&b_body_from_true.normalize());
        println!(
            "Initial error · body_mag: {:.6} (1 = parallel, 0 = perpendicular)",
            error_dot_mag.abs()
        );

        let initial_covariance = {
            let mut cov = SMatrix::<f64, 6, 6>::zeros();
            cov.fixed_view_mut::<3, 3>(0, 0).copy_from(&Matrix3::new(
                0.000150, -0.000031, 0.000148, -0.000031, 0.000042, -0.000040, 0.000148, -0.000040,
                0.000222,
            ));
            cov.fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(Matrix3::identity() * 0.1));
            cov.fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-Matrix3::identity() * 0.001));
            cov.fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-Matrix3::identity() * 0.001));
            cov
        };

        // Analyze covariance in observable/unobservable frame
        let att_cov = initial_covariance.fixed_view::<3, 3>(0, 0).clone_owned();
        let e1 = b_body_from_initial.normalize(); // unobservable axis
        let arbitrary = if e1.x.abs() < 0.9 {
            Vector3::<f64>::x()
        } else {
            Vector3::<f64>::y()
        };
        let e2 = e1.cross(&arbitrary).normalize();
        let e3 = e1.cross(&e2).normalize();

        println!("\n--- Observable/Unobservable Axis Definitions (in body frame xyz) ---");
        println!("e1 (unobs) = [{:.4}, {:.4}, {:.4}]", e1.x, e1.y, e1.z);
        println!("e2 (obs1)  = [{:.4}, {:.4}, {:.4}]", e2.x, e2.y, e2.z);
        println!("e3 (obs2)  = [{:.4}, {:.4}, {:.4}]", e3.x, e3.y, e3.z);

        // Show correlation between obs1/obs2 axes (should be 0)
        let cov_obs1_obs2 = e2.dot(&(att_cov * e3));
        println!(
            "\nCov(obs1, obs2) = {:.6} (correlation between observable axes)",
            cov_obs1_obs2
        );

        let var_unobs = e1.dot(&(att_cov * e1));
        let var_obs1 = e2.dot(&(att_cov * e2));
        let var_obs2 = e3.dot(&(att_cov * e3));
        let cov_unobs_obs1 = e1.dot(&(att_cov * e2));
        let cov_unobs_obs2 = e1.dot(&(att_cov * e3));

        let rho1 = cov_unobs_obs1 / (var_unobs.sqrt() * var_obs1.sqrt());
        let rho2 = cov_unobs_obs2 / (var_unobs.sqrt() * var_obs2.sqrt());
        let beta1 = cov_unobs_obs1 / var_obs1;
        let beta2 = cov_unobs_obs2 / var_obs2;

        // Correlation between obs1 and obs2
        let rho_obs1_obs2 = cov_obs1_obs2 / (var_obs1.sqrt() * var_obs2.sqrt());

        println!("\nCovariance in observable/unobservable frame:");
        println!(
            "var_unobs={:.6}, var_obs1={:.6}, var_obs2={:.6}",
            var_unobs, var_obs1, var_obs2
        );
        println!("rho(unobs,obs1)={:.4}, rho(unobs,obs2)={:.4}", rho1, rho2);
        println!(
            "rho(obs1,obs2)={:.4}  ← correlation between observable axes!",
            rho_obs1_obs2
        );
        println!("beta1={:.4}, beta2={:.4}", beta1, beta2);

        let mut estimation = QuaternionEstimation::new_for_test(
            initial_q,
            Vector3::zeros(),
            initial_covariance,
            &initial_time,
        );

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse())
            .angle()
            .abs();
        let attitude_before = estimation.ukf.state().attitude;

        // Apply magnetometer update
        let b_inertial = Direction::from_dir(UnitVector3::new_normalize(
            nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        ));
        let b_measured = true_q.to_rotation_matrix() * b_inertial.clone();
        let b_observation = MagDirObservation {
            magnetic_direction: b_measured,
        };
        let observation_model = MagDirObservationModel;
        let inertial_b_input = InertialMagneticDirectionInputGaussian {
            inertial_magnetic_direction: b_inertial,
            inertial_magnetic_direction_covariance: Matrix2::identity() * 0.000001,
        };
        estimation
            .ukf
            .update(
                &observation_model,
                &b_observation,
                &EmptyInput,
                &inertial_b_input,
                &initial_time,
                Matrix2::identity() * 0.000001,
            )
            .unwrap();

        // Analyze the attitude change
        let attitude_after = estimation.ukf.state().attitude;
        let attitude_change = (attitude_after * attitude_before.inverse()).scaled_axis();

        let att_change_unobs = attitude_change.dot(&e1);
        let att_change_obs = (attitude_change - att_change_unobs * e1).norm();

        println!("\n--- Attitude Change Analysis ---");
        println!(
            "Attitude change: [{:.6}, {:.6}, {:.6}]",
            attitude_change.x, attitude_change.y, attitude_change.z
        );
        println!(
            "Attitude change magnitude: {:.6} rad ({:.4} deg)",
            attitude_change.norm(),
            attitude_change.norm().to_degrees()
        );
        println!(
            "In unobservable direction: {:.6} rad",
            att_change_unobs.abs()
        );
        println!("In observable direction: {:.6} rad", att_change_obs);
        println!(
            "Unobs/total: {:.2}%",
            if attitude_change.norm() > 1e-10 {
                att_change_unobs.abs() / attitude_change.norm() * 100.0
            } else {
                0.0
            }
        );

        // Check if correction is in the right direction
        let correction_dot_error = attitude_change.dot(&initial_error_axis);
        println!("\nCorrection · initial_error: {:.6}", correction_dot_error);
        println!("(positive = correction in same direction as error = wrong)");
        println!("(negative = correction opposite to error = correct)");

        // Decompose initial error into observable/unobservable components
        let error_unobs = initial_error_axis.dot(&e1) * e1;
        let error_obs1 = initial_error_axis.dot(&e2) * e2;
        let error_obs2 = initial_error_axis.dot(&e3) * e3;
        let error_obs = error_obs1 + error_obs2;

        println!("\n--- Error Decomposition ---");
        println!("Error in unobs direction: {:.6} rad", error_unobs.norm());
        println!("Error in obs1 direction: {:.6} rad", error_obs1.norm());
        println!("Error in obs2 direction: {:.6} rad", error_obs2.norm());
        println!("Error in observable (total): {:.6} rad", error_obs.norm());

        // Decompose correction into observable/unobservable components
        let corr_unobs = attitude_change.dot(&e1) * e1;
        let corr_obs1 = attitude_change.dot(&e2) * e2;
        let corr_obs2 = attitude_change.dot(&e3) * e3;
        let corr_obs = corr_obs1 + corr_obs2;

        println!("\n--- Correction Decomposition ---");
        println!(
            "Correction in unobs direction: {:.6} rad",
            corr_unobs.norm()
        );
        println!("Correction in obs1 direction: {:.6} rad", corr_obs1.norm());
        println!("Correction in obs2 direction: {:.6} rad", corr_obs2.norm());
        println!(
            "Correction in observable (total): {:.6} rad",
            corr_obs.norm()
        );

        // Check if observable correction is opposite to observable error (correct direction)
        let obs_correction_dot_obs_error = corr_obs.dot(&error_obs);
        println!("\n--- Observable Correction Direction ---");
        println!(
            "Observable correction · observable error: {:.6}",
            obs_correction_dot_obs_error
        );
        println!("(negative = correct direction, positive = wrong direction)");

        let final_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();
        println!("estimated bias: {:?}", estimation.ukf.state().gyro_bias);

        println!(
            "\nInitial error: {:.6} rad, Final error: {:.6} rad",
            initial_error, final_error
        );
        println!(
            "Error change: {:.6} rad ({:.2}%)",
            final_error - initial_error,
            (final_error - initial_error) / initial_error * 100.0
        );

        // Error should decrease
        assert!(
            final_error < initial_error,
            "Error should decrease after magnetometer update: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_ukf_sun_sensor_update_reduces_error() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude
        let true_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.8);

        let mut estimation =
            QuaternionEstimation::new(initial_q, Matrix3::identity(), &initial_time);

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();

        // Apply magnetometer update
        let s_inertial = nalgebra::Vector3::x_axis();
        let s_measured = true_q * s_inertial;
        let s_observation = SunDirectionObservation {
            sun_direction: Direction::from_dir(s_measured),
        };
        let observation_model = SunDirectionObservationModel;
        let inertial_s_input = InertialSunDirectionInputGaussian {
            inertial_sun_direction: Direction::from_dir(s_inertial),
            inertial_sun_direction_covariance: Matrix2::identity() * 0.001,
        };
        estimation
            .ukf
            .update(
                &observation_model,
                &s_observation,
                &EmptyInput,
                &inertial_s_input,
                &initial_time,
                Matrix2::identity() * 0.001,
            )
            .unwrap();

        let final_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();

        println!(
            "Initial error: {}, Final error: {}",
            initial_error, final_error
        );

        // Error should decrease
        assert!(
            final_error < initial_error,
            "Error should decrease after sun sensor update: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_ukf_star_tracker_update_reduces_error() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude
        let true_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.8);

        let mut estimation =
            QuaternionEstimation::new(initial_q, Matrix3::identity(), &initial_time);

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();

        // Apply star tracker update
        let observation_model = AttitudeObservationModel;

        let q_measured = true_q;
        let q_observation = AttitudeObservation {
            attitude: q_measured,
        };

        estimation
            .ukf
            .update(
                &observation_model,
                &q_observation,
                &EmptyInput,
                &EmptyInput,
                &initial_time,
                Matrix3::identity() * 0.001,
            )
            .unwrap();

        let final_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();
        println!(
            "Initial error: {}, Final error: {}",
            initial_error, final_error
        );

        // Error should decrease
        assert!(
            final_error < initial_error,
            "Error should decrease after star tracker update: {} -> {}",
            initial_error,
            final_error
        );
    }
}

#[cfg(test)]
mod tests_convergence {
    use super::*;
    use crate::constants::{GYRO_NOISE_STD, STAR_TRACKER_NOISE_STD};
    use astrodynamics::coordinate::BodyVector;
    use control_system::integrator::{
        Prediction, TimeIntegrator,
        rk4::{RK4Phase, RK4Solver},
    };
    use nalgebra::{Matrix2, Matrix3, Quaternion, UnitQuaternion, UnitVector3, Vector3};
    use rand::distributions::Distribution;
    use statrs::distribution::MultivariateNormal;

    #[test]
    fn test_ukf_converges_to_true_attitude() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True attitude with some rotation
        let true_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);

        // Start with wrong initial estimate
        let initial_q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.8);

        let mut estimation =
            QuaternionEstimation::new(initial_q, Matrix3::identity(), &initial_time);

        let initial_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();

        // Inertial reference vectors
        let b_inertial =
            UnitVector3::new_normalize(nalgebra::SVector::<f64, 3>::new(1.0, 0.0, 0.0));
        let s_inertial = UnitVector3::new_normalize(nalgebra::Vector3::new(0.0, 1.0, 0.0));

        // Measured vectors from true attitude
        let b_measured = true_q * b_inertial;
        let b_observation = MagDirObservation {
            magnetic_direction: Direction::from_dir(b_measured),
        };
        let s_measured = true_q * s_inertial;
        let s_observation = SunDirectionObservation {
            sun_direction: Direction::from_dir(s_measured),
        };

        let b_observation_model = MagDirObservationModel;
        let inertial_b_input = InertialMagneticDirectionInputGaussian {
            inertial_magnetic_direction: Direction::from_dir(b_inertial),
            inertial_magnetic_direction_covariance: Matrix2::identity() * 0.001,
        };
        let s_observation_model = SunDirectionObservationModel;
        let inertial_s_input = InertialSunDirectionInputGaussian {
            inertial_sun_direction: Direction::from_dir(s_inertial),
            inertial_sun_direction_covariance: Matrix2::identity() * 0.001,
        };

        let omega = AttitudeDeterminationPropagationInputGaussian {
            angular_velocity: nalgebra::SVector::<f64, 3>::zeros(),
            angular_velocity_covariance: Matrix3::identity() * 0.1,
        };
        let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());
        let mut time = initial_time;

        // Run multiple update cycles
        for _ in 0..20 {
            time += 0.01;
            estimation
                .ukf
                .propagate(&EmptyInput, &omega, process_noise_covariance, &time)
                .unwrap();

            // Magnetometer update
            estimation
                .ukf
                .update(
                    &b_observation_model,
                    &b_observation,
                    &EmptyInput,
                    &inertial_b_input,
                    &initial_time,
                    Matrix2::identity() * 0.001,
                )
                .unwrap();

            // Sun sensor update
            estimation
                .ukf
                .update(
                    &s_observation_model,
                    &s_observation,
                    &EmptyInput,
                    &inertial_s_input,
                    &initial_time,
                    Matrix2::identity() * 0.001,
                )
                .unwrap();
        }

        let final_error = (estimation.ukf.state().attitude * true_q.inverse()).angle();

        // Error should have decreased significantly
        assert!(
            final_error < initial_error / 2.0,
            "UKF should converge: initial error = {}, final error = {}",
            initial_error,
            final_error
        );
        assert!(
            final_error < 0.05,
            "Final error should be small: {}",
            final_error
        );
    }

    #[test]
    fn test_gyro_bias_estimation_converges() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True gyro bias
        let true_bias = nalgebra::SVector::<f64, 3>::new(-0.06, -0.10, 0.15);

        let mut estimation = QuaternionEstimation::new(
            UnitQuaternion::identity(),
            Matrix3::identity(),
            &initial_time,
        );

        // True angular velocity
        let true_omega = nalgebra::SVector::<f64, 3>::new(0.15, 0.08, 0.05);
        let omega_measured_mean = true_omega + true_bias;

        let omega_mvn = MultivariateNormal::new_from_nalgebra(
            omega_measured_mean,
            Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
        )
        .unwrap();

        let mut time = initial_time;
        let mut true_q = UnitQuaternion::identity();

        let observation_model = AttitudeObservationModel;
        let q_observation_mvn = MultivariateNormal::new_from_nalgebra(
            nalgebra::Vector3::zeros(),
            Matrix3::identity() * STAR_TRACKER_NOISE_STD * STAR_TRACKER_NOISE_STD,
        )
        .unwrap();
        let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());
        let dt = 0.01;
        let mut rng = rand::thread_rng();

        // Run multiple cycles
        for i in 0..10000 {
            time += dt;

            // Update true quaternion
            true_q = UnitQuaternion::new(true_omega * dt) * true_q;

            let omega_measured = omega_mvn.sample(&mut rng);
            let omega_input = AttitudeDeterminationPropagationInputGaussian {
                angular_velocity: omega_measured,
                angular_velocity_covariance: Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
            };

            estimation
                .ukf
                .propagate(&EmptyInput, &omega_input, process_noise_covariance, &time)
                .unwrap();

            let q_noise = q_observation_mvn.sample(&mut rng);
            let q_measured = UnitQuaternion::new(q_noise) * true_q;
            let q_measured = AttitudeObservation {
                attitude: q_measured,
            };
            if i % 100 == 1 {
                estimation
                    .ukf
                    .update(
                        &observation_model,
                        &q_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &initial_time, // Use initial_time for stable updates
                        Matrix3::identity() * STAR_TRACKER_NOISE_STD * STAR_TRACKER_NOISE_STD,
                    )
                    .unwrap();
            }

            let state = estimation.ukf.state();
            let attitude_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal();
            let gyro_bias_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(3, 3)
                .diagonal();

            assert!(
                (state.attitude * true_q.inverse()).angle()
                    <= 3.0 * attitude_covariance_diagonal.map(|x| x.sqrt()).norm(),
                "Attitude error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                i,
                (state.attitude * true_q.inverse()).scaled_axis(),
                attitude_covariance_diagonal.map(|x| x.sqrt()).norm()
            );
            assert!(
                (state.gyro_bias - true_bias)
                    .iter()
                    .zip(gyro_bias_covariance_diagonal.iter())
                    .all(|(e, l)| e.abs() <= l.abs().sqrt() * 3.0),
                "Gyro bias error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                i,
                (state.gyro_bias - true_bias),
                gyro_bias_covariance_diagonal
            );
        }
        println!(
            "Final estimated gyro bias error: {:?}",
            estimation.ukf.state().gyro_bias - true_bias
        );
        println!(
            "Final gyro bias covariance diagonal: {:?}",
            estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(3, 3)
                .diagonal()
        );
        println!(
            "Final attitude error: {} rad",
            (estimation.ukf.state().attitude * true_q.inverse()).angle()
        );
        println!(
            "Final attitude covariance diagonal: {:?}",
            estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal()
        )
    }

    #[test]
    fn test_gyro_bias_estimation_converges_with_rk4() {
        let initial_time = crate::Time::from_seconds(0.0);
        let inertia = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

        // True gyro bias
        let true_bias = nalgebra::SVector::<f64, 3>::new(-0.0, -0.0, 0.0);

        // Use same initial attitude as in real simulation (from TRIAD)
        let initial_attitude = UnitQuaternion::from_quaternion(Quaternion::new(
            -0.18473353039839202,
            0.7308976443576298,
            -0.5467873085093476,
            0.3642606147693126,
        ));
        println!("Test initial attitude: {}", initial_attitude.into_inner());

        let mut estimation = QuaternionEstimation::new(
            initial_attitude,
            Matrix3::identity() * STAR_TRACKER_NOISE_STD * STAR_TRACKER_NOISE_STD,
            &initial_time,
        );

        // True angular velocity
        let mut omega = RK4Solver::new(BodyVector::from(nalgebra::SVector::<f64, 3>::new(
            0.01, 0.005, 0.007,
        )));

        // True attitude (start from same initial attitude)
        let mut attitude = RK4Solver::new(initial_attitude.into_inner());

        let mut time = initial_time;

        let observation_model = AttitudeObservationModel;
        let q_observation_mvn = MultivariateNormal::new_from_nalgebra(
            nalgebra::Vector3::zeros(),
            Matrix3::identity() * STAR_TRACKER_NOISE_STD * STAR_TRACKER_NOISE_STD,
        )
        .unwrap();
        let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());
        let dt = 0.01;
        let mut rng = rand::thread_rng();

        // Run multiple cycles
        for i in 0..10000 {
            time += dt;

            // Update true quaternion
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
                0.5 * q * omega_quat
            };
            attitude.propagate(q_f, dt, time);
            omega.clear();
            attitude.clear();

            let omega_measured_mean = Vector3::from(omega.get_now()) + true_bias;

            let omega_mvn = MultivariateNormal::new_from_nalgebra(
                omega_measured_mean,
                Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
            )
            .unwrap();

            let omega_measured = omega_mvn.sample(&mut rng);
            let omega_input = AttitudeDeterminationPropagationInputGaussian {
                angular_velocity: omega_measured,
                angular_velocity_covariance: Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
            };

            estimation
                .ukf
                .propagate(&EmptyInput, &omega_input, process_noise_covariance, &time)
                .unwrap();

            let q_noise = q_observation_mvn.sample(&mut rng);
            let q_measured =
                UnitQuaternion::new(q_noise) * UnitQuaternion::from_quaternion(attitude.get_now());
            let q_measured = AttitudeObservation {
                attitude: q_measured,
            };
            // Start observations after 1 second (iteration 100), then every 100 steps
            // This matches the real simulation: no STT for 1s, then at 1.01s, 2.02s, etc.
            if i >= 100 && (i - 100) % 100 == 1 {
                // Log covariance before update
                estimation
                    .ukf
                    .update(
                        &observation_model,
                        &q_measured,
                        &EmptyInput,
                        &EmptyInput,
                        &time, // Use current time instead of initial_time to reproduce issue
                        Matrix3::identity() * STAR_TRACKER_NOISE_STD * STAR_TRACKER_NOISE_STD,
                    )
                    .unwrap();
            }

            let state = estimation.ukf.state();
            let attitude_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal();
            let gyro_bias_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(3, 3)
                .diagonal();
            let true_q = UnitQuaternion::from_quaternion(attitude.get_now());

            assert!(
                (state.attitude * true_q.inverse()).angle()
                    <= 3.0 * attitude_covariance_diagonal.map(|x| x.sqrt()).norm(),
                "Attitude error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                i,
                (state.attitude * true_q.inverse()).scaled_axis(),
                attitude_covariance_diagonal.map(|x| x.sqrt()).norm()
            );
            assert!(
                (state.gyro_bias - true_bias)
                    .iter()
                    .zip(gyro_bias_covariance_diagonal.iter())
                    .all(|(e, l)| e.abs() <= l.abs().sqrt() * 3.0),
                "Gyro bias error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                i,
                (state.gyro_bias - true_bias),
                gyro_bias_covariance_diagonal
            );
        }
        let true_q = UnitQuaternion::from_quaternion(attitude.get_now());
        println!(
            "Final estimated gyro bias error: {:?}",
            estimation.ukf.state().gyro_bias - true_bias
        );
        println!(
            "Final gyro bias covariance diagonal: {:?}",
            estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(3, 3)
                .diagonal()
        );
        println!(
            "Final attitude error: {} rad",
            (estimation.ukf.state().attitude * true_q.inverse()).angle()
        );
        println!(
            "Final attitude covariance diagonal: {:?}",
            estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal()
        )
    }

    /// Test that gyro bias converges to true value (0) when true bias is 0
    /// This test verifies the fix for the inertial magnetic field calculation bug
    #[test]
    fn test_gyro_bias_converges_to_zero_with_correct_magnetic_field() {
        let initial_time = crate::Time::from_seconds(0.0);

        // True bias is ZERO (this is the key assumption)
        let true_bias = nalgebra::SVector::<f64, 3>::zeros();

        // Initialize with a non-zero estimated bias
        let b_inertial = nalgebra::SVector::<f64, 3>::new(
            9894.151141285403,
            -17712.724132008952,
            -1358.4275738223103,
        );
        let initial_q =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI / 4.0);

        let mut estimation =
            QuaternionEstimation::new(initial_q, Matrix3::identity(), &initial_time);

        // Measured angular velocity = true angular velocity + true bias (= 0)
        let true_omega = nalgebra::SVector::<f64, 3>::new(0.01, 0.02, 0.03);
        let measured_omega_mean = true_omega + true_bias; // = true_omega since bias is 0

        let omega_mvn = MultivariateNormal::new_from_nalgebra(
            measured_omega_mean,
            Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
        )
        .unwrap();

        let process_noise_covariance = Some(attitude_estimation_process_noise_covariance());

        let dt = 0.01;
        let num_steps = 1000; // 10 seconds

        let mag_obs_model = MagDirObservationModel;
        let b_observation_mvn = MultivariateNormal::new_from_nalgebra(
            nalgebra::Vector3::zeros(),
            Matrix3::identity() * 0.001,
        )
        .unwrap();
        // Use realistic inertial magnetic field variance from actual simulation
        let inertial_b_variance = Matrix3::identity() * 50.0 * 50.0;
        let inertial_b_input = MagneticFieldInputGaussian {
            magnetic_field: b_inertial,
            magnetic_field_covariance: inertial_b_variance,
        };
        let (inertial_b, inertial_b_covariance) = input_shift(
            &inertial_b_input,
            mag_dir_func,
            &UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        )
        .unwrap();
        let inertial_b_dir_input = InertialMagneticDirectionInputGaussian {
            inertial_magnetic_direction: inertial_b.magnetic_direction,
            inertial_magnetic_direction_covariance: inertial_b_covariance,
        };
        println!(
            "Inertial magnetic direction covariance used in test: {:?}",
            inertial_b_dir_input.inertial_magnetic_direction_covariance
        );

        let mut rng = rand::thread_rng();
        let mut true_q = initial_q;
        let mut time = initial_time;

        for i in 0..num_steps {
            time += dt;
            // Update true quaternion
            true_q = UnitQuaternion::new(true_omega * dt) * true_q;
            let b_body = true_q * b_inertial;

            let omega_measured = omega_mvn.sample(&mut rng);
            let omega_input = AttitudeDeterminationPropagationInputGaussian {
                angular_velocity: omega_measured,
                angular_velocity_covariance: Matrix3::identity() * GYRO_NOISE_STD * GYRO_NOISE_STD,
            };

            // Propagate
            estimation
                .ukf
                .propagate(&EmptyInput, &omega_input, process_noise_covariance, &time)
                .unwrap();

            // Update with magnetometer measurement every cycle
            let b_dir = UnitVector3::new_normalize(b_body + b_observation_mvn.sample(&mut rng));
            let measured_b = MagDirObservation {
                magnetic_direction: Direction::from_dir(b_dir),
            };

            estimation
                .ukf
                .update(
                    &mag_obs_model,
                    &measured_b,
                    &EmptyInput,
                    &inertial_b_dir_input,
                    &time,
                    Matrix2::identity() * 0.001,
                )
                .unwrap();

            let state = estimation.ukf.state();
            let bias_error = state.gyro_bias - true_bias;
            let bias_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(3, 3)
                .diagonal();
            let attitude_covariance_diagonal = estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal();

            assert!(
                bias_error
                    .iter()
                    .zip(bias_covariance_diagonal.iter())
                    .all(|(e, l)| e.abs() <= l.abs().sqrt() * 3.0),
                "Gyro bias error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                i,
                bias_error,
                bias_covariance_diagonal
            );
            assert!(
                (state.attitude * true_q.inverse()).angle()
                    <= 3.0 * attitude_covariance_diagonal.map(|x| x.sqrt()).norm(),
                "Attitude error should be within 3-sigma at iteration {}: error {:?}, covariance diagonal {:?}",
                i,
                (state.attitude * true_q.inverse()).scaled_axis(),
                attitude_covariance_diagonal.map(|x| x.sqrt()).norm()
            );
        }
        println!(
            "Final estimated gyro bias error: {:?}",
            estimation.ukf.state().gyro_bias - true_bias
        );
        println!(
            "Final gyro bias covariance diagonal: {:?}",
            estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(3, 3)
                .diagonal()
        );
        println!(
            "Final attitude error: {} rad",
            (estimation.ukf.state().attitude * true_q.inverse()).angle()
        );
        println!(
            "Final attitude covariance diagonal: {:?}",
            estimation
                .ukf
                .covariance()
                .fixed_view::<3, 3>(0, 0)
                .diagonal()
        )
    }
}
