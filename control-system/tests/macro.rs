use control_system::{
    components::Direction,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::{EmptyInput, GaussianInputTrait, NominalStructTrait, ValueStructTrait},
};
use control_system_macro::{EstimationGaussianInput, EstimationOutputStruct, EstimationState};
use nalgebra::{SMatrix, SVector, Vector2, Vector3};

#[derive(EstimationGaussianInput, Clone, Debug)]
pub struct AttitudeInput {
    omega_measured: Vector3<f64>,
}

#[derive(EstimationState, Clone, Debug)]
pub struct DirectionState {
    sun_direction: Direction,
    magnetic_field: Vector3<f64>,
}

#[derive(EstimationOutputStruct, Clone, Debug)]
pub struct SunObservation {
    sun_direction: Direction,
}

struct DirectionStateModel;

impl PropagationModel for DirectionStateModel {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = AttitudeInput;
    type Time = f64;
    type Dt = f64;
    fn propagate(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
        dt: &Self::Dt,
    ) -> Self::State {
        let axisangle = gaussian_input.omega_measured * *dt;
        let rot = nalgebra::Rotation3::new(axisangle);
        let new_magnetic_field = rot * state.magnetic_field;
        let new_sun_direction = rot * state.sun_direction.clone();
        DirectionState {
            sun_direction: new_sun_direction,
            magnetic_field: new_magnetic_field,
        }
    }
}

struct SunObservationPrediction;

impl ObservationModel for SunObservationPrediction {
    type State = DirectionState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Observation = SunObservation;
    type Time = f64;
    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        _gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        SunObservation {
            sun_direction: state.sun_direction.clone(),
        }
    }
}

#[test]
fn test_macro() {
    let state = DirectionState {
        sun_direction: Direction::default(),
        magnetic_field: Vector3::zeros(),
    };
    let time = 0.0;
    let dt = 0.1;
    let covariance_matrix = SMatrix::identity();
    let sigma = DirectionStateSigmaPoint {
        sun_direction: Vector2::zeros(),
        magnetic_field: Vector3::zeros(),
    };
    let v: SVector<f64, 5> = sigma.into();
    let _state_sg: DirectionStateSigmaPoint = v.into();

    let u = AttitudeInputGaussian {
        omega_measured: Vector3::zeros(),
        omega_measured_covariance: SMatrix::identity(),
    };
    let input_sg_set = u.to_sigma().unwrap();
    let input_sg = input_sg_set.positive_delta[0];
    let model = DirectionStateModel;
    let u_nominal = u.mean().algebraize().0;
    model.propagate(
        &state,
        &EmptyInput,
        &u_nominal.merge_sigma(&input_sg),
        &time,
        &dt,
    );
    let observation_model = SunObservationPrediction;
    let sun_direction = observation_model.predict(&state, &EmptyInput, &EmptyInput, &time);
    let mut ukf = UnscentedKalmanFilter::new(
        model,
        state,
        covariance_matrix,
        &0.0,
        UKFParameters::default(),
    );
    ukf.propagate(&EmptyInput, &u, None, &time).unwrap();
    ukf.update(
        &observation_model,
        &sun_direction,
        &EmptyInput,
        &EmptyInput,
        &0.0,
        SMatrix::identity(),
    )
    .unwrap();
}

// グループ機能のテスト
#[derive(EstimationGaussianInput, Clone)]
pub struct GroupedInput {
    #[group("angular_velocity")]
    omega_x: f64,
    #[group("angular_velocity")]
    omega_y: f64,
    #[group("angular_velocity")]
    omega_z: f64,
    // グループなしのフィールド
    temperature: f64,
}

#[test]
fn test_grouped_input() {
    // angular_velocity グループの共分散行列 (3x3)
    let angular_velocity_cov = SMatrix::<f64, 3, 3>::identity() * 0.1;
    // temperature の共分散 (1x1)
    let temperature_cov = SMatrix::<f64, 1, 1>::identity() * 0.01;

    let grouped_input = GroupedInputGaussian {
        angular_velocity_covariance: angular_velocity_cov,
        temperature_covariance: temperature_cov,
        omega_x: 0.1,
        omega_y: 0.2,
        omega_z: 0.3,
        temperature: 25.0,
    };

    // シグマ点生成のテスト
    let _sigma_points = grouped_input.to_sigma().unwrap();

    // 平均値の確認
    let mean = grouped_input.mean();
    assert!((mean.omega_x - 0.1).abs() < 1e-10);
    assert!((mean.omega_y - 0.2).abs() < 1e-10);
    assert!((mean.omega_z - 0.3).abs() < 1e-10);
    assert!((mean.temperature - 25.0).abs() < 1e-10);
}
