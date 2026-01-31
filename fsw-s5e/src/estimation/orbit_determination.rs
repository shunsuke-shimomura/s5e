use control_system::{
    EstimationGaussianInput, EstimationOutputStruct, EstimationState,
    ukf::{ObservationModel, PropagationModel, UKFParameters, UnscentedKalmanFilter},
    value_structs::EmptyInput,
};
use nalgebra::{ComplexField, SVector, Vector3, Vector6};

use crate::{
    constants::{
        GME, GNSS_POSITION_NOISE_STD, GNSS_VELOCITY_NOISE_STD, UKF_ALPHA, UKF_BETA, UKF_KAPPA,
    },
    data,
};

/// Kepler: M = E - e sin E (elliptic), Newton
fn kepler_eccentric_anomaly(m: f64, e: f64) -> f64 {
    let mut large_e = if e < 0.8 { m } else { std::f64::consts::PI };
    for _ in 0..30 {
        let f = large_e - e * large_e.sin() - m;
        let fp = 1.0 - e * large_e.cos();
        let d = f / fp;
        large_e -= d;
        if d.abs() < 1e-14 {
            break;
        }
    }
    large_e
}
fn true_anomaly_from_e(large_e: f64, e: f64) -> f64 {
    let s = ((1.0 + e).sqrt()) * (large_e * 0.5).sin();
    let c = ((1.0 - e).sqrt()) * (large_e * 0.5).cos();
    2.0 * s.atan2(c)
}

fn eccentric_anomaly_from_true(nu: f64, e: f64) -> f64 {
    // E = 2 atan( sqrt((1-e)/(1+e)) * tan(nu/2) )
    let s = ((1.0 - e).sqrt()) * (0.5 * nu).sin();
    let c = ((1.0 + e).sqrt()) * (0.5 * nu).cos();
    2.0 * s.atan2(c)
}

/// true anomaly -> hyperbolic anomaly H
fn hyperbolic_anomaly_from_true(nu: f64, e: f64) -> f64 {
    // tanh(H/2) = sqrt((e-1)/(e+1)) * tan(nu/2)
    let factor = ((e - 1.0) / (e + 1.0)).sqrt();
    let tan_half_nu = (0.5 * nu).tan();
    let tanh_half_h = factor * tan_half_nu;

    // NOTE: for physically valid hyperbola motion, |tanh_half_h| < 1.
    // If numerical noise pushes it slightly outside, clamp to avoid NaN.
    let tanh_half_h = tanh_half_h.clamp(-0.999_999_999_999, 0.999_999_999_999);

    2.0 * tanh_half_h.atanh()
}

/// hyperbolic anomaly H -> true anomaly
fn true_from_hyperbolic_anomaly(h: f64, e: f64) -> f64 {
    // tan(nu/2) = sqrt((e+1)/(e-1)) * tanh(H/2)
    let factor = ((e + 1.0) / (e - 1.0)).sqrt();
    let tanh_half_h = (0.5 * h).tanh();
    let tan_half_nu = factor * tanh_half_h;
    2.0 * tan_half_nu.atan()
}

/// Solve e*sinh(H) - H = Mh for H (Newton)
fn kepler_hyperbolic_anomaly(mh: f64, e: f64) -> f64 {
    // Good all-round initial guess:
    // asinh(mh/e) behaves well for large |mh|
    let mut h = (mh / e).asinh();

    let tol = 1e-13;
    let max_iter = 30;

    for _ in 0..max_iter {
        let sh = h.sinh();
        let ch = h.cosh();
        let f = e * sh - h - mh;
        let fp = e * ch - 1.0;
        let dh = f / fp;
        h -= dh;
        if dh.abs() < tol {
            break;
        }
    }
    h
}

#[derive(Debug, Clone)]
struct EquinoctialOrbit {
    pub p: f64, // semi-latus rectum
    pub f: f64, // e*cos(omega + Omega)
    pub g: f64, // e*sin(omega + Omega)
    pub h: f64, // tan(i/2)*cos(Omega)
    pub k: f64, // tan(i/2)*sin(Omega)
    pub l: f64, // true longitude
}

impl EquinoctialOrbit {
    fn propagate_kepler_mee(&self, dt: f64) -> Self {
        let (p, f, g) = (self.p, self.f, self.g);

        let e = (f * f + g * g).try_sqrt().unwrap();
        // Using m units (GME is in m³/s²)
        let mu = GME;

        // psi = Ω + ω
        let psi = g.atan2(f);

        // nu = L - (Ω+ω)
        let nu = (self.l - psi).rem_euclid(std::f64::consts::TAU);

        let propagated_l = if e < 1.0 {
            // ----- Elliptic (e<1) -----
            let a = p / (1.0 - e * e);
            let n = (mu / (a * a * a)).try_sqrt().unwrap(); // rad/s
            // nu -> E -> M
            let large_e = eccentric_anomaly_from_true(nu, e);
            let large_m = (large_e - e * large_e.sin()).rem_euclid(std::f64::consts::TAU);

            // propagate mean anomaly
            let propagated_large_m = (large_m + n * dt).rem_euclid(std::f64::consts::TAU);

            // M -> E -> nu
            let propagated_large_e = kepler_eccentric_anomaly(propagated_large_m, e);
            let propagated_nu = true_anomaly_from_e(propagated_large_e, e);

            (psi + propagated_nu).rem_euclid(std::f64::consts::TAU)
        } else {
            // ----- Hyperbolic (e>1) -----
            // a is negative: a = p / (1 - e^2)
            let a = p / (1.0 - e * e); // a<0
            let n = (mu / ((-a) * (-a) * (-a))).sqrt(); // "hyperbolic mean motion" rad/s (positive)

            // nu -> H -> Mh
            let h0 = hyperbolic_anomaly_from_true(nu, e);
            let mh0 = e * h0.sinh() - h0; // unwrapped (not modulo 2π)

            // propagate hyperbolic mean anomaly
            let mh1 = mh0 + n * dt;

            // Mh -> H -> nu
            let h1 = kepler_hyperbolic_anomaly(mh1, e);
            let nu1 = true_from_hyperbolic_anomaly(h1, e);

            (psi + nu1).rem_euclid(std::f64::consts::TAU)
        };

        EquinoctialOrbit {
            p,
            f,
            g,
            h: self.h,
            k: self.k,
            l: propagated_l,
        }
    }
}

#[derive(Debug, Clone)]
struct ECIPositionVelocity {
    pub position: SVector<f64, 3>,
    pub velocity: SVector<f64, 3>,
}

impl From<EquinoctialOrbit> for ECIPositionVelocity {
    fn from(orb: EquinoctialOrbit) -> Self {
        let (p, f, g, h, k, l) = (orb.p, orb.f, orb.g, orb.h, orb.k, orb.l);

        let (s_l, c_l) = (l.sin(), l.cos());
        let w = 1.0 + f * c_l + g * s_l;

        let r = p / w;
        let smp = (GME / p).sqrt();

        let kk = k * k;
        let hh = h * h;
        let s2 = 1.0 + hh + kk;
        let tkh = 2.0 * k * h;

        let fhat = SVector::<f64, 3>::new(1.0 - kk + hh, tkh, -2.0 * k) / s2;
        let ghat = SVector::<f64, 3>::new(tkh, 1.0 + kk - hh, 2.0 * h) / s2;

        let x = r * c_l;
        let y = r * s_l;
        let xdot = -smp * (g + s_l);
        let ydot = smp * (f + c_l);

        let position = x * fhat + y * ghat;
        let velocity = xdot * fhat + ydot * ghat;

        ECIPositionVelocity { position, velocity }
    }
}

impl From<ECIPositionVelocity> for EquinoctialOrbit {
    fn from(eci: ECIPositionVelocity) -> Self {
        let r = eci.position;
        let v = eci.velocity;
        let rmag = r.norm();
        let rdv = r.dot(&v);

        let rhat = r / rmag;
        let hvec = r.cross(&v);
        let hmag = hvec.norm();
        let hhat = hvec / hmag;

        // Using m units (GME is in m³/s²)
        let p = hmag * hmag / GME;

        // posigrade: k = hhat_x/(1+hhat_z), h = -hhat_y/(1+hhat_z)
        let denom = 1.0 + hhat[2];
        let k = hhat[0] / denom;
        let h = -hhat[1] / denom;

        // vhat = (rmag*v - (r·v)*rhat)/hmag
        let vhat = (rmag * v - rdv * rhat) / hmag;

        // ecc = (v×h)/mu - rhat
        let ecc = v.cross(&hvec) / GME - rhat;

        // fhat, ghat (equinoctial frame basis)
        let kk = k * k;
        let hh = h * h;
        let s2 = 1.0 + hh + kk;
        let tkh = 2.0 * k * h;

        let fhat = SVector::<f64, 3>::new(1.0 - kk + hh, tkh, -2.0 * k) / s2;
        let ghat = SVector::<f64, 3>::new(tkh, 1.0 + kk - hh, 2.0 * h) / s2;

        // f,g are projections of ecc onto fhat/ghat
        let f = ecc.dot(&fhat);
        let g = ecc.dot(&ghat);

        // true longitude L
        let l = (rhat[1] - vhat[0])
            .atan2(rhat[0] + vhat[1])
            .rem_euclid(2.0 * std::f64::consts::PI);

        EquinoctialOrbit { p, f, g, h, k, l }
    }
}

#[derive(Debug, Clone, EstimationState)]
struct EciPositionVelocityState {
    position: SVector<f64, 3>, // m
    velocity: SVector<f64, 3>, // m/s
}

impl From<EquinoctialOrbit> for EciPositionVelocityState {
    fn from(orb: EquinoctialOrbit) -> Self {
        let eci: ECIPositionVelocity = orb.into();
        EciPositionVelocityState {
            position: eci.position,
            velocity: eci.velocity,
        }
    }
}

impl From<EciPositionVelocityState> for EquinoctialOrbit {
    fn from(state: EciPositionVelocityState) -> Self {
        let eci = ECIPositionVelocity {
            position: state.position,
            velocity: state.velocity,
        };
        eci.into()
    }
}

#[derive(Debug, Clone, EstimationGaussianInput)]
struct EquinoctialOrbitProcessNoise {
    process_noise: SVector<f64, 6>,
}

struct KeplerianPropagationModel;

impl PropagationModel for KeplerianPropagationModel {
    type State = EciPositionVelocityState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EquinoctialOrbitProcessNoise;
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
        let equinoctial: EquinoctialOrbit = state.clone().into();
        let propagated_equinoctial = equinoctial.propagate_kepler_mee(*dt);
        let processed_equinoctial = EquinoctialOrbit {
            p: propagated_equinoctial.p + gaussian_input.process_noise[0],
            f: propagated_equinoctial.f + gaussian_input.process_noise[1],
            g: propagated_equinoctial.g + gaussian_input.process_noise[2],
            h: propagated_equinoctial.h + gaussian_input.process_noise[3],
            k: propagated_equinoctial.k + gaussian_input.process_noise[4],
            l: propagated_equinoctial.l + gaussian_input.process_noise[5],
        };
        processed_equinoctial.into()
    }
}

#[derive(Debug, Clone, EstimationOutputStruct)]
struct ECIGnssObservation {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
}

struct ECIGnssObservationModel;

impl ObservationModel for ECIGnssObservationModel {
    type State = EciPositionVelocityState;
    type DeterministicInput = EmptyInput;
    type GaussianInput = EmptyInput;
    type Time = crate::Time;
    type Observation = ECIGnssObservation;
    fn predict(
        &self,
        state: &Self::State,
        _deterministic_input: &Self::DeterministicInput,
        _gaussian_input: &Self::GaussianInput,
        _time: &Self::Time,
    ) -> Self::Observation {
        ECIGnssObservation {
            position: state.position,
            velocity: state.velocity,
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum OrbitDeterminationMode {
    Initial,
    Keplerian(KeplerianOrbitEstimator),
}

struct KeplerianOrbitEstimator {
    ukf: UnscentedKalmanFilter<
        EciPositionVelocityState,
        crate::Time,
        f64,
        KeplerianPropagationModel,
        EmptyInput,
        EquinoctialOrbitProcessNoiseGaussian,
        6,
        6,
    >,
    process_noise_std: SVector<f64, 6>,
    observation_model: ECIGnssObservationModel,
}

impl KeplerianOrbitEstimator {
    pub fn new(initial_state: EquinoctialOrbit, initial_time: &crate::Time) -> Self {
        let process_noise_std = SVector::<f64, 6>::from_iterator(
            // 適当な値
            [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2]
                .iter()
                .cloned(),
        );
        let propagation = KeplerianPropagationModel;

        let ukf = UnscentedKalmanFilter::new(
            propagation,
            initial_state.into(),
            nalgebra::SMatrix::<f64, 6, 6>::identity() * 1.0e3,
            initial_time,
            UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        );
        KeplerianOrbitEstimator {
            ukf,
            process_noise_std,
            observation_model: ECIGnssObservationModel,
        }
    }
}
pub struct OrbitDetermination {
    mode: OrbitDeterminationMode,
}

impl Default for OrbitDetermination {
    fn default() -> Self {
        Self::new()
    }
}

impl OrbitDetermination {
    pub fn new() -> Self {
        OrbitDetermination {
            mode: OrbitDeterminationMode::Initial,
        }
    }
    pub fn main_loop(
        &mut self,
        time: &crate::Time,
        input: Option<data::ECIObservationData>,
    ) -> Option<data::OrbitEstimationData> {
        match &mut self.mode {
            OrbitDeterminationMode::Initial => {
                if let Some(observation) = input {
                    let initial_state = ECIPositionVelocity {
                        position: observation.position,
                        velocity: observation.velocity,
                    };
                    let estimator = KeplerianOrbitEstimator::new(initial_state.into(), time);
                    self.mode = OrbitDeterminationMode::Keplerian(estimator);
                }
                None
            }
            OrbitDeterminationMode::Keplerian(estimator) => {
                let process_noise_covariance = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(
                    &estimator.process_noise_std.map(|x| x * x),
                );
                estimator
                    .ukf
                    .propagate(
                        &EmptyInput,
                        &EquinoctialOrbitProcessNoiseGaussian {
                            process_noise: Vector6::zeros(),
                            process_noise_covariance,
                        },
                        None,
                        time,
                    )
                    .unwrap();
                if let Some(observation) = input {
                    let position = observation.position;
                    let velocity = observation.velocity;
                    let measurement = ECIGnssObservation { position, velocity };
                    let measurement_noise_covariance = {
                        let mut cov = nalgebra::SMatrix::<f64, 6, 6>::zeros();
                        cov.fixed_view_mut::<3, 3>(0, 0).copy_from(
                            &(nalgebra::SMatrix::<f64, 3, 3>::identity()
                                * GNSS_POSITION_NOISE_STD.powi(2)),
                        );
                        cov.fixed_view_mut::<3, 3>(3, 3).copy_from(
                            &(nalgebra::SMatrix::<f64, 3, 3>::identity()
                                * GNSS_VELOCITY_NOISE_STD.powi(2)),
                        );
                        cov
                    };
                    estimator
                        .ukf
                        .update(
                            &estimator.observation_model,
                            &measurement,
                            &EmptyInput,
                            &EmptyInput,
                            time,
                            measurement_noise_covariance,
                        )
                        .unwrap();
                }
                let state = estimator.ukf.state();
                let covariance = estimator.ukf.covariance();
                Some(data::OrbitEstimationData {
                    position: state.position,
                    position_variance: covariance.fixed_view::<3, 3>(0, 0).clone_owned(),
                    velocity: state.velocity,
                    velocity_variance: covariance.fixed_view::<3, 3>(3, 3).clone_owned(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests_keplerian {
    use rand::distributions::Distribution;
    use sgp4::{
        Elements,
        chrono::{Datelike, NaiveDateTime, Timelike},
    };
    use statrs::distribution::MultivariateNormal;

    impl EquinoctialOrbit {
        fn from_sgp4_with_datetime(elements: &Elements, datetime: &NaiveDateTime) -> Self {
            // propagator 作成
            let propagator = sgp4::Constants::from_elements(elements).unwrap();

            // epochからの経過分（分）
            let minutes = elements.datetime_to_minutes_since_epoch(datetime).unwrap();

            // 伝播（戻り値型は crate の Prediction に依存）
            let pred = propagator.propagate(minutes).unwrap();

            // pred.position / pred.velocity の型に合わせて取り出す
            // SGP4 returns position in km and velocity in km/s, convert to m and m/s
            let r = SVector::<f64, 3>::from_row_slice(&pred.position) * 1000.0;
            let v = SVector::<f64, 3>::from_row_slice(&pred.velocity) * 1000.0;

            EquinoctialOrbit::from(ECIPositionVelocity {
                position: r,
                velocity: v,
            })
        }
    }

    use crate::AbsoluteTime;

    use super::*;

    const TLE_LINE1: &str = "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005";
    const TLE_LINE2: &str = "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647";
    #[test]
    fn test_eci_equinoctial_conversion() {
        let elements = EquinoctialOrbit {
            p: 6786000.0, // meters (was km, now m)
            f: 0.001,
            g: 0.001,
            h: 0.1,
            k: 0.1,
            l: 0.0,
        };
        let eci: ECIPositionVelocity = elements.clone().into();
        let elements_converted: EquinoctialOrbit = eci.into();
        assert!((elements.p - elements_converted.p).abs() < 1.0e-3); // increased tolerance for meters
        assert!((elements.f - elements_converted.f).abs() < 1.0e-6);
        assert!((elements.g - elements_converted.g).abs() < 1.0e-6);
        assert!((elements.h - elements_converted.h).abs() < 1.0e-6);
        assert!((elements.k - elements_converted.k).abs() < 1.0e-6);
        assert!((elements.l - elements_converted.l).abs() < 1.0e-6);
    }

    #[test]
    fn test_tle_to_equinoctial() {
        let elements =
            Elements::from_tle(None, TLE_LINE1.as_bytes(), TLE_LINE2.as_bytes()).unwrap();
        let constants = sgp4::Constants::from_elements(&elements).unwrap();
        let datetime = elements.datetime;
        let propagate_dt = 60.0; // seconds
        let datetime_plus_dt = datetime + sgp4::chrono::Duration::seconds(propagate_dt as i64);

        let equinoctial = EquinoctialOrbit::from_sgp4_with_datetime(&elements, &datetime);
        let propagated_equinoctial = equinoctial.propagate_kepler_mee(propagate_dt);
        let propagated_sgp4 = {
            let minutes = elements
                .datetime_to_minutes_since_epoch(&datetime_plus_dt)
                .unwrap();
            let pred = constants.propagate(minutes).unwrap();
            // SGP4 returns position in km and velocity in km/s, convert to m and m/s
            ECIPositionVelocity {
                position: SVector::<f64, 3>::from_row_slice(&pred.position) * 1000.0,
                velocity: SVector::<f64, 3>::from_row_slice(&pred.velocity) * 1000.0,
            }
        };
        let propagated_equinoctial_eci: ECIPositionVelocity = propagated_equinoctial.into();
        assert!((propagated_equinoctial_eci.position - propagated_sgp4.position).norm() < 1000.0);
        assert!((propagated_equinoctial_eci.velocity - propagated_sgp4.velocity).norm() < 1000.0);
    }

    #[test]
    fn test_estimator_initialization() {
        let estimator = OrbitDetermination::new();
        assert!(matches!(estimator.mode, OrbitDeterminationMode::Initial));
    }

    #[test]
    fn test_keplerian_mode_initialization() {
        let elements =
            Elements::from_tle(None, TLE_LINE1.as_bytes(), TLE_LINE2.as_bytes()).unwrap();
        let datetime = elements.datetime;
        let equinoctial = EquinoctialOrbit::from_sgp4_with_datetime(&elements, &datetime);
        let initial_time = crate::Time::from_seconds(0.0);
        let estimator = KeplerianOrbitEstimator::new(equinoctial.clone(), &initial_time);
        let state = estimator.ukf.state();
        let state_eci = ECIPositionVelocity {
            position: state.position,
            velocity: state.velocity,
        };
        let equinoctial_converted: EquinoctialOrbit = state_eci.into();
        assert!((equinoctial.p - equinoctial_converted.p).abs() < 1.0e-6);
        assert!((equinoctial.f - equinoctial_converted.f).abs() < 1.0e-6);
        assert!((equinoctial.g - equinoctial_converted.g).abs() < 1.0e-6);
        assert!((equinoctial.h - equinoctial_converted.h).abs() < 1.0e-6);
        assert!((equinoctial.k - equinoctial_converted.k).abs() < 1.0e-6);
        assert!((equinoctial.l - equinoctial_converted.l).abs() < 1.0e-6);
    }

    impl From<sgp4::chrono::NaiveDateTime> for AbsoluteTime {
        fn from(datetime: sgp4::chrono::NaiveDateTime) -> Self {
            AbsoluteTime {
                year: datetime.year(),
                month: datetime.month(),
                day: datetime.day(),
                hour: datetime.hour(),
                minute: datetime.minute(),
                second: datetime.second(),
                nanosecond: datetime.nanosecond(),
            }
        }
    }

    #[test]
    fn test_ukf_propagation() {
        let elements =
            Elements::from_tle(None, TLE_LINE1.as_bytes(), TLE_LINE2.as_bytes()).unwrap();
        let datetime = elements.datetime;
        let equinoctial = EquinoctialOrbit::from_sgp4_with_datetime(&elements, &datetime);
        let mut initial_time = crate::Time::from_seconds(0.0);
        initial_time.absolute = Some(AbsoluteTime::from(datetime));
        let mut estimator = KeplerianOrbitEstimator::new(equinoctial.clone(), &initial_time);

        let cov_before = estimator.ukf.covariance().trace();

        // Propagate with zero angular velocity
        let dt = 60.0; // seconds
        let time = initial_time + dt;
        let process_noise_covariance = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(
            &estimator.process_noise_std.map(|x| x * x),
        );

        let mut rng = rand::thread_rng();
        let process_noise_mvn =
            MultivariateNormal::new_from_nalgebra(Vector6::zeros(), process_noise_covariance)
                .unwrap();
        let state = estimator.ukf.state();
        let state_vec = SVector::<f64, 6>::from_iterator(
            state
                .position
                .iter()
                .cloned()
                .chain(state.velocity.iter().cloned()),
        );
        let state_mvn = MultivariateNormal::new_from_nalgebra(
            state_vec,
            estimator.ukf.covariance().clone_owned(),
        )
        .unwrap();
        let mut expected_state_vec_list = Vec::new();
        let iterations = 20;
        for _ in 0..iterations {
            let mut shifted_state_vec_list = Vec::new();
            // Monte Carlo expected value
            for _ in 0..500 {
                let process_noise_sample = process_noise_mvn.sample(&mut rng);
                let state_vec_sample = state_mvn.sample(&mut rng);
                let state_sample = EciPositionVelocityState {
                    position: SVector::<f64, 3>::from_row_slice(&state_vec_sample.as_slice()[0..3]),
                    velocity: SVector::<f64, 3>::from_row_slice(&state_vec_sample.as_slice()[3..6]),
                };
                let equinoctial: EquinoctialOrbit = state_sample.clone().into();
                let propagated_equinoctial = equinoctial.propagate_kepler_mee(dt);
                let processed_equinoctial = EquinoctialOrbit {
                    p: propagated_equinoctial.p + process_noise_sample[0],
                    f: propagated_equinoctial.f + process_noise_sample[1],
                    g: propagated_equinoctial.g + process_noise_sample[2],
                    h: propagated_equinoctial.h + process_noise_sample[3],
                    k: propagated_equinoctial.k + process_noise_sample[4],
                    l: propagated_equinoctial.l + process_noise_sample[5],
                };
                let propagated_state: EciPositionVelocityState = processed_equinoctial.into();
                let propagated_state_vec = SVector::<f64, 6>::from_iterator(
                    propagated_state
                        .position
                        .iter()
                        .cloned()
                        .chain(propagated_state.velocity.iter().cloned()),
                );
                shifted_state_vec_list.push(propagated_state_vec);
            }
            let expected_state_vec = shifted_state_vec_list
                .iter()
                .fold(nalgebra::SVector::<f64, 6>::zeros(), |acc, x| acc + x)
                / (shifted_state_vec_list.len() as f64);
            expected_state_vec_list.push(expected_state_vec);
        }
        let expected_state_vec = expected_state_vec_list
            .iter()
            .fold(nalgebra::SVector::<f64, 6>::zeros(), |acc, x| acc + x)
            / (expected_state_vec_list.len() as f64);
        estimator
            .ukf
            .propagate(
                &EmptyInput,
                &EquinoctialOrbitProcessNoiseGaussian {
                    process_noise: Vector6::zeros(),
                    process_noise_covariance,
                },
                None,
                &time,
            )
            .unwrap();

        let state_after = estimator.ukf.state();
        let state_after_vec = SVector::<f64, 6>::from_iterator(
            state_after
                .position
                .iter()
                .cloned()
                .chain(state_after.velocity.iter().cloned()),
        );
        let cov_after = estimator.ukf.covariance().trace();

        // Covariance should increase due to process noise
        // Tolerance scaled for meters (was 1.0e2 for km, now 1.0e5 for m)
        assert!(
            (state_after_vec - expected_state_vec).norm() < 1.0e5,
            "State expected: {:?}, got: {:?}, error: {}",
            expected_state_vec,
            state_after_vec,
            (state_after_vec - expected_state_vec).norm()
        );
        assert!(
            cov_after >= cov_before,
            "Covariance should increase: {} -> {}",
            cov_before,
            cov_after
        );
    }

    #[test]
    fn test_ukf_update() {
        let elements =
            Elements::from_tle(None, TLE_LINE1.as_bytes(), TLE_LINE2.as_bytes()).unwrap();
        let datetime = elements.datetime;
        let initial_equinoctial = {
            let equinoctial = EquinoctialOrbit::from_sgp4_with_datetime(&elements, &datetime);
            let initial_posvel: ECIPositionVelocity = {
                let mut pos_vel: ECIPositionVelocity = equinoctial.clone().into();
                // Add some initial error
                pos_vel.position += SVector::<f64, 3>::new(100.0, -50.0, 25.0);
                pos_vel.velocity += SVector::<f64, 3>::new(-1.0, 0.5, 0.25);
                pos_vel
            };
            EquinoctialOrbit::from(initial_posvel)
        };

        let mut initial_time = crate::Time::from_seconds(0.0);
        initial_time.absolute = Some(AbsoluteTime::from(datetime));
        let mut estimator =
            KeplerianOrbitEstimator::new(initial_equinoctial.clone(), &initial_time);

        // Apply gnss observation with small noise
        let observation_model = ECIGnssObservationModel;
        let equinoctial = EquinoctialOrbit::from_sgp4_with_datetime(&elements, &datetime);
        let pos_vel_true: ECIPositionVelocity = equinoctial.into();

        let posvel_measured = ECIGnssObservation {
            position: pos_vel_true.position,
            velocity: pos_vel_true.velocity,
        };

        estimator
            .ukf
            .update(
                &observation_model,
                &posvel_measured,
                &EmptyInput,
                &EmptyInput,
                &crate::Time::from_seconds(1200.0),
                {
                    let mut cov = nalgebra::SMatrix::<f64, 6, 6>::zeros();
                    cov.fixed_view_mut::<3, 3>(0, 0)
                        .copy_from(&(nalgebra::SMatrix::<f64, 3, 3>::identity() * 1.0e-5));
                    cov.fixed_view_mut::<3, 3>(3, 3)
                        .copy_from(&(nalgebra::SMatrix::<f64, 3, 3>::identity() * 1.0e-5));
                    cov
                },
            )
            .unwrap();
        let state_after = estimator.ukf.state();
        let final_pos_error = (state_after.position - pos_vel_true.position).norm();
        let final_vel_error = (state_after.velocity - pos_vel_true.velocity).norm();
        assert!(
            final_pos_error < 10.0,
            "Final position error too large: {} m",
            final_pos_error
        );
        assert!(
            final_vel_error < 1.0,
            "Final velocity error too large: {} m/s",
            final_vel_error
        );
    }

    #[test]
    fn test_ukf_converges() {
        let elements =
            Elements::from_tle(None, TLE_LINE1.as_bytes(), TLE_LINE2.as_bytes()).unwrap();
        let constants = sgp4::Constants::from_elements(&elements).unwrap();
        let datetime = elements.datetime;
        let initial_equinoctial = {
            let equinoctial = EquinoctialOrbit::from_sgp4_with_datetime(&elements, &datetime);
            let initial_posvel: ECIPositionVelocity = {
                let mut pos_vel: ECIPositionVelocity = equinoctial.clone().into();
                // Add some initial error
                pos_vel.position += SVector::<f64, 3>::new(100.0, -500.0, 250.0);
                pos_vel.velocity += SVector::<f64, 3>::new(-1.0, 0.5, 0.25);
                pos_vel
            };
            EquinoctialOrbit::from(initial_posvel)
        };

        let mut initial_time = crate::Time::from_seconds(0.0);
        initial_time.absolute = Some(AbsoluteTime::from(datetime));
        let mut estimator =
            KeplerianOrbitEstimator::new(initial_equinoctial.clone(), &initial_time);
        for i in 0..50 {
            // Propagate
            let dt = 60.0; // seconds
            let time = initial_time + dt;
            let process_noise_covariance = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(
                &estimator.process_noise_std.map(|x| x * x),
            );
            estimator
                .ukf
                .propagate(
                    &EmptyInput,
                    &EquinoctialOrbitProcessNoiseGaussian {
                        process_noise: Vector6::zeros(),
                        process_noise_covariance,
                    },
                    None,
                    &time,
                )
                .unwrap();
            initial_time = time;

            // Apply gnss observation with small noise
            let observation_model = ECIGnssObservationModel;
            let pos_vel_true = {
                let datetime_plus_dt =
                    datetime + sgp4::chrono::Duration::seconds((dt * (i + 1) as f64) as i64);
                let minutes = elements
                    .datetime_to_minutes_since_epoch(&datetime_plus_dt)
                    .unwrap();
                let pred = constants.propagate(minutes).unwrap();
                // SGP4 returns position in km and velocity in km/s, convert to m and m/s
                ECIPositionVelocity {
                    position: SVector::<f64, 3>::from_row_slice(&pred.position) * 1000.0,
                    velocity: SVector::<f64, 3>::from_row_slice(&pred.velocity) * 1000.0,
                }
            };

            let posvel_measured = ECIGnssObservation {
                position: pos_vel_true.position,
                velocity: pos_vel_true.velocity,
            };

            estimator
                .ukf
                .update(
                    &observation_model,
                    &posvel_measured,
                    &EmptyInput,
                    &EmptyInput,
                    &initial_time,
                    {
                        let mut cov = nalgebra::SMatrix::<f64, 6, 6>::zeros();
                        cov.fixed_view_mut::<3, 3>(0, 0).copy_from(
                            &(nalgebra::SMatrix::<f64, 3, 3>::identity()
                                * GNSS_POSITION_NOISE_STD.powi(2)),
                        );
                        cov.fixed_view_mut::<3, 3>(3, 3).copy_from(
                            &(nalgebra::SMatrix::<f64, 3, 3>::identity()
                                * GNSS_VELOCITY_NOISE_STD.powi(2)),
                        );
                        cov
                    },
                )
                .unwrap();
        }
        let state_after = estimator.ukf.state();
        let pos_vel_true: ECIPositionVelocity = {
            let datetime_plus_dt = datetime + sgp4::chrono::Duration::seconds((60 * 50) as i64);
            let minutes = elements
                .datetime_to_minutes_since_epoch(&datetime_plus_dt)
                .unwrap();
            let pred = constants.propagate(minutes).unwrap();
            // SGP4 returns position in km and velocity in km/s, convert to m and m/s
            ECIPositionVelocity {
                position: SVector::<f64, 3>::from_row_slice(&pred.position) * 1000.0,
                velocity: SVector::<f64, 3>::from_row_slice(&pred.velocity) * 1000.0,
            }
        };
        let final_pos_error = (state_after.position - pos_vel_true.position).norm();
        let final_vel_error = (state_after.velocity - pos_vel_true.velocity).norm();
        assert!(
            final_pos_error < 10000.0,
            "Final position error too large: {} m",
            final_pos_error
        );
        assert!(
            final_vel_error < 10.0,
            "Final velocity error too large: {} m/s",
            final_vel_error
        );
    }
}
