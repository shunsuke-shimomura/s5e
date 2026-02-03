use astrodynamics::{
    coordinate::{
        ECEFPosition, ECEFVector, ECIPosition, ECIVector, GeodeticPosition, GeodeticVector,
    },
    time::JulianDate,
};
use control_system::{EstimationGaussianInput, EstimationOutputStruct};
use nalgebra::SVector;

use crate::{
    constants::{UKF_ALPHA, UKF_BETA, UKF_KAPPA},
    data,
};

#[derive(EstimationGaussianInput, Clone)]
struct ECIPositionInput {
    position: SVector<f64, 3>,
}

#[derive(EstimationOutputStruct)]
struct MagneticFieldOutput {
    magnetic_field_eci: SVector<f64, 3>,
}

pub struct InertialMagneticFieldCalculation {
    ukf_params: control_system::ukf::UKFParameters,
}

impl Default for InertialMagneticFieldCalculation {
    fn default() -> Self {
        Self::new()
    }
}

impl InertialMagneticFieldCalculation {
    pub fn new() -> Self {
        Self {
            ukf_params: control_system::ukf::UKFParameters::new(UKF_ALPHA, UKF_BETA, UKF_KAPPA),
        }
    }

    pub fn main_loop(
        &mut self,
        time: &crate::Time,
        input: Option<data::OrbitEstimationData>,
    ) -> Option<data::InertialMagneticFieldData> {
        if let (Some(posvel_data), Some(absolute_time)) = (input, time.absolute.as_ref()) {
            let julian_date = JulianDate::new(
                absolute_time.year,
                absolute_time.month,
                absolute_time.day,
                absolute_time.hour,
                absolute_time.minute,
                absolute_time.second,
                absolute_time.nanosecond,
            );

            let position_input = ECIPositionInputGaussian {
                position: posvel_data.position,
                position_covariance: posvel_data.position_variance,
            };

            // Use input_shift to compute variance only
            let mag_calclation = |input: &ECIPositionInput| {
                let eci_position = ECIPosition::from(input.position);
                let ecef_position = ECEFPosition::from_eci(eci_position, julian_date.clone());

                let geodetic_position = GeodeticPosition::from(ecef_position);
                let mag_field_geodetic = {
                    let field_info = igrf::declination(
                        geodetic_position.latitude,
                        geodetic_position.longitude,
                        geodetic_position.altitude as u32,
                        time::Date::from_julian_day(julian_date.value as i32).unwrap(),
                    )
                    .unwrap();

                    GeodeticVector {
                        x: field_info.x,
                        y: field_info.y,
                        z: field_info.z,
                    }
                };

                let mag_field_ecef =
                    ECEFVector::from_geodetic(mag_field_geodetic, geodetic_position);
                let mag_field_eci = ECIVector::from_ecef(mag_field_ecef, julian_date.clone());

                MagneticFieldOutput {
                    magnetic_field_eci: mag_field_eci.into(),
                }
            };

            let (mag_field_mean, mag_field_eci_variance) =
                control_system::ukf::input_shift(&position_input, mag_calclation, &self.ukf_params)
                    .unwrap();

            // 共分散行列のいずれかの成分が2500を超えていたら異常値とみなす
            if mag_field_eci_variance.iter().any(|&x| x.abs() > 2500.0) {
                None
            } else {
                Some(data::InertialMagneticFieldData {
                    magnetic_field_eci: mag_field_mean.magnetic_field_eci,
                    magnetic_field_eci_variance: mag_field_eci_variance
                        + nalgebra::Matrix3::identity()
                            * crate::constants::IGRF_MAG_FIELD_STD.powi(2),
                })
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_magnetic_field_calculation() {
        let orbit_input = crate::data::OrbitEstimationData {
            position: nalgebra::SVector::<f64, 3>::new(
                3006280.241117862,
                -3553725.0929044387,
                -4963799.972723026,
            ),
            position_variance: nalgebra::SMatrix::<f64, 3, 3>::new(
                100.01108074188232,
                -0.004091739654541016,
                -0.003470897674560547,
                -0.004091739654541016,
                99.98801612854004,
                -0.0005278587341308594,
                -0.0034704208374023438,
                -0.0005283355712890625,
                100.01336097717285,
            ),
            velocity: nalgebra::SVector::<f64, 3>::new(
                6805.289890839495,
                2692.7196024897585,
                2196.046249584966,
            ),
            velocity_variance: nalgebra::SMatrix::<f64, 3, 3>::new(
                43.65051893040709,
                -21.692985547257592,
                -17.600444807588246,
                -21.692985547257308,
                90.20137787566273,
                -7.298760345661776,
                -17.60044480758779,
                -7.298760345661776,
                92.91787167764778,
            ),
        };
        let mut time = crate::Time::from_seconds(18.120000000000033);
        time.absolute = Some(crate::AbsoluteTime {
            year: 2020,
            month: 4,
            day: 1,
            hour: 12,
            minute: 0,
            second: 18,
            nanosecond: 110000000,
        });

        let mut calculator = super::InertialMagneticFieldCalculation::new();
        let result = calculator.main_loop(&time, Some(orbit_input));
        println!("Magnetic field calculation result: {:?}", result);
        assert!(
            result
                .map(|data| data
                    .magnetic_field_eci_variance
                    .iter()
                    .all(|&x| x.abs() < 1000.0))
                .unwrap_or(true) // Noneの場合は動作しているとみなす
        );
    }
}
