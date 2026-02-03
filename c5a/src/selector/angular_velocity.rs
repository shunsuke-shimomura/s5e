use nalgebra::Vector3;
use z_filter::{Filter, ZFilter1stOrder};

use crate::{
    constants::{DELTA_T, GYRO_CUTOFF_HZ},
    data,
};

pub struct AngularVelocitySelector<const N: usize> {
    gyro_filter_list: [ZFilter1stOrder<f64>; 3],
}

impl<const N: usize> Default for AngularVelocitySelector<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> AngularVelocitySelector<N> {
    pub fn new() -> Self {
        let gyro_filter = z_filter::bilinear_transform(
            z_filter::Poly1::new([2.0 * core::f64::consts::PI * GYRO_CUTOFF_HZ, 0.0]), // 0 + wc
            z_filter::Poly1::new([2.0 * core::f64::consts::PI * GYRO_CUTOFF_HZ, 1.0]), // s + wc
            DELTA_T,
        )
        .ok()
        .unwrap();
        let gyro_filter_list = [
            gyro_filter.clone(),
            gyro_filter.clone(),
            gyro_filter.clone(),
        ];
        Self { gyro_filter_list }
    }

    pub fn main_loop(
        &mut self,
        input: [Option<data::AngularVelocityData>; N],
    ) -> Option<data::AngularVelocityData> {
        input
            .iter()
            .filter_map(|data| data.as_ref())
            .min_by(|a, b| {
                a.angular_velocity_variance
                    .determinant()
                    .partial_cmp(&b.angular_velocity_variance.determinant())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|data| {
                let filtered_angular_velocity = Vector3::new(
                    self.gyro_filter_list[0].process_sample(data.angular_velocity.x),
                    self.gyro_filter_list[1].process_sample(data.angular_velocity.y),
                    self.gyro_filter_list[2].process_sample(data.angular_velocity.z),
                );
                data::AngularVelocityData {
                    angular_velocity: filtered_angular_velocity,
                    angular_velocity_variance: data.angular_velocity_variance,
                }
            })
    }
}
