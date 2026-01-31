use crate::data;

pub struct SunDirectionSelector<const N: usize> {}

impl<const N: usize> Default for SunDirectionSelector<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> SunDirectionSelector<N> {
    pub fn new() -> Self {
        Self {}
    }

    pub fn main_loop(
        &mut self,
        input: [Option<data::SunSensorData>; N],
    ) -> Option<data::SunSensorData> {
        input
            .iter()
            .filter_map(|data| data.as_ref())
            .min_by(|a, b| {
                a.std
                    .partial_cmp(&b.std)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}
