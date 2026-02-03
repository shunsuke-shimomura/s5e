use crate::data;

// ECI GNSS データをそのまま位置・速度データに分離して流す変換コンポーネント
// 正確にやろうとする場合は姿勢・角速度も必要になるが、ここでは簡易的に GNSS データをそのまま流す
pub struct EciGnssPassThroughConversion {}

impl Default for EciGnssPassThroughConversion {
    fn default() -> Self {
        Self::new()
    }
}

impl EciGnssPassThroughConversion {
    pub fn new() -> Self {
        Self {}
    }

    pub fn main_loop(
        &mut self,
        input: Option<data::ECIGnssData>,
    ) -> Option<data::ECIObservationData> {
        input.map(|data| data::ECIObservationData {
            position: data.component_position,
            position_std: data.position_std,
            velocity: data.component_velocity,
            velocity_std: data.velocity_std,
        })
    }
}
