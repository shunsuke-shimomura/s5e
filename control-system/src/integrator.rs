pub mod euler;
pub mod rk4;

pub trait TimeIntegrator<V>: Prediction<V> {
    fn new(now: V) -> Self;
    fn propagate<F>(&mut self, f: F, dt: Self::Interval, time: Self::Time)
    where
        F: FnOnce(Self::Phase, Self::Time, V) -> V + Clone;
    fn clear(&mut self);
}

pub trait Prediction<V> {
    type Phase;
    type Interval;
    type Time;
    fn get_now(&self) -> V;
    fn get(&self, phase: Self::Phase) -> Option<V>;
    fn dt(&self) -> Option<Self::Interval>;
    fn time(&self) -> Option<Self::Time>;
}
