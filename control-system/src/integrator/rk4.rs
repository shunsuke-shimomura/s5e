use core::ops::{Add, Div, Mul};

use crate::integrator::{Prediction, TimeIntegrator};

#[derive(Debug)]
pub struct RK4Solver<V, T, DT>
where
    V: Add<Output = V> + Mul<f64, Output = V> + Div<f64, Output = V> + Clone,
{
    now: V,
    after_halfdt: Option<V>,
    after_dt: Option<V>,
    dt: Option<DT>,
    time: Option<T>,
}

impl<V, T, DT> TimeIntegrator<V> for RK4Solver<V, T, DT>
where
    V: Add<Output = V> + Mul<f64, Output = V> + Div<f64, Output = V> + Mul<DT, Output = V> + Clone,
    T: Clone,
    DT: Clone,
{
    fn new(now: V) -> Self {
        RK4Solver {
            now,
            after_halfdt: None,
            after_dt: None,
            dt: None,
            time: None,
        }
    }

    // let f = |phase, time, y| { function_of_y_dot() };
    fn propagate<F>(&mut self, f: F, dt: Self::Interval, time: Self::Time)
    where
        F: FnOnce(RK4Phase, Self::Time, V) -> V + Clone,
    {
        let k1 = f.clone()(RK4Phase::Now, time.clone(), self.now.clone());
        let k2 = f.clone()(
            RK4Phase::HalfDt,
            time.clone(),
            self.now.clone() + k1.clone() * 0.5 * dt.clone(),
        );

        self.after_halfdt = Some(self.now.clone() + (k1.clone() + k2.clone()) * dt.clone() / 2.0);

        let k3 = f.clone()(
            RK4Phase::HalfDt,
            time.clone(),
            self.now.clone() + k2.clone() * 0.5 * dt.clone(),
        );
        let k4 = f.clone()(
            RK4Phase::Dt,
            time.clone(),
            self.now.clone() + k3.clone() * dt.clone(),
        );
        self.after_dt = Some(self.now.clone() + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * dt.clone() / 6.0);
        self.dt = Some(dt);
        self.time = Some(time.clone());
    }

    fn clear(&mut self) {
        self.now = self.after_dt.take().unwrap();
        self.after_halfdt = None;
    }
}

impl<V, T, DT> Prediction<V> for RK4Solver<V, T, DT>
where
    V: Clone + Add<Output = V> + Mul<f64, Output = V> + Div<f64, Output = V>,
    T: Clone,
    DT: Clone,
{
    type Phase = RK4Phase;
    type Time = T;
    type Interval = DT;

    fn get_now(&self) -> V {
        self.now.clone()
    }

    fn get(&self, phase: RK4Phase) -> Option<V> {
        match phase {
            RK4Phase::Now => Some(self.now.clone()),
            RK4Phase::HalfDt => self.after_halfdt.clone(),
            RK4Phase::Dt => self.after_dt.clone(),
        }
    }

    fn dt(&self) -> Option<Self::Interval> {
        self.dt.clone()
    }

    fn time(&self) -> Option<Self::Time> {
        self.time.clone()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RK4Phase {
    Now,
    HalfDt,
    Dt,
}

#[derive(Debug)]
pub struct RK4Input<V: Clone, T: Clone, DT: Clone> {
    now: V,
    after_halfdt: Option<V>,
    after_dt: Option<V>,
    dt: Option<DT>,
    time: Option<T>,
}

impl<V: Clone, T: Clone, DT: Clone> RK4Input<V, T, DT> {
    pub fn new(now: V) -> Self {
        RK4Input {
            now,
            after_halfdt: None,
            after_dt: None,
            dt: None,
            time: None,
        }
    }

    pub fn set(&mut self, prediction: RK4InputPrediction<V, T, DT>) {
        self.after_halfdt = Some(prediction.after_halfdt);
        self.after_dt = Some(prediction.after_dt);
        self.dt = Some(prediction.dt);
        self.time = Some(prediction.time);
    }

    pub fn clear(&mut self) {
        self.now = self.after_dt.take().unwrap();
        self.after_halfdt = None;
    }
}

impl<V, T, DT> Prediction<V> for RK4Input<V, T, DT>
where
    V: Clone,
    T: Clone,
    DT: Clone,
{
    type Phase = RK4Phase;
    type Interval = DT;
    type Time = T;

    fn get_now(&self) -> V {
        self.now.clone()
    }

    fn get(&self, phase: RK4Phase) -> Option<V> {
        match phase {
            RK4Phase::Now => Some(self.now.clone()),
            RK4Phase::HalfDt => self.after_halfdt.clone(),
            RK4Phase::Dt => self.after_dt.clone(),
        }
    }

    fn dt(&self) -> Option<Self::Interval> {
        self.dt.clone()
    }

    fn time(&self) -> Option<Self::Time> {
        self.time.clone()
    }
}

#[derive(Debug)]
pub struct RK4InputPrediction<V: Clone, T: Clone, DT: Clone> {
    pub after_halfdt: V,
    pub after_dt: V,
    pub dt: DT,
    pub time: T,
}
