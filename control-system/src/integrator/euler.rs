use core::ops::{Add, Div, Mul};

use crate::integrator::{Prediction, TimeIntegrator};

#[derive(Debug)]
pub struct EulerSolver<V, T, DT>
where
    V: Add<Output = V> + Mul<f64, Output = V> + Div<f64, Output = V> + Clone,
    T: Clone,
    DT: Clone,
{
    now: V,
    after_dt: Option<V>,
    dt: Option<DT>,
    time: Option<T>,
}

impl<V, T, DT> TimeIntegrator<V> for EulerSolver<V, T, DT>
where
    V: std::fmt::Debug
        + Add<Output = V>
        + Mul<f64, Output = V>
        + Div<f64, Output = V>
        + Mul<DT, Output = V>
        + Clone,
    T: Clone,
    DT: Clone + std::fmt::Debug,
{
    fn new(now: V) -> Self {
        EulerSolver {
            now,
            after_dt: None,
            dt: None,
            time: None,
        }
    }

    // let f = |phase, time, y| { function_of_y_dot() };
    fn propagate<F>(&mut self, f: F, dt: Self::Interval, time: Self::Time)
    where
        F: FnOnce(EulerPhase, Self::Time, V) -> V,
    {
        let dot = f(EulerPhase::Now, time.clone(), self.now.clone());
        self.after_dt = Some(self.now.clone() + dot * dt.clone());
        self.dt = Some(dt);
        self.time = Some(time.clone());
    }

    fn clear(&mut self) {
        self.now = self.after_dt.take().unwrap();
        self.dt = None;
    }
}

impl<V, T, DT> Prediction<V> for EulerSolver<V, T, DT>
where
    V: Add<Output = V> + Mul<f64, Output = V> + Div<f64, Output = V> + Clone,
    T: Clone,
    DT: Clone,
{
    type Phase = EulerPhase;
    type Time = T;
    type Interval = DT;

    fn get_now(&self) -> V {
        self.now.clone()
    }

    fn get(&self, phase: EulerPhase) -> Option<V> {
        match phase {
            EulerPhase::Now => Some(self.now.clone()),
            EulerPhase::Dt => self.after_dt.clone(),
        }
    }

    fn dt(&self) -> Option<Self::Interval> {
        self.dt.clone()
    }

    fn time(&self) -> Option<Self::Time> {
        self.time.clone()
    }
}

impl<V, T, DT> From<EulerInput<V, T, DT>> for EulerSolver<V, T, DT>
where
    V: Clone + Add<Output = V> + Mul<f64, Output = V> + Div<f64, Output = V>,
    T: Clone,
    DT: Clone,
{
    fn from(euler_input: EulerInput<V, T, DT>) -> Self {
        EulerSolver {
            now: euler_input.now,
            after_dt: euler_input.after_dt,
            dt: euler_input.dt,
            time: euler_input.time,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EulerPhase {
    Now,
    Dt,
}

#[derive(Debug)]
pub struct EulerInput<V: Clone, T: Clone, DT: Clone> {
    now: V,
    after_dt: Option<V>,
    dt: Option<DT>,
    time: Option<T>,
}

impl<V: Clone, T: Clone, DT: Clone> EulerInput<V, T, DT> {
    pub fn new(now: V) -> Self {
        EulerInput {
            now,
            after_dt: None,
            dt: None,
            time: None,
        }
    }

    pub fn set(&mut self, prediction: EulerInputPrediction<V, T, DT>) {
        self.after_dt = Some(prediction.after_dt);
        self.dt = Some(prediction.dt);
        self.time = Some(prediction.time);
    }

    pub fn clear(&mut self) {
        self.now = self.after_dt.take().unwrap();
        self.dt = None;
    }
}

impl<V: Clone, T: Clone, DT: Clone> Prediction<V> for EulerInput<V, T, DT>
where
    V: Clone,
{
    type Phase = EulerPhase;
    type Time = T;
    type Interval = DT;

    fn get_now(&self) -> V {
        self.now.clone()
    }

    fn get(&self, phase: EulerPhase) -> Option<V> {
        match phase {
            EulerPhase::Now => Some(self.now.clone()),
            EulerPhase::Dt => self.after_dt.clone(),
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
pub struct EulerInputPrediction<V: Clone, T: Clone, DT: Clone> {
    pub after_dt: V,
    pub time: T,
    pub dt: DT,
}
