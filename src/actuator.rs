use std::{cell::RefCell, fmt::Debug, rc::Rc};

use chrono::NaiveDateTime;
use debug_s5e::debug_actuator;
use nalgebra::{Matrix3, Vector3};
use rand_distr::{Distribution, Normal};

use control_system::integrator::{
    Prediction, TimeIntegrator,
    rk4::{RK4Input, RK4InputPrediction, RK4Phase, RK4Solver},
};

use astrodynamics::coordinate::BodyVector;

use crate::sensor::Sensor;

pub trait Actuator {
    type PORTIN: Clone + Debug;
    type TICKIN: Clone;
    type OUT: Clone;
    type TIME;

    fn actuator_tick(&mut self, input: Self::TICKIN, time: Self::TIME, dt: f64);
    fn output(&self, phase: RK4Phase) -> Self::OUT;
    fn actuator_port(&mut self) -> &mut s5e_port::S5ESubscribePort<Self::PORTIN>;
    fn actuator_clear(&mut self);
}

pub struct Magnetorquer {
    magnetic_moment: RK4Input<BodyVector, NaiveDateTime, f64>,
    moment_noise: Normal<f64>,
    rng: Rc<RefCell<rand::prelude::ThreadRng>>,

    magnetorquer_ctrl: s5e_port::S5ESubscribePort<s5e_port::MagnetorquerCtrlEvent>,
}

impl Magnetorquer {
    pub fn new(rng: Rc<RefCell<rand::prelude::ThreadRng>>, noise_std: f64) -> Self {
        Magnetorquer {
            magnetic_moment: RK4Input::new(BodyVector::new()),
            moment_noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            magnetorquer_ctrl: s5e_port::S5ESubscribePort::new(),
        }
    }
}

impl Actuator for Magnetorquer {
    type PORTIN = s5e_port::MagnetorquerCtrlEvent;
    type TICKIN = ();
    type OUT = BodyVector;
    type TIME = chrono::NaiveDateTime;

    fn output(&self, phase: RK4Phase) -> Self::OUT {
        let mut rng_ref = self.rng.borrow_mut();
        let moment = match self.magnetic_moment.get(phase) {
            Some(m) => m,
            None => self.magnetic_moment.get_now(),
        };
        BodyVector::from(
            Vector3::from(moment)
                + Vector3::new(
                    self.moment_noise.sample(&mut *rng_ref),
                    self.moment_noise.sample(&mut *rng_ref),
                    self.moment_noise.sample(&mut *rng_ref),
                ),
        )
    }

    fn actuator_tick(&mut self, _input: Self::TICKIN, time: Self::TIME, dt: f64) {
        let moment = if let Some(event) = self.magnetorquer_ctrl.subscribe() {
            let m = BodyVector::from(event.magnetic_moment);
            debug_actuator!(
                "Magnetorquer: received new target moment: x={:.6}, y={:.6}, z={:.6} Am²",
                m.x,
                m.y,
                m.z
            );
            m
        } else {
            self.magnetic_moment.get_now()
        };

        let halfdt = BodyVector::from(
            (Vector3::from(self.magnetic_moment.get_now()) + Vector3::from(moment.clone())) * 0.5,
        );

        debug_actuator!(
            "Magnetorquer: current moment: x={:.6}, y={:.6}, z={:.6} Am²",
            moment.x,
            moment.y,
            moment.z
        );

        self.magnetic_moment.set(RK4InputPrediction {
            after_halfdt: halfdt,
            after_dt: moment,
            dt,
            time,
        });
    }

    fn actuator_port(&mut self) -> &mut s5e_port::S5ESubscribePort<Self::PORTIN> {
        &mut self.magnetorquer_ctrl
    }

    fn actuator_clear(&mut self) {
        self.magnetic_moment.clear();
    }
}

pub struct IdealTorquer {
    torque: RK4Input<Vector3<f64>, NaiveDateTime, f64>,
    torque_noise: Normal<f64>,
    rng: Rc<RefCell<rand::prelude::ThreadRng>>,

    ideal_torquer_ctrl: s5e_port::S5ESubscribePort<s5e_port::IdealTorquerCtrlEvent>,
}

impl IdealTorquer {
    pub fn new(rng: Rc<RefCell<rand::prelude::ThreadRng>>, noise_std: f64) -> Self {
        IdealTorquer {
            torque: RK4Input::new(Vector3::new(0.0, 0.0, 0.0)),
            torque_noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            ideal_torquer_ctrl: s5e_port::S5ESubscribePort::new(),
        }
    }
}
impl Actuator for IdealTorquer {
    type PORTIN = s5e_port::IdealTorquerCtrlEvent;
    type TICKIN = ();
    type OUT = BodyVector;
    type TIME = chrono::NaiveDateTime;

    fn output(&self, phase: RK4Phase) -> Self::OUT {
        let mut rng_ref = self.rng.borrow_mut();
        let torque = match self.torque.get(phase) {
            Some(t) => t,
            None => self.torque.get_now(),
        };
        BodyVector::from(
            Vector3::from(torque)
                + Vector3::new(
                    self.torque_noise.sample(&mut *rng_ref),
                    self.torque_noise.sample(&mut *rng_ref),
                    self.torque_noise.sample(&mut *rng_ref),
                ),
        )
    }

    fn actuator_tick(&mut self, _input: Self::TICKIN, time: Self::TIME, dt: f64) {
        let torque = if let Some(event) = self.ideal_torquer_ctrl.subscribe() {
            debug_actuator!(
                "ReactionWheel: received new target torque: x={:.6}, y={:.6}, z={:.6} Nm",
                event.torque.x,
                event.torque.y,
                event.torque.z
            );
            event.torque
        } else {
            self.torque.get_now()
        };

        let halfdt = (self.torque.get_now() + torque) * 0.5;

        debug_actuator!(
            "ReactionWheel: current torque: x={:.6}, y={:.6}, z={:.6} Nm",
            torque.x,
            torque.y,
            torque.z
        );

        self.torque.set(RK4InputPrediction {
            after_halfdt: halfdt,
            after_dt: torque,
            dt,
            time,
        });
    }
    fn actuator_port(&mut self) -> &mut s5e_port::S5ESubscribePort<Self::PORTIN> {
        &mut self.ideal_torquer_ctrl
    }

    fn actuator_clear(&mut self) {
        self.torque.clear();
    }
}

pub struct ReactionWheel {
    angular_velocity: RK4Solver<Vector3<f64>, NaiveDateTime, f64>,
    torque: RK4Input<Vector3<f64>, NaiveDateTime, f64>,
    angular_acceleration: RK4Input<Vector3<f64>, NaiveDateTime, f64>,

    inertia: Matrix3<f64>,
    torque_noise: Normal<f64>,
    rng: Rc<RefCell<rand::prelude::ThreadRng>>,

    reaction_wheel_snapshot: s5e_port::S5EPublishPort<s5e_port::ReactionWheelRotationData>,
    reaction_wheel_ctrl: s5e_port::S5ESubscribePort<s5e_port::ReactionWheelCtrlEvent>,
}

impl ReactionWheel {
    pub fn new(
        inertia: Matrix3<f64>,
        noise_std: f64,
        rng: Rc<RefCell<rand::prelude::ThreadRng>>,
    ) -> Self {
        ReactionWheel {
            angular_velocity: RK4Solver::new(Vector3::new(0.0, 0.0, 0.0)),
            angular_acceleration: RK4Input::new(Vector3::new(0.0, 0.0, 0.0)),
            torque: RK4Input::new(Vector3::new(0.0, 0.0, 0.0)),
            inertia,
            torque_noise: Normal::new(0.0, noise_std).unwrap(),
            rng,
            reaction_wheel_snapshot: s5e_port::S5EPublishPort::new(),
            reaction_wheel_ctrl: s5e_port::S5ESubscribePort::new(),
        }
    }
}
impl Actuator for ReactionWheel {
    type PORTIN = s5e_port::ReactionWheelCtrlEvent;
    type TICKIN = ();
    type OUT = BodyVector;
    type TIME = chrono::NaiveDateTime;

    fn output(&self, phase: RK4Phase) -> Self::OUT {
        let mut rng_ref = self.rng.borrow_mut();
        let torque = match self.torque.get(phase) {
            Some(t) => t,
            None => self.torque.get_now(),
        };
        BodyVector::from(
            Vector3::from(torque)
                + Vector3::new(
                    self.torque_noise.sample(&mut *rng_ref),
                    self.torque_noise.sample(&mut *rng_ref),
                    self.torque_noise.sample(&mut *rng_ref),
                ),
        )
    }

    fn actuator_tick(&mut self, _input: Self::TICKIN, time: Self::TIME, dt: f64) {
        let angular_acceleration_dt = if let Some(event) = self.reaction_wheel_ctrl.subscribe() {
            debug_actuator!(
                "ReactionWheel: received new target acceleration: x={:.6}, y={:.6}, z={:.6} rad/s²",
                event.angular_acceleration.x,
                event.angular_acceleration.y,
                event.angular_acceleration.z
            );
            event.angular_acceleration
        } else {
            self.angular_acceleration.get_now()
        };

        // calc acc t=t+0.5*dt
        let angular_acc_halfdt =
            (angular_acceleration_dt + self.angular_acceleration.get_now()) * 0.5; // 線形補間

        self.angular_acceleration.set(RK4InputPrediction {
            after_halfdt: angular_acc_halfdt,
            after_dt: angular_acceleration_dt,
            dt,
            time,
        });

        let torque_dt = -self.inertia * angular_acceleration_dt;
        let torque_halfdt = -self.inertia * angular_acc_halfdt;
        debug_actuator!(
            "ReactionWheel: angular velocity: x={:.6}, y={:.6}, z={:.6} rad/s",
            self.angular_velocity.get_now().x,
            self.angular_velocity.get_now().y,
            self.angular_velocity.get_now().z
        );
        debug_actuator!(
            "ReactionWheel: torque: x={:.6}, y={:.6}, z={:.6} Nm",
            torque_dt.x,
            torque_dt.y,
            torque_dt.z
        );

        self.torque.set(RK4InputPrediction {
            after_halfdt: torque_halfdt,
            after_dt: torque_dt,
            dt,
            time,
        });

        let f = |phase, _, _| self.angular_acceleration.get(phase).unwrap();

        self.angular_velocity.propagate(f, dt, time);
    }
    fn actuator_port(&mut self) -> &mut s5e_port::S5ESubscribePort<Self::PORTIN> {
        &mut self.reaction_wheel_ctrl
    }

    fn actuator_clear(&mut self) {
        self.angular_acceleration.clear();
        self.torque.clear();
        self.angular_velocity.clear();
    }
}

impl Sensor for ReactionWheel {
    type IN = ();
    type OUT = s5e_port::ReactionWheelRotationData;

    fn sensor_tick(&mut self, _: Self::IN) {
        self.reaction_wheel_snapshot
            .publish(s5e_port::ReactionWheelRotationData {
                speed_rpm: self.angular_velocity.get_now() * (60.0 / (2.0 * std::f64::consts::PI)),
            });
    }

    fn sensor_port(&self) -> &s5e_port::S5EPublishPort<Self::OUT> {
        &self.reaction_wheel_snapshot
    }

    fn sensor_clear(&mut self) {
        self.reaction_wheel_snapshot.clear();
    }
}
