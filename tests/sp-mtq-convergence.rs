use std::{cell::RefCell, fs, rc::Rc};

use astrodynamics::coordinate::{BodyVector, ECIVector};
use chrono::{NaiveDateTime, TimeDelta};
use control_system::integrator::Prediction;
use csv::Writer;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, UnitVector3, Vector3};
use s5e_lib::{
    sat::{
        VirtualMagFieldSim, VirtualMagFieldSimActuatorInput, VirtualMagFieldSimInterface,
        VirtualMagFieldSimSensorOutput, VirtualMagneticFieldModel,
    },
    SimInputTransfer, SimOutputTransfer,
};

#[test]
fn sp_mtq_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 128.0;
    let dt = 0.1;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s5e_lib::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );
    let mut sun = s5e_lib::sun::Sun::new(datetime);
    let mut earth = s5e_lib::earth::Earth::new(datetime);
    let mut moon = s5e_lib::moon::Moon::new(datetime);

    let sat_initial_state = c5a_sim::SpaceCraftInitialState {
        tle: tle.clone(),
        attitude: UnitQuaternion::new_normalize(Quaternion::new(
            -1.0, 4.0, -3.0, 2.0,
        )),
        angular_velocity: BodyVector {
            x: 0.00,
            y: 0.00,
            z: 0.00,
        },
        inertia: Matrix3::new(
            0.168125, 0.001303, 0.000698, 0.001303, 0.183472, 0.000542, 0.000698, 0.000542,
            0.111208,
        ),
    };

    let components_profile = c5a_sim::SpaceCraftComponentsProfile {
        gyro_std: c5a::constants::GYRO_NOISE_STD,
        magnetometer_std: c5a::constants::MAGNETOMETER_NOISE_STD,
        star_tracker_std: c5a::constants::STAR_TRACKER_NOISE_STD,
        sun_sensor_dir_std: c5a::constants::SUN_SENSOR_NOISE_STD,
        rw_noise_std: 0.0,
        mtq_noise_std: 0.0,
        mtq_max_dipole_moment: c5a::constants::MTQ_MAX_DIPOLE_MOMENT,
    };

    let mut sat = c5a_sim::SpaceCraft::new(
        datetime,
        rng,
        c5a::Fsw::new(false),
        components_profile,
        sat_initial_state,
    );

    let mut cmd_port = s5e_port::S5EPublishPort::<s5e_port::GSCommandData>::new();

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/sp_mtq")?;
    let mut writer = Writer::from_path("tests/out/sp_mtq/sun_direction_error.csv")?;

    while time < end_time {
        // tick earth
        earth.tick(dt, datetime);
        // tick moon
        moon.tick(dt, datetime);
        // tick sun
        sun.tick(dt, datetime);

        // tick satellite
        sat.tick(
            dt,
            datetime,
            sun.light_source(),
            vec![earth.shadow_source(), moon.shadow_source()],
            &cmd_port,
        );

        // Calculate sun direction error
        let attitude_q = UnitQuaternion::from_quaternion(sat.attitude.get_now());
        let sat_z_axis = attitude_q.conjugate() * Vector3::z();
        let sun_dir = Vector3::from(sun.position.get_now() - sat.eci_position.get_now()).normalize();
        let angle_error = sat_z_axis.angle(&sun_dir);

        // Calculate angular momentum in ECI frame: H = R * I * ω
        // where R is body-to-ECI rotation, I is inertia, ω is angular velocity in body
        let omega_body = Vector3::new(
            sat.angular_velocity.get_now().x,
            sat.angular_velocity.get_now().y,
            sat.angular_velocity.get_now().z,
        );
        let angular_momentum_body = sat.inertia * omega_body;
        let angular_momentum_eci = attitude_q.conjugate() * angular_momentum_body;

        // Get magnetic field direction in ECI (already in nT)
        let mag_field_eci = Vector3::new(
            sat.magnetic_field.x,
            sat.magnetic_field.y,
            sat.magnetic_field.z,
        );
        let mag_field_dir = mag_field_eci.normalize();

        // Calculate control error vector: sat_z × sun_dir (in ECI)
        // This is proportional to the torque needed to align sat_z with sun_dir
        let control_error_eci = sat_z_axis.cross(&sun_dir);

        // Record sun direction error data with angular momentum and magnetic field
        writer.write_record(&[
            time.to_string(),
            angle_error.to_degrees().to_string(),
            sat_z_axis.x.to_string(),
            sat_z_axis.y.to_string(),
            sat_z_axis.z.to_string(),
            sun_dir.x.to_string(),
            sun_dir.y.to_string(),
            sun_dir.z.to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
            mag_field_dir.x.to_string(),
            mag_field_dir.y.to_string(),
            mag_field_dir.z.to_string(),
            control_error_eci.x.to_string(),
            control_error_eci.y.to_string(),
            control_error_eci.z.to_string(),
            sat.shadow_coefficient.value.to_string(),
        ])?;

        // Dump status at around 13000 seconds for debugging
        if (time - 13000.0_f64).abs() < dt {
            let q = sat.attitude.get_now();
            let omega = sat.angular_velocity.get_now();
            println!("\n=== Status dump at t = {:.1} sec ===", time);
            println!("Datetime: {}", datetime);
            println!("Attitude quaternion (w, x, y, z): ({:.10}, {:.10}, {:.10}, {:.10})", q.w, q.i, q.j, q.k);
            println!("Angular velocity (body, rad/s): ({:.10}, {:.10}, {:.10})", omega.x, omega.y, omega.z);
            println!("Angle error: {:.4} deg", angle_error.to_degrees());
            println!("Sat Z-axis (ECI): ({:.6}, {:.6}, {:.6})", sat_z_axis.x, sat_z_axis.y, sat_z_axis.z);
            println!("Sun direction (ECI): ({:.6}, {:.6}, {:.6})", sun_dir.x, sun_dir.y, sun_dir.z);
            println!("Angular momentum (ECI, kg·m²/s): ({:.10}, {:.10}, {:.10})", angular_momentum_eci.x, angular_momentum_eci.y, angular_momentum_eci.z);
            println!("Magnetic field dir (ECI): ({:.6}, {:.6}, {:.6})", mag_field_dir.x, mag_field_dir.y, mag_field_dir.z);
            println!("Control error (sat_z × sun): ({:.6}, {:.6}, {:.6})", control_error_eci.x, control_error_eci.y, control_error_eci.z);
            println!("Shadow coefficient: {:.4}", sat.shadow_coefficient.value);
            println!("===================================\n");
        }

        // update time
        time += dt;
        datetime += TimeDelta::nanoseconds((dt * 1.0e9) as i64);
        // Clear states for next iteration
        sat.clear_state();
        sun.clear_state();
        earth.clear_state();
        moon.clear_state();
        cmd_port.clear();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_sp_mtq.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    let sat_z_axis = UnitQuaternion::from_quaternion(sat.attitude.get_now()).conjugate()
        * Vector3::z();
    let sun_dir = Vector3::from(sun.position.get_now());
    let angle = sat_z_axis.angle(&sun_dir);
    if angle < 10.0f64.to_radians() {
        println!("Sun Pointing MTQ convergence test passed.");
        Ok(())
    } else {
        anyhow::bail!(
            "Sun Pointing MTQ convergence test failed. Final angle: {} deg",
            angle.to_degrees()
        );
    }
}

// ============================================================================
// MTQ Sun Pointing FSW for Virtual Magnetic Field Simulation
// ============================================================================

/// FSW for MTQ-based sun pointing control with virtual magnetic field
/// This is a simplified FSW that directly uses ports without drivers
pub struct SPMtqFsw {
    pub time: c5a::Time,
    #[allow(dead_code)]
    timer: c5a::estimation::Timer,

    // Direct sensor ports (no drivers for simplicity)
    pub gyro_port: s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer_port: s5e_port::S5ESubscribePort<s5e_port::MagnetometerData>,
    pub sun_dir_port: s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,

    // MTQ output port
    pub mtq_ctrl_port: s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,

    // Controller gains
    p_gain: Matrix3<f64>,
    d_gain: Matrix3<f64>,

    inertia: Matrix3<f64>,
}

impl Default for SPMtqFsw {
    fn default() -> Self {
        Self::new()
    }
}

impl SPMtqFsw {
    pub fn new() -> Self {
        Self {
            time: c5a::Time::new(),
            timer: c5a::estimation::Timer::new(),

            gyro_port: s5e_port::S5ESubscribePort::new(),
            magnetometer_port: s5e_port::S5ESubscribePort::new(),
            sun_dir_port: s5e_port::S5ESubscribePort::new(),
            mtq_ctrl_port: s5e_port::S5EPublishPort::new(),

            // Controller gains (tuned for MTQ control)
            // Lower gains for stability with limited MTQ authority
            p_gain: Matrix3::identity() * 0.1,
            d_gain: Matrix3::identity() * 50.0,
            inertia: Matrix3::new(
                0.168125, 0.001303, 0.000698,
                0.001303, 0.183472, 0.000542,
                0.000698, 0.000542, 0.111208,
            )
        }
    }

    pub fn init(&mut self) {
        println!("SPMtqFsw initialized");
        *self = Self::new();
    }

    pub fn main_loop(&mut self, _dt: f64) {
        // Get sensor readings directly from ports
        let gyro_data = self.gyro_port.subscribe();
        let mag_data = self.magnetometer_port.subscribe();
        let sun_data = self.sun_dir_port.subscribe();

        // Calculate MTQ control
        if let (Some(mag), Some(gyro), Some(sun)) = (mag_data, gyro_data, sun_data) {
            // Convert magnetic field from nT to T
            let mag_field_nt = mag.magnetic_field;
            let mag_field = mag_field_nt * 1e-9;  // nT -> T
            let ang_vel = gyro.angular_velocity;
            let sun_dir = UnitVector3::new_normalize(sun.light_direction);

            // Lyapunov-based sun pointing control
            //
            // Lyapunov function: V = (1/2) ω^T I ω + k_p (1 - s·z)
            // where s is sun direction in body frame, z is target axis (+Z)
            //
            // V̇ = ω^T τ - k_p (ω × s)·z
            //    = ω^T τ - k_p ω·(s × z)
            //    = ω·(τ - k_p (s × z))
            //
            // For stability (V̇ ≤ 0), choose τ_desired such that:
            // τ_desired = k_p (s × z) - k_d ω
            //
            // Then V̇ = ω·(τ - k_p (s × z)) = -k_d |ω|² ≤ 0 (if τ = τ_desired)
            //
            // However, MTQ can only generate torque perpendicular to B:
            // τ_actual = m × B
            //
            // Using m = (B × τ_desired) / |B|²:
            // τ_actual = ((B × τ_desired) × B) / |B|² = τ_desired - (τ_desired·b̂)b̂
            //
            // This gives the component of τ_desired perpendicular to B.

            let target_axis = Vector3::z();
            let b_hat = mag_field.normalize();

            // Error term: s × z (cross product for rotation)
            // This is the rotation axis needed to align z with s
            let pointing_error_plane = sun_dir.cross(&target_axis).normalize();
            let pointing_error_angle = sun_dir.angle(&target_axis);
            let pointing_error = pointing_error_plane * pointing_error_angle;            

            // Desired moment
            // Cross Product
            let moment_p = self.p_gain * b_hat.cross(&pointing_error);
            let moment_d = self.d_gain * b_hat.cross(&(self.inertia * ang_vel));
            let magnetic_moment = moment_p - moment_d;
            self.mtq_ctrl_port.publish(s5e_port::MagnetorquerCtrlEvent {
                magnetic_moment,
            });
        }
    }
}

pub struct SPMtqFswInputPortSet<'a> {
    pub gyro: &'a mut s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer: &'a mut s5e_port::S5ESubscribePort<s5e_port::MagnetometerData>,
    pub sun_direction: &'a mut s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,
}

impl<'a> SimInputTransfer<VirtualMagFieldSimSensorOutput<'a>> for SPMtqFswInputPortSet<'a> {
    fn transfer_from(&mut self, sensor_output: &VirtualMagFieldSimSensorOutput<'a>) {
        s5e_port::transfer(sensor_output.gyro, self.gyro);
        s5e_port::transfer(sensor_output.magnetometer, self.magnetometer);
        s5e_port::transfer(sensor_output.sun_sensor, self.sun_direction);
    }
}

pub struct SPMtqFswOutputPortSet<'a> {
    pub mtq_ctrl: &'a s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,
}

impl<'a> SimOutputTransfer<VirtualMagFieldSimActuatorInput<'a>> for SPMtqFswOutputPortSet<'a> {
    fn transfer_to(&self, actuator_input: &mut VirtualMagFieldSimActuatorInput<'a>) {
        s5e_port::transfer(self.mtq_ctrl, actuator_input.mtq_ctrl);
    }
}

impl VirtualMagFieldSimInterface for SPMtqFsw {
    type InputPortSet<'a> = SPMtqFswInputPortSet<'a>;
    type OutputPortSet<'a> = SPMtqFswOutputPortSet<'a>;

    fn init(&mut self) {
        self.init();
    }

    fn main_loop(&mut self, dt: f64) {
        self.main_loop(dt);
    }

    fn input_ports(&mut self) -> Self::InputPortSet<'_> {
        SPMtqFswInputPortSet {
            gyro: &mut self.gyro_port,
            magnetometer: &mut self.magnetometer_port,
            sun_direction: &mut self.sun_dir_port,
        }
    }

    fn output_ports(&mut self) -> Self::OutputPortSet<'_> {
        SPMtqFswOutputPortSet {
            mtq_ctrl: &self.mtq_ctrl_port,
        }
    }
}

// ============================================================================
// Energy-Dissipative MTQ Controller
// ============================================================================

/// Control modes for energy-dissipative MTQ control
#[derive(Clone, Copy, Debug)]
pub enum MtqControlMode {
    /// Pure damping: m = -k_d (b̂ × ω)
    /// Only reduces kinetic energy, no pointing control
    PureDamping,
    /// Lyapunov-based pointing: m = (B × τ_d) / |B|²
    /// where τ_d = k_p (s × z) - k_d I ω
    LyapunovPointing,
    /// Conditional control: use pointing only when V̇ < 0 is guaranteed
    ConditionalPointing,
    /// Gain-scheduled: reduce pointing gain near target to avoid energy injection
    GainScheduled,
    /// Cross-product control: classical approach m ∝ B × (s × z)
    CrossProduct,
    /// Slow control: very conservative gains for small errors
    /// Uses higher damping-to-pointing ratio to prevent energy injection
    SlowControl,
    /// Periodic averaging: only applies control when aligned with energy dissipation
    PeriodicAveraging { phase_accumulator: f64 },
}

/// FSW with energy-dissipative MTQ control
pub struct EnergyDissipativeMtqFsw {
    pub gyro_port: s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer_port: s5e_port::S5ESubscribePort<s5e_port::MagnetometerData>,
    pub sun_dir_port: s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,
    pub mtq_ctrl_port: s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,

    pub mode: MtqControlMode,
    k_p: f64,
    k_d: f64,
    inertia: Matrix3<f64>,
}

impl EnergyDissipativeMtqFsw {
    pub fn new(mode: MtqControlMode, k_p: f64, k_d: f64) -> Self {
        Self {
            gyro_port: s5e_port::S5ESubscribePort::new(),
            magnetometer_port: s5e_port::S5ESubscribePort::new(),
            sun_dir_port: s5e_port::S5ESubscribePort::new(),
            mtq_ctrl_port: s5e_port::S5EPublishPort::new(),
            mode,
            k_p,
            k_d,
            inertia: Matrix3::new(
                0.168125, 0.001303, 0.000698,
                0.001303, 0.183472, 0.000542,
                0.000698, 0.000542, 0.111208,
            ),
        }
    }

    pub fn main_loop(&mut self, _dt: f64) {
        let gyro_data = self.gyro_port.subscribe();
        let mag_data = self.magnetometer_port.subscribe();
        let sun_data = self.sun_dir_port.subscribe();

        if let (Some(mag), Some(gyro), Some(sun)) = (mag_data, gyro_data, sun_data) {
            let mag_field_nt = mag.magnetic_field;
            let mag_field = mag_field_nt * 1e-9;
            let ang_vel = gyro.angular_velocity;
            let sun_dir = UnitVector3::new_normalize(sun.light_direction);
            let target_axis = Vector3::z();

            let mag_field_norm = mag_field.norm();
            if mag_field_norm < 1e-15 {
                return;
            }
            let b_hat = mag_field / mag_field_norm;

            let moment = match self.mode {
                MtqControlMode::PureDamping => {
                    // Pure damping: m = -k_d (b̂ × ω)
                    // Produces torque: τ = m × B = -k_d (b̂ × ω) × B
                    //                           = -k_d |B| (b̂ × ω) × b̂
                    //                           = -k_d |B| [ω - (ω·b̂)b̂]
                    // This removes angular velocity perpendicular to B
                    //
                    // Energy rate: V̇ = ω·τ = -k_d |B| |ω|² + k_d |B| (ω·b̂)²
                    //                       = -k_d |B| [|ω|² - (ω·b̂)²]
                    //                       = -k_d |B| |b̂ × ω|² ≤ 0
                    -self.k_d * b_hat.cross(&ang_vel)
                }
                MtqControlMode::LyapunovPointing => {
                    // Desired torque: τ_d = k_p (s × z) - k_d I ω
                    let pointing_error = sun_dir.cross(&target_axis);
                    let torque_p = self.k_p * pointing_error;
                    let torque_d = self.k_d * self.inertia * ang_vel;
                    let desired_torque = torque_p - torque_d;

                    // Magnetic moment: m = (B × τ_d) / |B|²
                    mag_field.cross(&desired_torque) / mag_field.norm_squared()
                }
                MtqControlMode::ConditionalPointing => {
                    // Lyapunov function: V = (1/2) ω^T I ω + k_p (1 - s·z)
                    //
                    // Desired torque: τ_d = k_p (s × z) - k_d I ω
                    // Actual torque: τ = τ_d - (τ_d·b̂)b̂
                    //
                    // V̇ = ω·τ - k_p ω·(s × z)
                    //    = ω·[τ_d - (τ_d·b̂)b̂] - k_p ω·(s × z)
                    //    = ω·τ_d - (τ_d·b̂)(ω·b̂) - k_p ω·(s × z)
                    //    = ω·[k_p(s×z) - k_d I ω] - (τ_d·b̂)(ω·b̂) - k_p ω·(s×z)
                    //    = -k_d ω·I·ω - (τ_d·b̂)(ω·b̂)
                    //
                    // The term -(τ_d·b̂)(ω·b̂) can be positive.
                    // For guaranteed dissipation, check if V̇ < 0

                    let pointing_error = sun_dir.cross(&target_axis);
                    let torque_p = self.k_p * pointing_error;
                    let torque_d = self.k_d * self.inertia * ang_vel;
                    let desired_torque = torque_p - torque_d;

                    // Calculate V̇ estimate
                    let kinetic_dissipation = -self.k_d * ang_vel.dot(&(self.inertia * ang_vel));
                    let coupling_term = -(desired_torque.dot(&b_hat)) * (ang_vel.dot(&b_hat));
                    let v_dot_estimate = kinetic_dissipation + coupling_term;

                    if v_dot_estimate < 0.0 {
                        // Use full Lyapunov control
                        mag_field.cross(&desired_torque) / mag_field.norm_squared()
                    } else {
                        // Fall back to pure damping
                        -self.k_d * b_hat.cross(&ang_vel)
                    }
                }
                MtqControlMode::GainScheduled => {
                    // Gain-scheduled control: reduce k_p near the target to avoid
                    // energy injection when close to equilibrium.
                    //
                    // The idea: near the target, the coupling term can dominate
                    // and cause energy increase. By reducing k_p, we reduce the
                    // pointing torque and rely more on damping.
                    //
                    // Schedule: k_p_eff = k_p * sin(θ)² where θ is pointing error
                    // This gives k_p_eff → 0 as θ → 0

                    let pointing_error = sun_dir.cross(&target_axis);
                    let sin_theta_sq = pointing_error.norm_squared();  // |s × z|² = sin²(θ)

                    // Use scheduled gain
                    let k_p_scheduled = self.k_p * sin_theta_sq;

                    let torque_p = k_p_scheduled * pointing_error;
                    let torque_d = self.k_d * self.inertia * ang_vel;
                    let desired_torque = torque_p - torque_d;

                    mag_field.cross(&desired_torque) / mag_field.norm_squared()
                }
                MtqControlMode::CrossProduct => {
                    // Classical cross-product law for MTQ pointing:
                    // m = k_p (B × e) - k_d (B × ω)
                    // where e = s × z is the pointing error
                    //
                    // This directly computes magnetic moment without going
                    // through desired torque → magnetic moment conversion.
                    //
                    // The torque produced is:
                    // τ = m × B = k_p (B × e) × B - k_d (B × ω) × B
                    //           = k_p |B|²[e - (e·b̂)b̂] - k_d |B|²[ω - (ω·b̂)b̂]
                    //
                    // This removes components parallel to B from both error and damping.

                    let pointing_error = sun_dir.cross(&target_axis);

                    // Scale by 1/|B|² to normalize
                    let scale = 1.0 / mag_field.norm_squared();
                    let moment_p = self.k_p * mag_field.cross(&pointing_error) * scale;
                    let moment_d = self.k_d * mag_field.cross(&ang_vel) * scale;

                    moment_p - moment_d
                }
                MtqControlMode::SlowControl => {
                    // Very conservative control for small error situations.
                    // Key insight: energy injection occurs when pointing torque
                    // causes angular velocity that doesn't align with the desired motion.
                    //
                    // Strategy:
                    // 1. Use much higher damping ratio (10x normal)
                    // 2. Scale pointing gain by error magnitude cubed (very slow near target)
                    // 3. Only apply pointing when damping term dominates

                    let pointing_error = sun_dir.cross(&target_axis);
                    let error_norm = pointing_error.norm();
                    let omega_norm = ang_vel.norm();

                    // Scale pointing gain: k_p_eff = k_p * |error|²
                    // This makes control very weak near equilibrium
                    let k_p_scaled = self.k_p * error_norm * error_norm;

                    // Use 10x damping
                    let k_d_scaled = self.k_d * 10.0;

                    // Only apply pointing if error is significant relative to velocity
                    let pointing_dominance = if omega_norm > 1e-10 {
                        error_norm / omega_norm
                    } else {
                        1.0
                    };

                    let torque_p = if pointing_dominance > 0.1 {
                        k_p_scaled * pointing_error
                    } else {
                        Vector3::zeros()
                    };
                    let torque_d = k_d_scaled * self.inertia * ang_vel;
                    let desired_torque = torque_p - torque_d;

                    mag_field.cross(&desired_torque) / mag_field.norm_squared()
                }
                MtqControlMode::PeriodicAveraging { .. } => {
                    // Control based on periodic averaging over magnetic field rotation.
                    // The magnetic field rotates with period T = 2π/ω_B.
                    // We want the time-averaged energy rate to be negative.
                    //
                    // Simplified approach: use damping-dominant control with
                    // pointing term scaled by instantaneous energy rate sign.

                    let pointing_error = sun_dir.cross(&target_axis);
                    let torque_d = self.k_d * self.inertia * ang_vel;

                    // Estimate if pointing control would add or remove energy
                    // V̇_pointing ≈ ω · τ_pointing = ω · (τ_p - (τ_p·b̂)b̂)
                    let torque_p_desired = self.k_p * pointing_error;
                    let torque_p_actual = torque_p_desired - torque_p_desired.dot(&b_hat) * b_hat;
                    let energy_rate_pointing = ang_vel.dot(&torque_p_actual);

                    // Only add pointing term if it would reduce energy
                    let torque_p = if energy_rate_pointing < 0.0 {
                        torque_p_desired
                    } else {
                        // Reduce pointing gain when it would add energy
                        torque_p_desired * 0.1
                    };

                    let desired_torque = torque_p - torque_d;
                    mag_field.cross(&desired_torque) / mag_field.norm_squared()
                }
            };

            self.mtq_ctrl_port.publish(s5e_port::MagnetorquerCtrlEvent {
                magnetic_moment: moment,
            });
        }
    }
}

pub struct EnergyDissipativeFswInputPortSet<'a> {
    pub gyro: &'a mut s5e_port::S5ESubscribePort<s5e_port::GyroSensorData>,
    pub magnetometer: &'a mut s5e_port::S5ESubscribePort<s5e_port::MagnetometerData>,
    pub sun_direction: &'a mut s5e_port::S5ESubscribePort<s5e_port::LightDirectionData>,
}

impl<'a> SimInputTransfer<VirtualMagFieldSimSensorOutput<'a>> for EnergyDissipativeFswInputPortSet<'a> {
    fn transfer_from(&mut self, sensor_output: &VirtualMagFieldSimSensorOutput<'a>) {
        s5e_port::transfer(sensor_output.gyro, self.gyro);
        s5e_port::transfer(sensor_output.magnetometer, self.magnetometer);
        s5e_port::transfer(sensor_output.sun_sensor, self.sun_direction);
    }
}

pub struct EnergyDissipativeFswOutputPortSet<'a> {
    pub mtq_ctrl: &'a s5e_port::S5EPublishPort<s5e_port::MagnetorquerCtrlEvent>,
}

impl<'a> SimOutputTransfer<VirtualMagFieldSimActuatorInput<'a>> for EnergyDissipativeFswOutputPortSet<'a> {
    fn transfer_to(&self, actuator_input: &mut VirtualMagFieldSimActuatorInput<'a>) {
        s5e_port::transfer(self.mtq_ctrl, actuator_input.mtq_ctrl);
    }
}

impl VirtualMagFieldSimInterface for EnergyDissipativeMtqFsw {
    type InputPortSet<'a> = EnergyDissipativeFswInputPortSet<'a>;
    type OutputPortSet<'a> = EnergyDissipativeFswOutputPortSet<'a>;

    fn init(&mut self) {}

    fn main_loop(&mut self, dt: f64) {
        self.main_loop(dt);
    }

    fn input_ports(&mut self) -> Self::InputPortSet<'_> {
        EnergyDissipativeFswInputPortSet {
            gyro: &mut self.gyro_port,
            magnetometer: &mut self.magnetometer_port,
            sun_direction: &mut self.sun_dir_port,
        }
    }

    fn output_ports(&mut self) -> Self::OutputPortSet<'_> {
        EnergyDissipativeFswOutputPortSet {
            mtq_ctrl: &self.mtq_ctrl_port,
        }
    }
}

#[test]
fn sp_mtq_virtual_magfield_convergence() -> anyhow::Result<()> {
    let mut time = 0.0;
    let end_time = 20000.0;
    let dt = 0.1;

    // Initial conditions
    let initial_angular_velocity = BodyVector {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698,
        0.001303, 0.183472, 0.000542,
        0.000698, 0.000542, 0.111208,
    );

    // Sun direction in ECI (fixed) - not parallel to initial magnetic field
    let sun_direction_eci = ECIVector {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    let initial_attitude = UnitQuaternion::new(Vector3::new(0.01,0.01,0.01)) * c5a::estimation::triad_method(&Vector3::x(), &Vector3::z(),&Vector3::x(), &Vector3::y());
    let initial_attitude = UnitQuaternion::new_normalize(Quaternion::new(
            -1.0, 4.0, -3.0, 2.0,
        ));

    // Virtual magnetic field model
    // Simulates magnetic field rotation similar to LEO orbit (~90 min period)
    let magnetic_field_model = VirtualMagneticFieldModel::new(
        40000.0,  // 40000 nT (typical LEO)
        2.0 * std::f64::consts::PI / 5400.0,  // ~90 min period
        Vector3::new(0.0, 0.0, 1.0),  // Rotation around Z-axis
        Vector3::new(1.0, 0.0, 0.0),  // Initial direction along X
    );

    // Create FSW and simulation
    let fsw = SPMtqFsw::new();
    let mut sim = VirtualMagFieldSim::new(
        inertia,
        fsw,
        initial_attitude,
        initial_angular_velocity,
        sun_direction_eci.clone(),
        magnetic_field_model,
    );

    // Setup CSV writer for test output
    fs::create_dir_all("tests/out/sp_mtq_virtual")?;
    let mut writer = Writer::from_path("tests/out/sp_mtq_virtual/sun_direction_error.csv")?;

    while time < end_time {
        // Publish sensor data to FSW
        let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
        let omega = sim.angular_velocity.get_now();

        // Publish gyro data
        sim.gyro_port.publish(s5e_port::GyroSensorData {
            angular_velocity: Vector3::new(omega.x, omega.y, omega.z),
        });

        // Publish sun sensor data (sun direction in body frame)
        let sun_dir_body = BodyVector::from_eci(sun_direction_eci.clone(), attitude_q);
        sim.sun_sensor_port.publish(s5e_port::LightDirectionData {
            light_direction: Vector3::new(sun_dir_body.x, sun_dir_body.y, sun_dir_body.z),
        });

        // Publish magnetometer data (magnetic field in body frame)
        let mag_field_body = sim.magnetic_field_body();
        sim.magnetometer_port.publish(s5e_port::MagnetometerData {
            magnetic_field: Vector3::new(mag_field_body.x, mag_field_body.y, mag_field_body.z),
        });

        // Tick simulation
        sim.tick(dt);

        // Calculate sun direction error
        let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
        let sat_z_axis_eci = attitude_q.conjugate() * Vector3::z();
        let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
        let angle_error = sat_z_axis_eci.angle(&sun_dir_eci);

        // Calculate angular velocity and angular momentum
        let omega = sim.angular_velocity.get_now();
        let omega_vec = Vector3::new(omega.x, omega.y, omega.z);
        let omega_norm = omega_vec.norm();
        let angular_momentum_body = inertia * omega_vec;
        let angular_momentum_eci = attitude_q.conjugate() * angular_momentum_body;
        let _angular_momentum_norm = angular_momentum_eci.norm();

        // Get magnetic field direction in ECI
        let mag_field_eci = sim.magnetic_field_eci();
        let mag_field_dir = Vector3::new(mag_field_eci.x, mag_field_eci.y, mag_field_eci.z).normalize();

        // Control error: sat_z × sun_dir (in ECI)
        let control_error_eci = sat_z_axis_eci.cross(&sun_dir_eci);

        // Write to CSV
        writer.write_record(&[
            time.to_string(),
            angle_error.to_degrees().to_string(),
            sat_z_axis_eci.x.to_string(),
            sat_z_axis_eci.y.to_string(),
            sat_z_axis_eci.z.to_string(),
            sun_dir_eci.x.to_string(),
            sun_dir_eci.y.to_string(),
            sun_dir_eci.z.to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
            mag_field_dir.x.to_string(),
            mag_field_dir.y.to_string(),
            mag_field_dir.z.to_string(),
            control_error_eci.x.to_string(),
            control_error_eci.y.to_string(),
            control_error_eci.z.to_string(),
            omega_norm.to_string(),
        ])?;

        // Update time
        time += dt;
        sim.clear_state();
    }

    writer.flush()?;

    // Run Python plotting script
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_sp_mtq_virtual.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    // Check convergence
    let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
    let sat_z_axis_eci = attitude_q.conjugate() * Vector3::z();
    let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
    let final_angle_error = sat_z_axis_eci.angle(&sun_dir_eci);

    println!("Final sun pointing error: {:.2} deg", final_angle_error.to_degrees());

    if final_angle_error < 30.0_f64.to_radians() {
        println!("Sun Pointing MTQ (virtual magfield) convergence test passed.");
        Ok(())
    } else {
        anyhow::bail!(
            "Sun Pointing MTQ (virtual magfield) convergence test failed. Final angle: {:.2} deg",
            final_angle_error.to_degrees()
        );
    }
}

/// Helper function to run simulation with a specific control mode and track energy
fn run_energy_tracking_sim(
    mode: MtqControlMode,
    initial_attitude: UnitQuaternion<f64>,
    initial_angular_velocity: BodyVector,
    sun_direction_eci: ECIVector,
    inertia: Matrix3<f64>,
    k_p: f64,
    k_d: f64,
    end_time: f64,
    dt: f64,
    output_prefix: &str,
) -> anyhow::Result<(f64, f64, Vec<f64>)> {
    // Virtual magnetic field model
    let magnetic_field_model = VirtualMagneticFieldModel::new(
        40000.0,  // 40000 nT
        2.0 * std::f64::consts::PI / 5400.0,  // ~90 min period
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 0.0),
    );

    let fsw = EnergyDissipativeMtqFsw::new(mode, k_p, k_d);
    let mut sim = VirtualMagFieldSim::new(
        inertia,
        fsw,
        initial_attitude,
        initial_angular_velocity.clone(),
        sun_direction_eci.clone(),
        magnetic_field_model,
    );

    let output_dir = format!("tests/out/{}", output_prefix);
    fs::create_dir_all(&output_dir)?;
    let mut writer = Writer::from_path(format!("{}/energy_tracking.csv", output_dir))?;

    let mut time = 0.0;
    let mut energy_history = Vec::new();

    while time < end_time {
        let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
        let omega = sim.angular_velocity.get_now();

        // Publish sensor data
        sim.gyro_port.publish(s5e_port::GyroSensorData {
            angular_velocity: Vector3::new(omega.x, omega.y, omega.z),
        });
        let sun_dir_body = BodyVector::from_eci(sun_direction_eci.clone(), attitude_q);
        sim.sun_sensor_port.publish(s5e_port::LightDirectionData {
            light_direction: Vector3::new(sun_dir_body.x, sun_dir_body.y, sun_dir_body.z),
        });
        let mag_field_body = sim.magnetic_field_body();
        sim.magnetometer_port.publish(s5e_port::MagnetometerData {
            magnetic_field: Vector3::new(mag_field_body.x, mag_field_body.y, mag_field_body.z),
        });

        sim.tick(dt);

        // Calculate energy components
        let attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
        let omega = sim.angular_velocity.get_now();
        let omega_vec = Vector3::new(omega.x, omega.y, omega.z);

        // Kinetic energy: (1/2) ω^T I ω
        let kinetic_energy = 0.5 * omega_vec.dot(&(inertia * omega_vec));

        // Potential energy: k_p (1 - s·z)
        // s = sun direction in body frame
        // z = target axis (+Z)
        let sun_dir_body = BodyVector::from_eci(sun_direction_eci.clone(), attitude_q);
        let sun_dir_body_vec = Vector3::new(sun_dir_body.x, sun_dir_body.y, sun_dir_body.z).normalize();
        let target_axis = Vector3::z();
        let cos_angle = sun_dir_body_vec.dot(&target_axis);
        let potential_energy = k_p * (1.0 - cos_angle);

        // Total Lyapunov energy
        let total_energy = kinetic_energy + potential_energy;
        energy_history.push(total_energy);

        // Angle error
        let sat_z_axis_eci = attitude_q.conjugate() * Vector3::z();
        let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
        let angle_error = sat_z_axis_eci.angle(&sun_dir_eci);

        // Angular momentum (for analysis)
        let angular_momentum_body = inertia * omega_vec;
        let angular_momentum_eci = attitude_q.conjugate() * angular_momentum_body;

        writer.write_record(&[
            time.to_string(),
            angle_error.to_degrees().to_string(),
            kinetic_energy.to_string(),
            potential_energy.to_string(),
            total_energy.to_string(),
            omega_vec.norm().to_string(),
            angular_momentum_eci.x.to_string(),
            angular_momentum_eci.y.to_string(),
            angular_momentum_eci.z.to_string(),
        ])?;

        time += dt;
        sim.clear_state();
    }

    writer.flush()?;

    let initial_angle = {
        let sat_z = initial_attitude.conjugate() * Vector3::z();
        let sun_dir = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
        sat_z.angle(&sun_dir).to_degrees()
    };

    let final_attitude_q = UnitQuaternion::from_quaternion(sim.attitude.get_now());
    let sat_z_axis_eci = final_attitude_q.conjugate() * Vector3::z();
    let sun_dir_eci = Vector3::new(sun_direction_eci.x, sun_direction_eci.y, sun_direction_eci.z).normalize();
    let final_angle = sat_z_axis_eci.angle(&sun_dir_eci).to_degrees();

    Ok((initial_angle, final_angle, energy_history))
}

/// Analyze peak behavior in energy time series
/// Returns: (peaks_times, peaks_values, decay_rate)
fn analyze_peak_decay(energy: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let mut peaks_times = Vec::new();
    let mut peaks_values = Vec::new();

    // Find local maxima (peaks)
    // Use a smaller window for finer peak detection
    let window = 20; // 2 seconds at dt=0.1
    for i in window..(energy.len() - window) {
        let center = energy[i];
        let is_peak = (1..=window).all(|j| energy[i - j] <= center)
            && (1..=window).all(|j| energy[i + j] <= center);

        if is_peak {
            // Check if this is significantly higher than neighbors
            let left_min = energy[(i - window)..i].iter().copied().fold(f64::INFINITY, f64::min);
            let right_min = energy[(i + 1)..(i + window + 1)].iter().copied().fold(f64::INFINITY, f64::min);
            let prominence = center - left_min.max(right_min);

            // Only count as peak if prominence is significant (relative to energy scale)
            let threshold = (center * 0.005).max(1e-8);
            if prominence > threshold {
                peaks_times.push(i as f64 * dt);
                peaks_values.push(center);
            }
        }
    }

    // Calculate decay rate using exponential fit: peak(t) = A * exp(-λ*t)
    // ln(peak) = ln(A) - λ*t
    // Linear regression on (t, ln(peak))
    let decay_rate = if peaks_values.len() >= 2 {
        let n = peaks_values.len() as f64;
        let sum_t: f64 = peaks_times.iter().sum();
        let sum_ln_p: f64 = peaks_values.iter().filter(|p| **p > 0.0).map(|p| p.ln()).sum();
        let sum_t_ln_p: f64 = peaks_times.iter().zip(peaks_values.iter())
            .filter(|(_, p)| **p > 0.0)
            .map(|(t, p)| t * p.ln()).sum();
        let sum_t2: f64 = peaks_times.iter().map(|t| t * t).sum();

        // Slope of linear regression = -λ
        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() > 1e-10 {
            let slope = (n * sum_t_ln_p - sum_t * sum_ln_p) / denom;
            -slope // λ (positive means decaying)
        } else {
            0.0
        }
    } else {
        0.0
    };

    (peaks_times, peaks_values, decay_rate)
}

/// Compute running maximum envelope
fn compute_envelope(energy: &[f64], window: usize) -> Vec<f64> {
    energy.iter().enumerate().map(|(i, _)| {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(energy.len());
        energy[start..end].iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }).collect()
}

/// Analyze energy envelope decay
fn analyze_envelope_decay(energy: &[f64], dt: f64, sample_interval: usize) -> f64 {
    // Compute envelope with ~100 second window
    let window = (100.0 / dt) as usize;
    let envelope = compute_envelope(energy, window);

    // Sample envelope at regular intervals
    let samples: Vec<(f64, f64)> = (0..energy.len())
        .step_by(sample_interval)
        .filter(|&i| envelope[i] > 1e-10)
        .map(|i| (i as f64 * dt, envelope[i]))
        .collect();

    if samples.len() < 2 {
        return 0.0;
    }

    // Linear regression on log(envelope)
    let n = samples.len() as f64;
    let sum_t: f64 = samples.iter().map(|(t, _)| t).sum();
    let sum_ln_e: f64 = samples.iter().map(|(_, e)| e.ln()).sum();
    let sum_t_ln_e: f64 = samples.iter().map(|(t, e)| t * e.ln()).sum();
    let sum_t2: f64 = samples.iter().map(|(t, _)| t * t).sum();

    let denom = n * sum_t2 - sum_t * sum_t;
    if denom.abs() > 1e-10 {
        let slope = (n * sum_t_ln_e - sum_t * sum_ln_e) / denom;
        -slope // positive = envelope decaying
    } else {
        0.0
    }
}

/// Comprehensive peak analysis result
#[derive(Debug, Clone)]
struct PeakAnalysis {
    name: String,
    final_angle: f64,
    num_peaks: usize,
    first_peak: f64,
    last_peak: f64,
    peak_decay_rate: f64,
    envelope_decay_rate: f64,
    peak_ratio: f64, // last_peak / first_peak (< 1 means decaying)
    max_energy: f64,
    final_energy: f64,
    peaks_times: Vec<f64>,
    peaks_values: Vec<f64>,
}

#[test]
fn sp_mtq_energy_comparison() -> anyhow::Result<()> {
    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698,
        0.001303, 0.183472, 0.000542,
        0.000698, 0.000542, 0.111208,
    );

    // Sun direction in ECI
    let sun_direction_eci = ECIVector { x: 0.0, y: 1.0, z: 0.0 };

    // Larger initial error to see energy dynamics clearly
    let initial_attitude = UnitQuaternion::new_normalize(Quaternion::new(-1.0, 4.0, -3.0, 2.0));

    // Non-zero initial angular velocity
    let initial_angular_velocity = BodyVector { x: 0.01, y: -0.005, z: 0.008 };

    // Simulation parameters
    let end_time = 15000.0;
    let dt = 0.1;
    let k_p = 0.001;
    let k_d = 0.1;

    println!("\n=== Energy-Dissipative MTQ Control Comparison ===\n");

    // Test 1: Pure Damping
    println!("Running PureDamping mode...");
    let (init_angle, final_angle_damping, energy_damping) = run_energy_tracking_sim(
        MtqControlMode::PureDamping,
        initial_attitude,
        initial_angular_velocity.clone(),
        sun_direction_eci.clone(),
        inertia,
        k_p,
        k_d,
        end_time,
        dt,
        "sp_mtq_pure_damping",
    )?;
    println!("  Initial angle: {:.2}°", init_angle);
    println!("  Final angle: {:.2}°", final_angle_damping);
    println!("  Energy change: {:.6} -> {:.6}", energy_damping.first().unwrap(), energy_damping.last().unwrap());

    // Test 2: Lyapunov Pointing
    println!("\nRunning LyapunovPointing mode...");
    let (_, final_angle_lyapunov, energy_lyapunov) = run_energy_tracking_sim(
        MtqControlMode::LyapunovPointing,
        initial_attitude,
        initial_angular_velocity.clone(),
        sun_direction_eci.clone(),
        inertia,
        k_p,
        k_d,
        end_time,
        dt,
        "sp_mtq_lyapunov",
    )?;
    println!("  Final angle: {:.2}°", final_angle_lyapunov);
    println!("  Energy change: {:.6} -> {:.6}", energy_lyapunov.first().unwrap(), energy_lyapunov.last().unwrap());

    // Test 3: Conditional Pointing
    println!("\nRunning ConditionalPointing mode...");
    let (_, final_angle_conditional, energy_conditional) = run_energy_tracking_sim(
        MtqControlMode::ConditionalPointing,
        initial_attitude,
        initial_angular_velocity.clone(),
        sun_direction_eci.clone(),
        inertia,
        k_p,
        k_d,
        end_time,
        dt,
        "sp_mtq_conditional",
    )?;
    println!("  Final angle: {:.2}°", final_angle_conditional);
    println!("  Energy change: {:.6} -> {:.6}", energy_conditional.first().unwrap(), energy_conditional.last().unwrap());

    // Test 4: Gain-Scheduled
    println!("\nRunning GainScheduled mode...");
    let (_, final_angle_scheduled, energy_scheduled) = run_energy_tracking_sim(
        MtqControlMode::GainScheduled,
        initial_attitude,
        initial_angular_velocity.clone(),
        sun_direction_eci.clone(),
        inertia,
        k_p,
        k_d,
        end_time,
        dt,
        "sp_mtq_scheduled",
    )?;
    println!("  Final angle: {:.2}°", final_angle_scheduled);
    println!("  Energy change: {:.6} -> {:.6}", energy_scheduled.first().unwrap(), energy_scheduled.last().unwrap());

    // Test 5: Cross-Product
    println!("\nRunning CrossProduct mode...");
    let (_, final_angle_crossproduct, energy_crossproduct) = run_energy_tracking_sim(
        MtqControlMode::CrossProduct,
        initial_attitude,
        initial_angular_velocity.clone(),
        sun_direction_eci.clone(),
        inertia,
        k_p,
        k_d,
        end_time,
        dt,
        "sp_mtq_crossproduct",
    )?;
    println!("  Final angle: {:.2}°", final_angle_crossproduct);
    println!("  Energy change: {:.6} -> {:.6}", energy_crossproduct.first().unwrap(), energy_crossproduct.last().unwrap());

    // Analyze energy behavior
    println!("\n=== Energy Analysis ===");

    fn analyze_energy(name: &str, energy: &[f64]) {
        let mut max_increase = 0.0_f64;
        let mut total_increase = 0.0_f64;
        let mut increase_count = 0;
        for i in 1..energy.len() {
            let de = energy[i] - energy[i - 1];
            if de > 0.0 {
                max_increase = max_increase.max(de);
                total_increase += de;
                increase_count += 1;
            }
        }
        let monotonic = max_increase < 1e-10;
        println!("{}: monotonic={}, max_increase={:.2e}, total_increase={:.2e}, increase_count={}",
                 name, monotonic, max_increase, total_increase, increase_count);
    }

    analyze_energy("PureDamping", &energy_damping);
    analyze_energy("LyapunovPointing", &energy_lyapunov);
    analyze_energy("ConditionalPointing", &energy_conditional);
    analyze_energy("GainScheduled", &energy_scheduled);
    analyze_energy("CrossProduct", &energy_crossproduct);

    // Create comparison plot script
    let plot_script = r#"#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    output_base = Path("out")

    modes = ["pure_damping", "lyapunov", "conditional", "scheduled", "crossproduct"]
    labels = ["Pure Damping", "Lyapunov Pointing", "Conditional", "Gain-Scheduled", "Cross-Product"]
    colors = ["blue", "red", "green", "purple", "orange"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for mode, label, color in zip(modes, labels, colors):
        csv_path = output_base / f"sp_mtq_{mode}" / "energy_tracking.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path, header=None, names=[
            "time", "angle_error_deg", "kinetic_energy", "potential_energy",
            "total_energy", "omega_norm", "H_x", "H_y", "H_z"
        ])

        # Angle error
        axes[0].plot(df["time"], df["angle_error_deg"], linewidth=1.5, alpha=0.8, label=label, color=color)

        # Total energy
        axes[1].plot(df["time"], df["total_energy"], linewidth=1.5, alpha=0.8, label=label, color=color)

        # Energy change (log scale)
        energy_change = df["total_energy"] - df["total_energy"].iloc[0]
        axes[2].plot(df["time"], energy_change, linewidth=1.5, alpha=0.8, label=label, color=color)

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Angle Error [deg]")
    axes[0].set_title("Sun Pointing Error Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Lyapunov Energy V")
    axes[1].set_title("Lyapunov Function V = (1/2)ω^T I ω + k_p(1-s·z)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Energy Change ΔV")
    axes[2].set_title("Energy Change from Initial")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle("MTQ Control Mode Comparison: Energy Dissipation", fontsize=14, y=0.995)
    plt.tight_layout()

    output_dir = output_base / "sp_mtq_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "energy_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
"#;

    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    fs::write(tests_dir.join("scripts/plot_energy_comparison.py"), plot_script)?;

    // Run plotting script
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_energy_comparison.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("\nPlot generated successfully");
        } else {
            println!("Warning: Plot generation failed");
        }
    }

    println!("\n=== Summary ===");
    println!("Initial angle error: {:.2}°", init_angle);
    println!("Final angle errors:");
    println!("  PureDamping:         {:.2}° (only kinetic energy dissipation)", final_angle_damping);
    println!("  LyapunovPointing:    {:.2}° (standard Lyapunov-based)", final_angle_lyapunov);
    println!("  ConditionalPointing: {:.2}° (falls back to damping if V̇ > 0)", final_angle_conditional);
    println!("  GainScheduled:       {:.2}° (reduces k_p near target)", final_angle_scheduled);
    println!("  CrossProduct:        {:.2}° (classical m ∝ B × e)", final_angle_crossproduct);

    // Determine best approach
    let results = [
        ("LyapunovPointing", final_angle_lyapunov),
        ("GainScheduled", final_angle_scheduled),
        ("CrossProduct", final_angle_crossproduct),
        ("ConditionalPointing", final_angle_conditional),
    ];
    let best = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("\nBest performing: {} ({:.2}°)", best.0, best.1);

    Ok(())
}

/// Test with small initial error - the problematic case
#[test]
fn sp_mtq_small_error_comparison() -> anyhow::Result<()> {
    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698,
        0.001303, 0.183472, 0.000542,
        0.000698, 0.000542, 0.111208,
    );

    // Sun direction in ECI
    let sun_direction_eci = ECIVector { x: 0.0, y: 1.0, z: 0.0 };

    // Small initial error (the problematic case)
    let initial_attitude = UnitQuaternion::new(Vector3::new(0.01, 0.01, 0.01))
        * c5a::estimation::triad_method(&Vector3::x(), &Vector3::z(), &Vector3::x(), &Vector3::y());

    // Small initial angular velocity
    let initial_angular_velocity = BodyVector { x: 0.001, y: -0.0005, z: 0.0008 };

    // Simulation parameters
    let end_time = 10000.0;
    let dt = 0.1;
    let k_p = 0.001;
    let k_d = 0.1;

    println!("\n=== Small Initial Error Test ===\n");

    // Test all modes
    let modes: Vec<(&str, MtqControlMode)> = vec![
        ("PureDamping", MtqControlMode::PureDamping),
        ("LyapunovPointing", MtqControlMode::LyapunovPointing),
        ("ConditionalPointing", MtqControlMode::ConditionalPointing),
        ("GainScheduled", MtqControlMode::GainScheduled),
        ("CrossProduct", MtqControlMode::CrossProduct),
        ("SlowControl", MtqControlMode::SlowControl),
        ("PeriodicAveraging", MtqControlMode::PeriodicAveraging { phase_accumulator: 0.0 }),
    ];

    let mut results = Vec::new();
    for (name, mode) in modes {
        let prefix = format!("sp_mtq_small_{}", name.to_lowercase().replace("-", "_"));
        let (init_angle, final_angle, energy) = run_energy_tracking_sim(
            mode,
            initial_attitude,
            initial_angular_velocity.clone(),
            sun_direction_eci.clone(),
            inertia,
            k_p,
            k_d,
            end_time,
            dt,
            &prefix,
        )?;

        // Check for energy amplification
        let initial_energy = *energy.first().unwrap();
        let final_energy = *energy.last().unwrap();
        let max_energy = energy.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let amplified = max_energy > initial_energy * 1.1; // 10% threshold

        println!("{}: {:.2}° -> {:.2}° | Energy: {:.2e} -> {:.2e} (max: {:.2e}) | Amplified: {}",
                 name, init_angle, final_angle, initial_energy, final_energy, max_energy, amplified);

        results.push((name, final_angle, amplified));
    }

    // Find best non-amplifying approach
    let best_stable = results.iter()
        .filter(|(_, _, amp)| !amp)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some((name, angle, _)) = best_stable {
        println!("\nBest stable (non-amplifying): {} ({:.2}°)", name, angle);
    } else {
        println!("\nNote: All approaches show temporary energy amplification.");
        println!("This is fundamental to MTQ-only control: torque ⊥ B means");
        println!("instantaneous energy control is impossible. Time-averaged");
        println!("dissipation matters more than instantaneous behavior.");

        // Rank by final pointing
        let mut sorted = results.clone();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        println!("\nRanked by final pointing accuracy:");
        for (i, (name, angle, _)) in sorted.iter().enumerate() {
            println!("  {}. {}: {:.2}°", i + 1, name, angle);
        }
    }

    Ok(())
}

/// Test focusing on peak decay behavior
#[test]
fn sp_mtq_peak_decay_analysis() -> anyhow::Result<()> {
    let inertia = Matrix3::new(
        0.168125, 0.001303, 0.000698,
        0.001303, 0.183472, 0.000542,
        0.000698, 0.000542, 0.111208,
    );

    let sun_direction_eci = ECIVector { x: 0.0, y: 1.0, z: 0.0 };

    // Use moderate initial error to see clear peak behavior
    let initial_attitude = UnitQuaternion::new(Vector3::new(0.5, 0.3, -0.2))
        * c5a::estimation::triad_method(&Vector3::x(), &Vector3::z(), &Vector3::x(), &Vector3::y());

    let initial_angular_velocity = BodyVector { x: 0.005, y: -0.003, z: 0.004 };

    // Longer simulation to see multiple peaks
    let end_time = 20000.0;
    let dt = 0.1;
    let k_p = 0.001;
    let k_d = 0.1;

    println!("\n=== Peak Decay Analysis ===\n");

    let modes: Vec<(&str, MtqControlMode)> = vec![
        ("LyapunovPointing", MtqControlMode::LyapunovPointing),
        ("ConditionalPointing", MtqControlMode::ConditionalPointing),
        ("GainScheduled", MtqControlMode::GainScheduled),
        ("CrossProduct", MtqControlMode::CrossProduct),
        ("SlowControl", MtqControlMode::SlowControl),
        ("PeriodicAveraging", MtqControlMode::PeriodicAveraging { phase_accumulator: 0.0 }),
    ];

    let mut analyses = Vec::new();

    for (name, mode) in &modes {
        let prefix = format!("sp_mtq_peak_{}", name.to_lowercase());
        let (_, final_angle, energy) = run_energy_tracking_sim(
            *mode,
            initial_attitude,
            initial_angular_velocity.clone(),
            sun_direction_eci.clone(),
            inertia,
            k_p,
            k_d,
            end_time,
            dt,
            &prefix,
        )?;

        let (peaks_times, peaks_values, peak_decay_rate) = analyze_peak_decay(&energy, dt);
        let envelope_decay_rate = analyze_envelope_decay(&energy, dt, 1000); // Sample every 100s
        let max_energy = energy.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let final_energy = *energy.last().unwrap_or(&0.0);

        let analysis = if peaks_values.len() >= 2 {
            PeakAnalysis {
                name: name.to_string(),
                final_angle,
                num_peaks: peaks_values.len(),
                first_peak: peaks_values[0],
                last_peak: *peaks_values.last().unwrap(),
                peak_decay_rate,
                envelope_decay_rate,
                peak_ratio: peaks_values.last().unwrap() / peaks_values[0],
                max_energy,
                final_energy,
                peaks_times: peaks_times.clone(),
                peaks_values: peaks_values.clone(),
            }
        } else {
            PeakAnalysis {
                name: name.to_string(),
                final_angle,
                num_peaks: peaks_values.len(),
                first_peak: max_energy,
                last_peak: final_energy,
                peak_decay_rate: 0.0,
                envelope_decay_rate,
                peak_ratio: if max_energy > 1e-10 { final_energy / max_energy } else { 1.0 },
                max_energy,
                final_energy,
                peaks_times,
                peaks_values,
            }
        };

        println!("{}: {} peaks, peak_decay={:.2e}/s, envelope_decay={:.2e}/s, final={:.2}°",
                 name, analysis.num_peaks, analysis.peak_decay_rate, analysis.envelope_decay_rate, final_angle);

        if analysis.num_peaks >= 2 {
            println!("  Peaks: first={:.2e}, last={:.2e}", analysis.first_peak, analysis.last_peak);
            print!("  Peak values: ");
            for (i, (t, v)) in analysis.peaks_times.iter().zip(analysis.peaks_values.iter()).take(5).enumerate() {
                if i > 0 { print!(", "); }
                print!("t={:.0}s:{:.2e}", t, v);
            }
            if analysis.num_peaks > 5 {
                print!(", ... ({} more)", analysis.num_peaks - 5);
            }
            println!();
        }
        println!();

        analyses.push(analysis);
    }

    // Create peak decay plot script
    let plot_script = r#"#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_envelope(energy, window=1000):
    """Compute running maximum envelope using numpy"""
    n = len(energy)
    envelope = np.zeros(n)
    half_w = window // 2
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        envelope[i] = np.max(energy[start:end])
    return envelope

def find_peaks(energy, window=20):
    """Find local maxima with prominence filtering"""
    peaks_idx = []
    peaks_val = []

    for i in range(window, len(energy) - window):
        center = energy[i]
        is_peak = all(energy[i-j] <= center for j in range(1, window+1))
        is_peak = is_peak and all(energy[i+j] <= center for j in range(1, window+1))

        if is_peak:
            left_min = min(energy[i-window:i])
            right_min = min(energy[i+1:i+window+1])
            prominence = center - max(left_min, right_min)
            threshold = max(center * 0.005, 1e-8)

            if prominence > threshold:
                peaks_idx.append(i)
                peaks_val.append(center)

    return peaks_idx, peaks_val

def main():
    output_base = Path("out")

    modes = ["lyapunovpointing", "conditionalpointing", "gainscheduled",
             "crossproduct", "slowcontrol", "periodicaveraging"]
    labels = ["Lyapunov", "Conditional", "Gain-Scheduled",
              "Cross-Product", "Slow Control", "Periodic Avg"]
    colors = ["red", "blue", "purple", "orange", "green", "brown"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 16))

    dt = 0.1
    envelope_data = {}

    for mode, label, color in zip(modes, labels, colors):
        csv_path = output_base / f"sp_mtq_peak_{mode}" / "energy_tracking.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path, header=None, names=[
            "time", "angle_error_deg", "kinetic_energy", "potential_energy",
            "total_energy", "omega_norm", "H_x", "H_y", "H_z"
        ])

        energy = df["total_energy"].values
        time = df["time"].values

        # Plot energy
        axes[0].plot(time, energy, linewidth=0.8, alpha=0.6, label=label, color=color)

        # Compute and plot envelope
        envelope = compute_envelope(energy, window=1000)  # 100 sec window
        axes[1].semilogy(time, envelope, linewidth=1.5, alpha=0.8, label=label, color=color)
        envelope_data[label] = (time, envelope)

        # Plot angle error
        axes[2].plot(time, df["angle_error_deg"], linewidth=1, alpha=0.7, label=label, color=color)

        # Plot kinetic energy (shows oscillation behavior)
        axes[3].plot(time, df["kinetic_energy"], linewidth=0.8, alpha=0.6, label=label, color=color)

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Total Energy V")
    axes[0].set_title("Lyapunov Energy V = (1/2)ω^T I ω + k_p(1-s·z)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Energy Envelope (log scale)")
    axes[1].set_title("Energy Envelope Decay (running max over 100s window)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Angle Error [deg]")
    axes[2].set_title("Sun Pointing Error")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Kinetic Energy")
    axes[3].set_title("Kinetic Energy (1/2)ω^T I ω")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    plt.suptitle("MTQ Control: Energy Envelope Decay Analysis", fontsize=14, y=0.995)
    plt.tight_layout()

    output_dir = output_base / "sp_mtq_peak_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "peak_decay.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Print decay analysis
    print("\n=== Envelope Decay Summary ===")
    for label, (times, env) in envelope_data.items():
        # Sample at regular intervals
        sample_idx = np.arange(0, len(env), 1000)
        sample_times = times[sample_idx]
        sample_env = env[sample_idx]
        valid = sample_env > 1e-10

        if np.sum(valid) >= 2:
            log_env = np.log(sample_env[valid])
            t_valid = sample_times[valid]
            slope, _ = np.polyfit(t_valid, log_env, 1)
            decay_rate = -slope
            reduction = (1 - sample_env[-1]/sample_env[0]) * 100
            print(f"{label}: decay_rate={decay_rate:.2e}/s, reduction={reduction:.1f}%")

if __name__ == "__main__":
    main()
"#;

    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    fs::write(tests_dir.join("scripts/plot_peak_decay.py"), plot_script)?;

    // Run plotting script
    let plot_result = std::process::Command::new("uv")
        .args(["run", "python", "scripts/plot_peak_decay.py"])
        .current_dir(&tests_dir)
        .status();
    if let Ok(status) = plot_result {
        if status.success() {
            println!("Plot generated successfully");
        }
    }

    // Summary: rank by envelope decay (more robust than peak decay)
    println!("\n=== Summary: Ranked by Envelope Decay Rate ===");
    println!("(Higher decay rate = energy envelope decreases faster = better stability)\n");

    let mut sorted = analyses.clone();
    sorted.sort_by(|a, b| b.envelope_decay_rate.partial_cmp(&a.envelope_decay_rate).unwrap());

    println!("{:<20} {:>14} {:>14} {:>12} {:>10} {:>10}",
             "Mode", "Envelope Decay", "Peak Decay", "Max/Final", "Peaks", "Final°");
    println!("{}", "-".repeat(85));
    for (i, a) in sorted.iter().enumerate() {
        let indicator = if a.envelope_decay_rate > 0.0 { "✓" } else { "✗" };
        let energy_ratio = if a.max_energy > 1e-10 { a.final_energy / a.max_energy } else { 1.0 };
        println!("{} {:<18} {:>12.2e}/s {:>12.2e}/s {:>12.4} {:>8} {:>10.2}°",
                 indicator, a.name, a.envelope_decay_rate, a.peak_decay_rate,
                 energy_ratio, a.num_peaks, a.final_angle);
        if i == 0 {
            println!("   ↑ BEST envelope decay (most stable energy reduction)");
        }
    }

    println!("\n=== Energy Reduction Summary ===");
    for a in &sorted {
        let reduction_pct = if a.max_energy > 1e-10 {
            (1.0 - a.final_energy / a.max_energy) * 100.0
        } else {
            0.0
        };
        println!("{:<20}: max={:.2e} -> final={:.2e} ({:.1}% reduction)",
                 a.name, a.max_energy, a.final_energy, reduction_pct);
    }

    println!("\n=== Interpretation ===");
    println!("- Envelope decay > 0: Energy envelope is decreasing over time (stable)");
    println!("- Max/Final ratio < 1: Final energy is lower than peak (converging)");
    println!("- For energy-stable control, prefer modes with:");
    println!("  1. High envelope decay rate");
    println!("  2. Low final/max energy ratio");
    println!("  3. Good final pointing accuracy");

    Ok(())
}

// ============================================================================
// Summary of MTQ Energy-Dissipative Control Investigation
// ============================================================================
//
// PROBLEM:
// With small initial pointing error, the standard Lyapunov-based MTQ controller
// can amplify system energy, causing the pointing error to increase temporarily.
//
// ROOT CAUSE:
// MTQ can only produce torque perpendicular to the magnetic field B:
//   τ = m × B = τ_desired - (τ_desired · b̂)b̂
//
// This constraint means that even with a Lyapunov-based controller that would
// guarantee V̇ ≤ 0 for ideal torque, the actual torque may not satisfy this.
//
// LYAPUNOV ANALYSIS:
// V = (1/2)ω^T I ω + k_p(1 - s·z)
// V̇ = ω·τ - k_p ω·(s × z)
//
// For ideal torque τ_d = k_p(s×z) - k_d I ω:
//   V̇ = -k_d ω·I·ω ≤ 0 (guaranteed)
//
// For MTQ torque τ = τ_d - (τ_d·b̂)b̂:
//   V̇ = -k_d ω·I·ω - (τ_d·b̂)(ω·b̂)
//   The term -(τ_d·b̂)(ω·b̂) can be positive or negative.
//
// FINDINGS:
//
// 1. Large Initial Error (131.81°):
//    - CrossProduct: 1.75° (BEST pointing)
//    - LyapunovPointing: 3.93°
//    - GainScheduled: 8.51°
//    - ConditionalPointing: 21.34°
//    - PureDamping: 142.39° (no pointing control)
//
// 2. Small Initial Error (0.81°):
//    - ConditionalPointing: 0.01° (BEST pointing, but highest energy amp)
//    - LyapunovPointing: 7.65°
//    - PeriodicAveraging: 8.45°
//    - GainScheduled: 18.37° (lowest energy amplification)
//    - PureDamping: 90.39° (drifts without pointing control)
//
// KEY INSIGHTS:
//
// 1. Temporary energy amplification is UNAVOIDABLE with MTQ-only control
//    because the torque constraint τ ⊥ B prevents instantaneous energy control.
//
// 2. Time-averaged energy dissipation is what matters. As B rotates, all
//    angular momentum components can eventually be removed.
//
// 3. For large errors: CrossProduct and LyapunovPointing work well.
//    For small errors: ConditionalPointing achieves best final accuracy.
//
// 4. Trade-off exists between pointing accuracy and energy smoothness:
//    - ConditionalPointing: best pointing, highest energy peaks
//    - GainScheduled: moderate pointing, lowest energy peaks
//
// RECOMMENDATIONS:
//
// 1. For general use: LyapunovPointing or CrossProduct (good balance)
//
// 2. For energy-sensitive applications: GainScheduled (lowest peak energy)
//
// 3. For precision pointing: ConditionalPointing (best final accuracy)
//
// 4. Accept that temporary energy amplification is normal for MTQ control.
//    The system will converge as long as the time-averaged energy rate is negative.
// 