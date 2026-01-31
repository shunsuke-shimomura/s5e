use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

use astrodynamics::coordinate::BodyVector;
use chrono::{NaiveDateTime, TimeDelta};
use clap::Parser;
use control_system::integrator::Prediction;
use csv::Writer;
use fsw_s5e::constants;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use s5e::S5ETriggerEvent;
use s5e_port::{Command, ControllerCommand, GSCommandData, S5EPublishPort};

#[derive(Parser, Debug)]
#[command(name = "sils-s5e")]
#[command(about = "s5e Satellite Simulation", long_about = None)]
struct Args {
    /// Enable sensor data debug output
    #[arg(long)]
    debug_sensor: bool,

    /// Enable actuator data debug output
    #[arg(long)]
    debug_actuator: bool,

    /// Enable all debug outputs
    #[arg(long)]
    debug_all: bool,
}

const TARGET_QUATERNION: Quaternion<f64> = Quaternion::new(1.0, 2.0, 3.0, 4.0);

fn main() {
    let args = Args::parse();

    // Initialize debug configuration
    let debug_config = debug_s5e::DebugConfig {
        sensor_data: args.debug_sensor || args.debug_all,
        actuator_data: args.debug_actuator || args.debug_all,
    };
    debug_s5e::init_debug_config(debug_config);

    if args.debug_sensor || args.debug_actuator || args.debug_all {
        println!("Debug configuration:");
        if debug_config.sensor_data {
            println!("  Sensor data: enabled");
        }
        if debug_config.actuator_data {
            println!("  Actuator data: enabled");
        }
        println!();
    }

    let mut time = 0.0;
    let mut datetime =
        NaiveDateTime::parse_from_str("2020-04-01 12:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let end_time = 28000.0;
    let dt = 0.1;
    let rng = Rc::new(RefCell::new(rand::thread_rng()));

    let tle = s5e::orbit::TLE::new(
        "1 25544U 98067A   20076.51604214  .00016717  00000-0  10270-3 0  9005".to_string(),
        "2 25544  51.6412  86.9962 0006063  30.9353 329.2153 15.49228202 17647".to_string(),
    );
    let mut sun = s5e::sun::Sun::new(datetime);
    let mut earth = s5e::earth::Earth::new(datetime);
    let mut moon = s5e::moon::Moon::new(datetime);

    let initial_sun_dir = {
        let sun_position = sun.position.get_now();
        control_system::components::Direction::from_dir(
            nalgebra::UnitVector3::new_normalize(sun_position.into())
        )
    };
    let initial_attitude = fsw_s5e::estimation::triad_method(&Vector3::new(1.0, 0.0, 0.0), &Vector3::new(0.0, 0.0, 1.0), &initial_sun_dir.basis_2d().column(0).into_owned(), &initial_sun_dir.dir().into_inner());

    let mut sat = s5e::SpaceCraft::new(
        &tle,
        datetime,
        rng,
        // Some(UnitQuaternion::identity()),
        Some(initial_attitude),
        Some(constants::SATELLITE_INERTIA),
        Some(BodyVector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }),
        true,
    );

    let mut cmd_port = s5e_port::S5EPublishPort::<s5e_port::GSCommandData>::new();

    let mut attitude_writer = Writer::from_path("out/attitude.csv").unwrap();
    let mut direction_writer = Writer::from_path("out/direction.csv").unwrap();
    println!(
        "Target quaternion: {:?}",
        UnitQuaternion::new_normalize(TARGET_QUATERNION).quaternion()
    );

    let mut three_axis_transition_event_mtq =
        S5ETriggerEvent::new(|cmd_port: &mut S5EPublishPort<GSCommandData>| {
            cmd_port.publish(GSCommandData {
                command: Command::ControllerCommand(ControllerCommand::ThreeAxisControlTransition(
                    UnitQuaternion::new_normalize(TARGET_QUATERNION),
                )),
            })
        });

    let mut rw_mode_transition_event =
        S5ETriggerEvent::new(|cmd_port: &mut S5EPublishPort<GSCommandData>| {
            cmd_port.publish(GSCommandData {
                command: Command::ControllerCommand(ControllerCommand::RWControlTransition),
            })
        });

    let mut three_axis_transition_event_rw = three_axis_transition_event_mtq.clone();

    let mut prof_env_tick = Duration::ZERO;
    let mut prof_sat_tick = Duration::ZERO;
    let mut prof_csv_write = Duration::ZERO;
    let mut prof_clear = Duration::ZERO;
    let mut prof_event_check = Duration::ZERO;
    let sim_start = Instant::now();

    while time < end_time {
        // tick earth/moon/sun
        let t0 = Instant::now();
        earth.tick(dt, datetime);
        moon.tick(dt, datetime);
        sun.tick(dt, datetime);
        prof_env_tick += t0.elapsed();

        let t0 = Instant::now();
        // if let fsw_s5e::controller::AttitudeControllMode::MTQ(mtq) =
        //     &sat.obc.fsw.attitude_controller.mode
        //     && matches!(
        //         mtq.mode,
        //         fsw_s5e::controller::mtq::MTQControlMode::SunPointing(_)
        //     )
        // {
        //     three_axis_transition_event_mtq.trigger(&mut cmd_port);
        // }
        prof_event_check += t0.elapsed();

        // if let fsw_s5e::controller::AttitudeControllMode::MTQ(mtq) =
        //     &sat.obc.fsw.attitude_controller.mode
        //     && matches!(
        //         mtq.mode,
        //         fsw_s5e::controller::mtq::MTQControlMode::SunPointing(_)
        //     )
        //     && (sat.attitude.get_now() * TARGET_QUATERNION)
        //         .conjugate()
        //         .norm()
        //         < 0.01
        // {
        //     rw_mode_transition_event.trigger(&mut cmd_port);
        // }

        // if let fsw_s5e::controller::AttitudeControllMode::RW(rw) =
        //     &sat.obc.fsw.attitude_controller.mode
        //     && matches!(
        //         rw.mode,
        //         fsw_s5e::controller::rw::RWControlMode::SunPointing(_)
        //     )
        // {
        //     three_axis_transition_event_rw.trigger(&mut cmd_port);
        // }

        // tick satellite
        let t0 = Instant::now();
        sat.tick(
            dt,
            datetime,
            sun.light_source(),
            vec![earth.shadow_source(), moon.shadow_source()],
            &cmd_port,
        );
        prof_sat_tick += t0.elapsed();

        // output log
        let t0 = Instant::now();
        attitude_writer
            .write_record(&[
                time.to_string(),
                sat.attitude.get_now().scalar().to_string(),
                sat.attitude.get_now().coords.x.to_string(),
                sat.attitude.get_now().coords.y.to_string(),
                sat.attitude.get_now().coords.z.to_string(),
                sat.angular_velocity.get_now().x.to_string(),
                sat.angular_velocity.get_now().y.to_string(),
                sat.angular_velocity.get_now().z.to_string(),
            ])
            .unwrap();
        attitude_writer.flush().unwrap();

        let sun_dir = Vector3::from(BodyVector::from_eci(
            sun.position.get_now() - sat.eci_position.get_now(),
            UnitQuaternion::new_normalize(sat.attitude.get_now()),
        ))
        .normalize();
        let mag_field = BodyVector::from_eci(
            sat.magnetic_field.clone(),
            UnitQuaternion::new_normalize(sat.attitude.get_now()),
        );

        direction_writer
            .write_record(&[
                time.to_string(),
                sun_dir.x.to_string(),
                sun_dir.y.to_string(),
                sun_dir.z.to_string(),
                mag_field.x.to_string(),
                mag_field.y.to_string(),
                mag_field.z.to_string(),
            ])
            .unwrap();
        prof_csv_write += t0.elapsed();

        // update time
        time += dt;
        datetime += TimeDelta::nanoseconds((dt * 1.0e9) as i64);

        // Clear states for next iteration
        let t0 = Instant::now();
        sat.clear_state();
        sun.clear_state();
        earth.clear_state();
        moon.clear_state();
        cmd_port.clear();
        prof_clear += t0.elapsed();
    }

    let sim_total = sim_start.elapsed();
    println!();
    println!(
        "=== Main Loop Profile (total {:.3}s) ===",
        sim_total.as_secs_f64()
    );
    let pct = |d: Duration| d.as_secs_f64() / sim_total.as_secs_f64() * 100.0;
    println!(
        "  env_tick (earth/moon/sun): {:>8.3}ms ({:>5.1}%)",
        prof_env_tick.as_secs_f64() * 1000.0,
        pct(prof_env_tick)
    );
    println!(
        "  event_check:              {:>8.3}ms ({:>5.1}%)",
        prof_event_check.as_secs_f64() * 1000.0,
        pct(prof_event_check)
    );
    println!(
        "  sat.tick:                 {:>8.3}ms ({:>5.1}%)",
        prof_sat_tick.as_secs_f64() * 1000.0,
        pct(prof_sat_tick)
    );
    println!(
        "  csv_write:                {:>8.3}ms ({:>5.1}%)",
        prof_csv_write.as_secs_f64() * 1000.0,
        pct(prof_csv_write)
    );
    println!(
        "  clear_state:              {:>8.3}ms ({:>5.1}%)",
        prof_clear.as_secs_f64() * 1000.0,
        pct(prof_clear)
    );
    println!();

    sat.profile.print_summary();
    println!();
    sat.obc.fsw.profile.print_summary();
    println!();
    sat.obc.fsw.direction_estimator.profile.print_summary();
    println!();
    sat.obc.fsw.attitude_determination.profile.print_summary();
}
