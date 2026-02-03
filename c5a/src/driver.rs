use crate::data::{self};

pub mod actuator;
pub mod sensor;

pub struct CommandReceiver {
    pub sim_port: s5e_port::S5ESubscribePort<s5e_port::GSCommandData>,
}

impl Default for CommandReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandReceiver {
    pub fn new() -> Self {
        Self {
            sim_port: s5e_port::S5ESubscribePort::new(),
        }
    }

    pub fn input(&mut self, cmd_port: &s5e_port::S5EPublishPort<s5e_port::GSCommandData>) {
        s5e_port::transfer(cmd_port, &mut self.sim_port);
    }

    pub fn main_loop(&mut self) -> Option<data::ControllerCommand> {
        self.sim_port
            .subscribe()
            .map(|cmd_data| match cmd_data.command {
                s5e_port::Command::ControllerCommand(cmd) => match cmd {
                    s5e_port::ControllerCommand::RWControlTransition => {
                        data::ControllerCommand::RWControlTransition
                    }
                    s5e_port::ControllerCommand::ThreeAxisControlTransition(target_quaternion) => {
                        data::ControllerCommand::ThreeAxisControlTransition(target_quaternion)
                    }
                    s5e_port::ControllerCommand::SunPointingControlTransition => {
                        data::ControllerCommand::SunPointingControlTransition
                    }
                },
            })
    }
}
