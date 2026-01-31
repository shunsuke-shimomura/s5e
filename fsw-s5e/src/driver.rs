use crate::data::{self};

pub mod actuator;
pub mod sensor;

pub struct CommandReceiver {
    pub s4e_port: s4e_port::S4ESubscribePort<s4e_port::GSCommandData>,
}

impl Default for CommandReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandReceiver {
    pub fn new() -> Self {
        Self {
            s4e_port: s4e_port::S4ESubscribePort::new(),
        }
    }

    pub fn input(&mut self, cmd_port: &s4e_port::S4EPublishPort<s4e_port::GSCommandData>) {
        s4e_port::transfer(cmd_port, &mut self.s4e_port);
    }

    pub fn main_loop(&mut self) -> Option<data::ControllerCommand> {
        self.s4e_port
            .subscribe()
            .map(|cmd_data| match cmd_data.command {
                s4e_port::Command::ControllerCommand(cmd) => match cmd {
                    s4e_port::ControllerCommand::RWControlTransition => {
                        data::ControllerCommand::RWControlTransition
                    }
                    s4e_port::ControllerCommand::ThreeAxisControlTransition(target_quaternion) => {
                        data::ControllerCommand::ThreeAxisControlTransition(target_quaternion)
                    }
                    s4e_port::ControllerCommand::SunPointingControlTransition => {
                        data::ControllerCommand::SunPointingControlTransition
                    }
                },
            })
    }
}
