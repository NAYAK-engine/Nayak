/*
 * NAYAK HAL — Hardware Abstraction in Rust
 *
 * Real GPIO, camera, and sensor control at hard real-time speeds.
 * This module is compiled into the nayak_spinal binary and runs
 * inside the 1000 Hz control loop, completely independent of the
 * Python GIL and garbage collector.
 *
 * On a Raspberry Pi: replace the stub read/write bodies with real
 * RPi.GPIO-equivalent calls via the `rppal` crate.
 * On x86 development machines: the stubs compile cleanly and return
 * simulated values so the rest of the spinal cord can be tested.
 */

/// Direction of a GPIO pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinMode {
    /// Pin is configured as a digital input (reads external signal).
    Input,
    /// Pin is configured as a digital output (drives external load).
    Output,
}

/// A single GPIO pin managed by the NAYAK HAL.
///
/// `GpioPin` owns the pin number, its mode, and its last known logical value.
/// On a real Raspberry Pi, `write` and `read` would delegate to the
/// `rppal::gpio` crate; here they operate on an in-process shadow register
/// so the spinal cord can be developed and tested on any machine.
pub struct GpioPin {
    /// BCM GPIO pin number (e.g. 17, 27, 22 …).
    pub pin: u8,
    /// Current direction of the pin.
    pub mode: PinMode,
    /// Shadow register: last value written (Output) or read (Input).
    pub value: u8,
}

impl GpioPin {
    /// Create a new `GpioPin` descriptor.
    ///
    /// # Arguments
    /// * `pin`  – BCM pin number.
    /// * `mode` – Initial direction (`Input` or `Output`).
    pub fn new(pin: u8, mode: PinMode) -> Self {
        Self { pin, mode, value: 0 }
    }

    /// Write a logical value to the pin.
    ///
    /// On a real Pi this would drive the GPIO line high (`1`) or low (`0`).
    /// In simulation it updates the internal shadow register.
    ///
    /// Returns `true` on success, `false` if the pin is configured as Input.
    pub fn write(&mut self, value: u8) -> bool {
        if self.mode == PinMode::Input {
            eprintln!(
                "HAL: attempted write to input pin {} — ignored",
                self.pin
            );
            return false;
        }
        self.value = value;
        // TODO (Pi deployment): call rppal::gpio::OutputPin::write() here
        true
    }

    /// Read the current logical value from the pin.
    ///
    /// On a real Pi this samples the GPIO line.
    /// In simulation it returns the last written value.
    pub fn read(&self) -> u8 {
        // TODO (Pi deployment): call rppal::gpio::InputPin::read() here
        self.value
    }
}

/// A simple bank of GPIO pins managed as a group (e.g. a motor controller).
pub struct GpioBus {
    pub pins: Vec<GpioPin>,
}

impl GpioBus {
    /// Create a bus from a list of (pin, mode) pairs.
    pub fn new(pins: Vec<(u8, PinMode)>) -> Self {
        Self {
            pins: pins.into_iter().map(|(p, m)| GpioPin::new(p, m)).collect(),
        }
    }

    /// Write `value` to every Output pin in the bank simultaneously.
    pub fn write_all(&mut self, value: u8) {
        for pin in &mut self.pins {
            if pin.mode == PinMode::Output {
                pin.write(value);
            }
        }
    }
}
