/*
 * NAYAK HAL — Hardware Abstraction in Rust
 *
 * Real GPIO, camera, and sensor control at hard real-time speeds.
 * This module is compiled into the nayak_spinal binary and runs
 * inside the 1000 Hz control loop, completely independent of the
 * Python GIL and garbage collector.
 *
 * Auto-detection:
 *   On a Raspberry Pi  → /sys/class/gpio exists → HardwareMode::Real
 *                         Reads/writes go through sysfs.
 *   On a dev machine   → /sys/class/gpio absent → HardwareMode::Simulated
 *                         Shadow registers only, zero errors.
 *
 * The rule: real hardware detected = use it immediately.
 *           No real hardware = simulate perfectly, zero errors.
 *           Never crash on missing hardware.
 */

use std::path::Path;

/// Operating mode of the hardware abstraction layer.
///
/// Determined automatically at `GpioBus::new()` time by probing for
/// the Linux sysfs GPIO interface at `/sys/class/gpio`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareMode {
    /// Running on real hardware — reads/writes go through sysfs GPIO.
    Real,
    /// Running on a development machine — shadow registers only.
    Simulated,
}

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
/// `GpioPin` owns the pin number, its mode, its last known logical value, and
/// the hardware mode (real or simulated).
///
/// In **Real** mode, `write` and `read` use the Linux sysfs GPIO interface
/// (`/sys/class/gpio/gpioN/value`).  In **Simulated** mode they operate on
/// an in-process shadow register so the spinal cord can be developed and tested
/// on any machine.
pub struct GpioPin {
    /// BCM GPIO pin number (e.g. 17, 27, 22 …).
    pub pin: u8,
    /// Current direction of the pin.
    pub mode: PinMode,
    /// Shadow register: last value written (Output) or read (Input).
    pub value: u8,
    /// Whether this pin operates on real hardware or in simulation.
    pub hw_mode: HardwareMode,
}

impl GpioPin {
    /// Create a new `GpioPin` descriptor.
    ///
    /// # Arguments
    /// * `pin`     – BCM pin number.
    /// * `mode`    – Initial direction (`Input` or `Output`).
    /// * `hw_mode` – `Real` if sysfs GPIO is available, `Simulated` otherwise.
    pub fn new(pin: u8, mode: PinMode, hw_mode: HardwareMode) -> Self {
        Self {
            pin,
            mode,
            value: 0,
            hw_mode,
        }
    }

    /// Write a logical value to the pin.
    ///
    /// **Real mode**: writes to `/sys/class/gpio/gpio{pin}/value` via sysfs.
    /// **Simulated mode**: updates the internal shadow register only.
    ///
    /// Returns `true` on success, `false` if the pin is configured as Input
    /// or the sysfs write fails.
    pub fn write(&mut self, value: u8) -> bool {
        if self.mode == PinMode::Input {
            eprintln!(
                "HAL: attempted write to input pin {} — ignored",
                self.pin
            );
            return false;
        }

        match self.hw_mode {
            HardwareMode::Real => {
                let path = format!("/sys/class/gpio/gpio{}/value", self.pin);
                let val_str = if value != 0 { "1" } else { "0" };
                match std::fs::write(&path, val_str) {
                    Ok(()) => {
                        self.value = value;
                        true
                    }
                    Err(e) => {
                        eprintln!(
                            "HAL [REAL]: failed to write pin {} via '{}': {}",
                            self.pin, path, e
                        );
                        // Fall back to shadow register so caller can continue.
                        self.value = value;
                        false
                    }
                }
            }
            HardwareMode::Simulated => {
                self.value = value;
                true
            }
        }
    }

    /// Read the current logical value from the pin.
    ///
    /// **Real mode**: reads from `/sys/class/gpio/gpio{pin}/value` via sysfs,
    /// parsing `"0\n"` or `"1\n"` to `u8`.
    /// **Simulated mode**: returns the shadow register value.
    pub fn read(&self) -> u8 {
        match self.hw_mode {
            HardwareMode::Real => {
                let path = format!("/sys/class/gpio/gpio{}/value", self.pin);
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        let trimmed = content.trim();
                        match trimmed.parse::<u8>() {
                            Ok(v) => v,
                            Err(_) => {
                                eprintln!(
                                    "HAL [REAL]: unexpected value '{}' from pin {}",
                                    trimmed, self.pin
                                );
                                self.value
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "HAL [REAL]: failed to read pin {} via '{}': {}",
                            self.pin, path, e
                        );
                        self.value
                    }
                }
            }
            HardwareMode::Simulated => self.value,
        }
    }
}

/// A bank of GPIO pins managed as a group (e.g. a motor controller).
///
/// `GpioBus` owns the hardware mode for the entire bus and propagates it
/// to every pin.  The mode is auto-detected at construction time by checking
/// for the existence of `/sys/class/gpio`.
pub struct GpioBus {
    /// The GPIO pins in this bank.
    pub pins: Vec<GpioPin>,
    /// Operating mode for the entire bus.
    pub mode: HardwareMode,
}

impl GpioBus {
    /// Create a bus from a list of (pin, mode) pairs.
    ///
    /// Auto-detects hardware mode by probing for `/sys/class/gpio`:
    /// * Path exists → `HardwareMode::Real`
    /// * Path absent → `HardwareMode::Simulated`
    ///
    /// All pins in the bus inherit the detected mode.
    pub fn new(pins: Vec<(u8, PinMode)>) -> Self {
        let hw_mode = if Path::new("/sys/class/gpio").exists() {
            eprintln!("HAL: /sys/class/gpio detected — REAL hardware mode");
            HardwareMode::Real
        } else {
            eprintln!("HAL: /sys/class/gpio not found — SIMULATED mode");
            HardwareMode::Simulated
        };

        Self {
            pins: pins
                .into_iter()
                .map(|(p, m)| GpioPin::new(p, m, hw_mode))
                .collect(),
            mode: hw_mode,
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

    /// Export a GPIO pin via sysfs so it becomes accessible for read/write.
    ///
    /// Writes the pin number to `/sys/class/gpio/export` and then sets the
    /// direction to `"out"` (for Output pins) or `"in"` (for Input pins).
    ///
    /// **Only effective in Real mode.**  In Simulated mode this is a no-op
    /// that always returns `true`.
    ///
    /// # Arguments
    /// * `pin`       – BCM pin number to export.
    /// * `direction` – The desired pin direction.
    ///
    /// # Returns
    /// `true` on success or in simulated mode, `false` on sysfs write failure.
    pub fn export_pin(&self, pin: u8, direction: PinMode) -> bool {
        if self.mode == HardwareMode::Simulated {
            return true;
        }

        // Export the pin.
        let export_path = "/sys/class/gpio/export";
        if let Err(e) = std::fs::write(export_path, pin.to_string()) {
            // EBUSY (errno 16) means the pin is already exported — that's OK.
            let msg = e.to_string();
            if !msg.contains("Device or resource busy") {
                eprintln!(
                    "HAL [REAL]: failed to export pin {} via '{}': {}",
                    pin, export_path, e
                );
                return false;
            }
        }

        // Set direction.
        let dir_path = format!("/sys/class/gpio/gpio{}/direction", pin);
        let dir_str = match direction {
            PinMode::Output => "out",
            PinMode::Input => "in",
        };
        if let Err(e) = std::fs::write(&dir_path, dir_str) {
            eprintln!(
                "HAL [REAL]: failed to set direction for pin {} via '{}': {}",
                pin, dir_path, e
            );
            return false;
        }

        eprintln!(
            "HAL [REAL]: pin {} exported with direction '{}'",
            pin, dir_str
        );
        true
    }

    /// Read a specific pin by its BCM number.
    ///
    /// Searches the bus for a pin matching `pin_number` and returns its value.
    /// Returns `None` if the pin is not in this bus.
    pub fn read_pin(&self, pin_number: u8) -> Option<u8> {
        self.pins
            .iter()
            .find(|p| p.pin == pin_number)
            .map(|p| p.read())
    }

    /// Write to a specific pin by its BCM number.
    ///
    /// Searches the bus for a pin matching `pin_number` and writes `value`.
    /// Returns `false` if the pin is not found or the write fails.
    pub fn write_pin(&mut self, pin_number: u8, value: u8) -> bool {
        if let Some(pin) = self.pins.iter_mut().find(|p| p.pin == pin_number) {
            pin.write(value)
        } else {
            eprintln!("HAL: pin {} not found in bus", pin_number);
            false
        }
    }
}
