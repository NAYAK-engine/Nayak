/*
 * NAYAK Spinal Cord — Hard Real-Time Runtime
 *
 * This is the Rust layer of NAYAK.
 * Runs independently of the Python brain.
 * Handles Layer 1 (HAL) and Layer 4 (Action) at 1000 Hz+.
 * Communicates with the Python brain via JSON over stdin/stdout.
 *
 * Architecture:
 *   Python Brain (Layer 3)  ←→  JSON IPC  ←→  Rust Spinal Cord (Layer 1+4)
 *
 * Data flow:
 *   Commands  : Python writes a JSON line to the spinal cord's stdin.
 *   Responses : Rust writes a JSON line to stdout; Python reads it.
 *   Heartbeat : Every 1 000 control-loop ticks (≈ 1 second) a heartbeat
 *               JSON line is written to stdout.
 *   Stderr    : Diagnostic / error messages only — never parsed by Python.
 *
 * JSON command schema  (Python → Rust):
 * {
 *   "command_type": "gpio_write" | "gpio_read" | "ping" | ...,
 *   "device_id":    "<string>",
 *   "payload":      { … },
 *   "timestamp_ms": <u64 unix ms>
 * }
 *
 * JSON response schema (Rust → Python):
 * {
 *   "command_type": "<echoed>",
 *   "device_id":    "<echoed>",
 *   "success":      true | false,
 *   "data":         { … },
 *   "timestamp_ms": <u64>,
 *   "latency_us":   <u64 microseconds to parse + dispatch command>
 * }
 */

mod hal;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time;

// ─────────────────────────────────────────────────────────────────────────────
// IPC types
// ─────────────────────────────────────────────────────────────────────────────

/// A command sent from the Python brain to the Rust spinal cord.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SpinalCommand {
    /// Opcode string (e.g. `"gpio_write"`, `"gpio_read"`, `"ping"`).
    command_type: String,
    /// Target device identifier (e.g. `"gpio.17"`, `"motor.left"`).
    device_id: String,
    /// Arbitrary JSON payload specific to the command type.
    payload: serde_json::Value,
    /// Unix timestamp in milliseconds at the time of dispatch.
    timestamp_ms: u64,
}

/// A response sent from the Rust spinal cord back to the Python brain.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SpinalResponse {
    /// Echoed command type for correlation.
    command_type: String,
    /// Echoed device identifier.
    device_id: String,
    /// Whether the command was processed successfully.
    success: bool,
    /// Result data (device reading, ack, etc.).
    data: serde_json::Value,
    /// Unix timestamp in milliseconds at the time of the response.
    timestamp_ms: u64,
    /// Microseconds elapsed between receiving the command line and emitting
    /// this response — useful for latency monitoring.
    latency_us: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Current Unix time in milliseconds.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─────────────────────────────────────────────────────────────────────────────
// Control loop  (1 000 Hz independent ticker)
// ─────────────────────────────────────────────────────────────────────────────

/// Hard real-time control loop running at 1 000 Hz.
///
/// Every iteration represents one 1 ms time-step.  Safety-critical logic
/// (PID updates, motor commands, watchdog resets) belongs here — it must
/// never block on I/O or Python.
///
/// Currently the loop emits a heartbeat JSON line to stdout every 1 000 ticks
/// (≈ 1 second) so the Python brain knows the spinal cord is alive.
async fn control_loop() {
    // 1 000 µs = 1 kHz
    let mut interval = time::interval(Duration::from_micros(1_000));
    let mut tick: u64 = 0;

    loop {
        interval.tick().await;
        tick += 1;

        // ── 1 Hz heartbeat ───────────────────────────────────────────────────
        if tick % 1_000 == 0 {
            let heartbeat = serde_json::json!({
                "type":       "heartbeat",
                "tick":       tick,
                "uptime_s":   tick / 1_000,
                "timestamp_ms": now_ms(),
                "status":     "nominal"
            });
            println!("{heartbeat}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IPC listener  (reads commands from Python via stdin)
// ─────────────────────────────────────────────────────────────────────────────

/// Reads newline-delimited JSON commands from stdin and dispatches them.
///
/// Each line must be a valid :struct:`SpinalCommand` JSON object.  Malformed
/// lines are logged to stderr and skipped — they never crash the loop.
async fn ipc_listener() {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();

    eprintln!("NAYAK Spinal: IPC listener ready — awaiting commands on stdin");

    while let Ok(Some(line)) = reader.next_line().await {
        let parse_start = Instant::now();

        match serde_json::from_str::<SpinalCommand>(&line) {
            Ok(cmd) => {
                // Dispatch by command type
                let (success, data) = dispatch_command(&cmd);

                let latency_us = parse_start.elapsed().as_micros() as u64;
                let response = SpinalResponse {
                    command_type: cmd.command_type.clone(),
                    device_id:    cmd.device_id.clone(),
                    success,
                    data,
                    timestamp_ms: now_ms(),
                    latency_us,
                };

                match serde_json::to_string(&response) {
                    Ok(json) => println!("{json}"),
                    Err(e)   => eprintln!("NAYAK Spinal: serialise error: {e}"),
                }
            }
            Err(e) => {
                eprintln!("NAYAK Spinal: parse error on line {:?}: {e}", &line[..line.len().min(80)]);
            }
        }
    }

    eprintln!("NAYAK Spinal: stdin closed — IPC listener exiting");
}

// ─────────────────────────────────────────────────────────────────────────────
// Command dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch a parsed command and return `(success, data)`.
///
/// Extend the match arms here to add new opcode handlers without touching
/// the IPC plumbing.
fn dispatch_command(cmd: &SpinalCommand) -> (bool, serde_json::Value) {
    match cmd.command_type.as_str() {
        // ── Ping / health check ──────────────────────────────────────────────
        "ping" => (
            true,
            serde_json::json!({ "pong": true, "echo": cmd.device_id }),
        ),

        // ── GPIO write ───────────────────────────────────────────────────────
        "gpio_write" => {
            let value = cmd.payload.get("value")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;
            let pin_num = cmd.payload.get("pin")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;
            let mut pin = hal::GpioPin::new(pin_num, hal::PinMode::Output);
            let ok = pin.write(value);
            (ok, serde_json::json!({ "pin": pin_num, "value": value, "written": ok }))
        }

        // ── GPIO read ────────────────────────────────────────────────────────
        "gpio_read" => {
            let pin_num = cmd.payload.get("pin")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8;
            let pin = hal::GpioPin::new(pin_num, hal::PinMode::Input);
            let val = pin.read();
            (true, serde_json::json!({ "pin": pin_num, "value": val }))
        }

        // ── Unknown opcode ───────────────────────────────────────────────────
        other => {
            eprintln!("NAYAK Spinal: unknown command_type '{other}'");
            (false, serde_json::json!({ "error": format!("unknown command: {other}") }))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    eprintln!("╔══════════════════════════════════════════╗");
    eprintln!("║  NAYAK Spinal Cord v0.1.0                ║");
    eprintln!("║  Hard Real-Time Runtime Starting         ║");
    eprintln!("║  Control loop: 1000 Hz                   ║");
    eprintln!("║  IPC: JSON over stdin/stdout             ║");
    eprintln!("╚══════════════════════════════════════════╝");

    // Run control loop and IPC listener concurrently.
    // Neither task blocks the other — tokio schedules them cooperatively.
    tokio::join!(
        control_loop(),
        ipc_listener(),
    );
}
