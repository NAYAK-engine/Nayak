/*
 * NAYAK Spinal Cord — Hard Real-Time Runtime
 *
 * This is the Rust layer of NAYAK.
 * Runs independently of the Python brain.
 * Handles Layer 1 (HAL) and Layer 4 (Action) at 1000 Hz+.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                  IPC Architecture                               │
 * │                                                                 │
 * │  JSON over stdin/stdout  →  control signals ONLY  (small text) │
 * │  Shared memory (mmap)    →  sensor frames  (large binary, RT)  │
 * │                                                                 │
 * │  RULE: Never mix them. Never put sensor data in JSON IPC.       │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Data flow:
 *   Commands   : Python writes a JSON line to the spinal cord's stdin.
 *   Responses  : Rust writes a JSON line to stdout; Python reads it.
 *   Heartbeat  : Every 1 000 control-loop ticks (≈ 1 second) a heartbeat
 *                JSON line is written to stdout.
 *   Sensor SHM : Every 100 ticks (≈ 10 Hz) a raw sensor frame is written
 *                into a memory-mapped file.  Python reads it with mmap —
 *                zero serialization, zero JSON overhead.
 *   Stderr     : Diagnostic / error messages only — never parsed by Python.
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
 *
 * Shared memory frame file layout (no header, raw bytes):
 *   [0 .. N-1]  Raw sensor bytes (camera, LIDAR, tensor, etc.)
 *   Python reads exactly N bytes via mmap — no parsing needed.
 */

mod hal;

use std::fs::OpenOptions;
use std::io;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use memmap2::MmapMut;
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

/// Write a raw sensor frame into a memory-mapped file.
///
/// Opens (or creates) the file at `shm_path`, sets its length to exactly
/// `data.len()` bytes, memory-maps it, and copies `data` directly into the
/// map.  No JSON serialization.  No stdout.  Pure binary.
///
/// # Channel contract
/// This function is the **only** place where sensor data leaves the Rust
/// process.  The Python side reads it with `mmap` — no JSON, no copy.
///
/// # Arguments
/// * `data`     – Raw sensor bytes (camera frame, LIDAR scan, tensor, …).
/// * `shm_path` – Path to the backing file (e.g. `/tmp/nayak_sensor_frame`).
///
/// # Errors
/// All I/O errors are logged to `stderr` and the function returns without
/// panicking.  A transient write failure (e.g. full disk) is non-fatal —
/// the control loop continues at 1 kHz regardless.
fn write_shm_frame(data: &[u8], shm_path: &str) {
    // ── Open or create the backing file ─────────────────────────────────────
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(shm_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("NAYAK Spinal [SHM]: cannot open '{}': {}", shm_path, e);
            return;
        }
    };

    // ── Resize the file to match the payload ─────────────────────────────────
    // set_len is cheap: it only adjusts the file metadata, not the data.
    if let Err(e) = file.set_len(data.len() as u64) {
        eprintln!(
            "NAYAK Spinal [SHM]: set_len({}) failed on '{}': {}",
            data.len(), shm_path, e
        );
        return;
    }

    // ── Memory-map the file ──────────────────────────────────────────────────
    // SAFETY: We own the file exclusively for the duration of this function.
    // The mmap is dropped before `file` is released, satisfying the aliasing
    // requirement documented in the memmap2 crate.
    let mut mmap: MmapMut = match unsafe { MmapMut::map_mut(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "NAYAK Spinal [SHM]: mmap failed for '{}': {}",
                shm_path, e
            );
            return;
        }
    };

    // ── Copy raw bytes directly into the map ─────────────────────────────────
    // No serialization.  No JSON.  Raw bytes only.
    mmap[..data.len()].copy_from_slice(data);

    // Explicitly flush to ensure the OS page cache is updated before
    // the Python reader maps the same file.
    if let Err(e) = mmap.flush() {
        eprintln!(
            "NAYAK Spinal [SHM]: flush failed for '{}': {}",
            shm_path, e
        );
        // Non-fatal: the data may still be readable by the consumer.
    }
}

// Silence the unused-import warning; io is referenced in future Pi HAL code.
#[allow(dead_code)]
fn _io_placeholder() -> io::Result<()> { Ok(()) }

// ─────────────────────────────────────────────────────────────────────────────
// Control loop  (1 000 Hz independent ticker)
// ─────────────────────────────────────────────────────────────────────────────

/// Hard real-time control loop running at 1 000 Hz.
///
/// Every iteration represents one 1 ms time-step.  Safety-critical logic
/// (PID updates, motor commands, watchdog resets) belongs here — it must
/// never block on I/O or Python.
///
/// # IPC channels used by this loop
///
/// | Cadence     | Channel             | Content                          |
/// |-------------|---------------------|----------------------------------|
/// | Every 100 t | SHM mmap file       | Raw sensor frame (binary, 10 Hz) |
/// | Every 1000 t| JSON → stdout       | Heartbeat control signal (1 Hz)  |
///
/// The two channels are **never mixed**: sensor data never enters JSON IPC
/// and control signals never go through shared memory.
async fn control_loop() {
    // 1 000 µs = 1 kHz
    let mut interval = time::interval(Duration::from_micros(1_000));
    let mut tick: u64 = 0;

    // Path where Python reads sensor frames via mmap.
    // On Linux/macOS this lives in the OS temp directory.
    // On Windows it will be created in C:\Users\<user>\AppData\Local\Temp\
    // or wherever the OS resolves "/tmp" via WSL / MSYS2 compatibility layers.
    // Adjust to a platform-specific path if needed (e.g. via env var).
    let shm_path = if cfg!(windows) {
        std::env::temp_dir()
            .join("nayak_sensor_frame")
            .to_string_lossy()
            .into_owned()
    } else {
        "/tmp/nayak_sensor_frame".to_owned()
    };

    loop {
        interval.tick().await;
        tick += 1;

        // ── 10 Hz sensor frame → shared memory ──────────────────────────────
        // Write a raw sensor frame every 100 ticks (≈ 10 Hz).
        // In production replace `vec![0u8; 1024]` with real sensor data
        // (camera frame bytes, LIDAR scan, IMU tensor, etc.).
        //
        // Rule: NO JSON. NO stdout. Raw bytes into the mmap file ONLY.
        if tick % 100 == 0 {
            let frame = vec![0u8; 1024]; // 1 KB simulated sensor payload
            write_shm_frame(&frame, &shm_path);
        }

        // ── 1 Hz heartbeat → JSON stdout ─────────────────────────────────────
        // Control signals only: tiny text payload, tells the Python brain
        // the spinal cord is alive.  Never carries sensor data.
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
    eprintln!("║  Control IPC : JSON over stdin/stdout    ║");
    eprintln!("║  Sensor IPC  : mmap shared memory (SHM)  ║");
    eprintln!("╚══════════════════════════════════════════╝");

    // Run control loop and IPC listener concurrently.
    // Neither task blocks the other — tokio schedules them cooperatively.
    tokio::join!(
        control_loop(),
        ipc_listener(),
    );
}
