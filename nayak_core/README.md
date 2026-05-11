# NAYAK Spinal Cord

The hard real-time Rust runtime for NAYAK OS.

Runs **independently** of the Python brain at **1 000 Hz**, handling:
- **Layer 1 (HAL):** GPIO, sensor reads, camera frame acquisition
- **Layer 4 (Action):** Motor commands, actuator control

The Python brain (Layer 3 Cognition) sends high-level intents; the Spinal Cord
translates them into deterministic, microsecond-precision hardware commands
without any Python GIL or garbage-collector jitter.

---

## Architecture

```
Python Brain (Layer 3)
        │
        │  JSON command  ──►  stdin
        │
   ┌────▼──────────────────────────┐
   │     Rust Spinal Cord          │
   │                               │
   │  ┌─────────────────────────┐  │
   │  │  1000 Hz control_loop   │  │  ← independent of Python
   │  └─────────────────────────┘  │
   │  ┌─────────────────────────┐  │
   │  │  ipc_listener (stdin)   │  │  ← reads commands from Python
   │  └─────────────────────────┘  │
   │  ┌─────────────────────────┐  │
   │  │  hal.rs (GPIO / sensors)│  │  ← hardware abstraction
   │  └─────────────────────────┘  │
   └───────────────────────────────┘
        │
        │  JSON response ──►  stdout
        ▼
Python Brain (SpinalBridge reads stdout)
```

---

## Build

```bash
cd nayak_core
cargo build --release
```

Compiled binary: `target/release/nayak_spinal`

---

## Run standalone (for testing)

```bash
./target/release/nayak_spinal
```

Send a test ping via stdin:
```bash
echo '{"command_type":"ping","device_id":"test","payload":{},"timestamp_ms":0}' \
  | ./target/release/nayak_spinal
```

---

## Run via Python bridge

The `SpinalBridge` in `nayak/hal/spinal_bridge.py` manages the process
lifecycle automatically:

```python
from nayak.hal.spinal_bridge import spinal_bridge

await spinal_bridge.start()
await spinal_bridge.send_command("gpio_write", "gpio.17", {"pin": 17, "value": 1})
await spinal_bridge.stop()
```

---

## IPC Protocol

### Command (Python → Rust, one JSON line per command)

```json
{
  "command_type": "gpio_write",
  "device_id":    "gpio.17",
  "payload":      { "pin": 17, "value": 1 },
  "timestamp_ms": 1715000000000
}
```

Supported `command_type` values:

| Opcode       | Description                          |
|:-------------|:-------------------------------------|
| `ping`       | Health check — returns `pong: true`  |
| `gpio_write` | Write `value` (0/1) to `pin`         |
| `gpio_read`  | Read current value from `pin`        |

### Response (Rust → Python, one JSON line per response)

```json
{
  "command_type": "gpio_write",
  "device_id":    "gpio.17",
  "success":      true,
  "data":         { "pin": 17, "value": 1, "written": true },
  "timestamp_ms": 1715000000001,
  "latency_us":   42
}
```

### Heartbeat (Rust → Python, every ~1 second)

```json
{
  "type":          "heartbeat",
  "tick":          1000,
  "uptime_s":      1,
  "timestamp_ms":  1715000001000,
  "status":        "nominal"
}
```

---

## Raspberry Pi Deployment

Replace the stub bodies in `src/hal.rs` with real `rppal` calls:

```toml
# Cargo.toml
[dependencies]
rppal = "0.17"
```

```rust
// src/hal.rs — GpioPin::write()
use rppal::gpio::Gpio;
let gpio = Gpio::new().unwrap();
let mut pin = gpio.get(self.pin).unwrap().into_output();
if value == 1 { pin.set_high() } else { pin.set_low() }
```
