"""
Serial bridge for Predictive Machine Vibration Monitoring.

Reads structured vibration packets from the Arduino Uno over USB Serial,
validates packet framing and numeric integrity, and forwards complete
vibration windows to the Flask API's /ingest endpoint.

Also handles bidirectional commands: checks the Flask backend for pending
motor commands (START, STOP, SPEED) and writes them to the Arduino serial port.

Usage:
  python serial_reader.py --port COM3 [--baud 115200] [--host http://127.0.0.1:5000]
  python serial_reader.py --list-ports
"""
import argparse
import sys
import time

import requests
import serial
import serial.tools.list_ports


def list_serial_ports():
    """Lists all available serial communication ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports detected. Connect Arduino Uno via USB and check Device Manager.")
        return []
    print("\nAvailable Serial Ports:")
    for p in ports:
        desc = p.description or "Unknown"
        hwid = p.hwid or ""
        print(f"  - {p.device:<10} : {desc} [{hwid}]")
    print()
    return ports


def parse_packet(line: str):
    """Parses and validates an incoming line from the Arduino firmware.

    Supported formats:
      1. Structured Protocol:
         $VIB,<seq>,<sample_rate>,<rpm>,<count>,<s0>,<s1>,...,<sN-1>
      2. Status / Diagnostic:
         $STATUS,... or $OK,... or $ERR,...
      3. Legacy CSV fallback:
         <rpm>,<s0>,<s1>,...
    """
    line = line.strip()
    if not line:
        return None

    # Handle status and diagnostic messages
    if line.startswith("$STATUS") or line.startswith("$OK") or line.startswith("$BOOT"):
        print(f"[Firmware Info] {line}")
        return None
    if line.startswith("$ERR"):
        print(f"[Firmware Error] {line}", file=sys.stderr)
        return None

    # Structured vibration packet
    if line.startswith("$VIB,"):
        parts = line.split(",")
        if len(parts) < 6:
            print(f"[Warning] Incomplete $VIB header: {line[:40]}")
            return None

        try:
            seq = int(parts[1])
            sample_rate = float(parts[2])
            rpm = float(parts[3])
            count = int(parts[4])
            raw_samples = parts[5:]

            if len(raw_samples) != count:
                print(f"[Warning] Packet count mismatch: header={count}, received={len(raw_samples)} (seq={seq})")
                return None

            samples = [float(v) for v in raw_samples]
            return {
                "sequence": seq,
                "sample_rate": sample_rate,
                "rpm": rpm,
                "window": samples,
            }
        except (ValueError, TypeError) as e:
            print(f"[Warning] Malformed $VIB numeric data: {e}")
            return None

    # Legacy CSV fallback
    if "," in line:
        parts = line.split(",")
        try:
            values = [float(v) for v in parts]
            if len(values) >= 10:
                return {
                    "sequence": None,
                    "sample_rate": 800.0,
                    "rpm": values[0],
                    "window": values[1:],
                }
        except ValueError:
            pass

    return None


def forward_to_api(host: str, payload: dict, timeout: float = 3.0):
    """Sends vibration window payload to Flask /ingest endpoint."""
    try:
        url = f"{host.rstrip('/')}/ingest"
        res = requests.post(url, json=payload, timeout=timeout)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"[API Error {res.status_code}] {res.text[:120]}")
            return None
    except requests.RequestException as e:
        print(f"[Network Error] Could not connect to {host}: {e}")
        return None


def check_and_forward_motor_commands(host: str, ser: serial.Serial):
    """Polls Flask for any pending user motor commands and writes to serial."""
    try:
        url = f"{host.rstrip('/')}/motor/pending_command"
        res = requests.get(url, timeout=0.5)
        if res.status_code == 200:
            cmd = res.json().get("command")
            if cmd:
                print(f"[Command] Forwarding to Arduino: '{cmd}'")
                ser.write(f"{cmd}\n".encode("utf-8"))
                ser.flush()
    except Exception:
        pass


def run_bridge(port: str, baud: int, host: str):
    """Main loop: manages serial connection, bidirectional data, and reconnection."""
    print(f"Target Arduino Port: {port} @ {baud} baud")
    print(f"Target Flask Server:  {host}")
    print("Press Ctrl+C to stop.\n")

    ser = None
    last_cmd_poll = 0

    while True:
        try:
            if ser is None or not ser.is_open:
                print(f"Connecting to {port}...")
                ser = serial.Serial(port, baud, timeout=1.0)
                # Allow Arduino bootloader to complete reset
                time.sleep(2.0)
                ser.reset_input_buffer()
                print(f"Connected to {port} successfully.\n")

            # Check for incoming line from Arduino
            if ser.in_waiting > 0:
                raw_bytes = ser.readline()
                line = raw_bytes.decode("utf-8", errors="replace").strip()
                packet = parse_packet(line)

                if packet is not None:
                    rpm_str = f"{packet['rpm']:.1f}" if packet.get("rpm") is not None else "--"
                    seq_str = f"#{packet['sequence']}" if packet.get("sequence") is not None else ""

                    # Forward to Flask
                    api_res = forward_to_api(host, {
                        "window": packet["window"],
                        "sample_rate": packet["sample_rate"],
                        "rpm": packet["rpm"],
                    })

                    if api_res:
                        pred = api_res.get("prediction") or "uncalibrated"
                        prob = api_res.get("probability")
                        conf = f"({prob * 100:.1f}%)" if prob is not None else ""
                        print(f"Packet {seq_str:<6} | RPM: {rpm_str:>6} | Samples: {len(packet['window']):<3} | Verdict: {pred:<12} {conf}")

            # Periodically poll Flask for motor commands (every 200 ms)
            now = time.time()
            if now - last_cmd_poll > 0.2:
                last_cmd_poll = now
                check_and_forward_motor_commands(host, ser)

            time.sleep(0.005)

        except serial.SerialException as se:
            print(f"[Serial Disconnect] Port {port} error: {se}")
            if ser and ser.is_open:
                try:
                    ser.close()
                except Exception:
                    pass
            ser = None
            print("Retrying connection in 3 seconds...")
            time.sleep(3.0)

        except KeyboardInterrupt:
            print("\nShutting down serial reader...")
            if ser and ser.is_open:
                try:
                    print("Sending safe motor stop to Arduino...")
                    ser.write(b"MOTOR STOP\n")
                    time.sleep(0.1)
                    ser.close()
                except Exception:
                    pass
            print("Shutdown complete.")
            break


def main():
    parser = argparse.ArgumentParser(description="Arduino Serial Bridge for Machine Vibration Monitoring.")
    parser.add_argument("--port", help="Serial port name (e.g., COM3 on Windows, /dev/ttyACM0 on Linux)")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate (default: 115200)")
    parser.add_argument("--host", default="http://127.0.0.1:5000", help="Flask backend URL")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        sys.exit(0)

    port = args.port
    if not port:
        ports = list_serial_ports()
        if len(ports) == 1:
            port = ports[0].device
            print(f"Auto-selected only available port: {port}")
        else:
            print("Error: Port not specified. Run with '--port <PORT_NAME>' or '--list-ports'.")
            sys.exit(1)

    run_bridge(port, args.baud, args.host)


if __name__ == "__main__":
    main()