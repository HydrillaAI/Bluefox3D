import os
import signal

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode()
    except OSError:
        continue
    if os.getpid() == int(pid):
        continue
    hit = (
        "run_compare.py" in cmd
        or "retry_failed_1536.py" in cmd
        or ("inference.py" in cmd and "work/BlueFox3D" in cmd)
    )
    if hit:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print("killed", pid)
        except OSError as exc:
            print("skip", pid, exc)
