"""Poll nvidia-smi for GPU utilization, temp, power, memory.
Returns dict with current values.  Caches the subprocess path."""
import subprocess

_NVSMI = r'C:\Windows\System32\nvidia-smi.exe'


def gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            [_NVSMI,
             '--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=2, text=True)
    except Exception as e:
        return {'error': str(e)}
    parts = [p.strip() for p in out.strip().splitlines()[0].split(',')]
    try:
        return {
            'util_pct': int(parts[0]),
            'temp_c': int(parts[1]),
            'power_w': float(parts[2]),
            'mem_used_mib': int(parts[3]),
            'mem_total_mib': int(parts[4]),
        }
    except (ValueError, IndexError):
        return {'raw': out.strip()}


if __name__ == '__main__':
    import json
    print(json.dumps(gpu_stats(), indent=2))
