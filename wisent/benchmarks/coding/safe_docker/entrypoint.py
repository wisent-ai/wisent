from __future__ import annotations
import json, os, shutil, subprocess, sys, time, signal, resource

JOB_FILE = "/job/job.json"
WORKDIR = "/work"

def set_limits(job):
    """
    Set resource limits for the sandboxed process.
    
    attributes:
        job:
            A Job object containing resource limit parameters.
    
    example:
        
    """
    resource.setrlimit(resource.RLIMIT_CPU,  (job["cpu_limit_s"],)*2)
    resource.setrlimit(resource.RLIMIT_AS,   (job["mem_limit_mb"]*1024*1024,)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(job["fsize_mb"]*1024*1024,)*2)
    resource.setrlimit(resource.RLIMIT_NPROC,(job["nproc"],)*2)
    resource.setrlimit(resource.RLIMIT_NOFILE,(job["nofile"],)*2)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.setsid()

def run(argv: list[str], job) -> tuple[int,str,str,float,str]:
    """
    Run a command in a subprocess with resource limits.

    attributes:
        argv:
            Command and arguments to run as a list of strings.
        job:
            A Job object containing resource limit parameters.

    returns:
        A tuple containing:
            - exit code (int)
            - standard output (str)
            - standard error (str)
            - elapsed time in seconds (float)
            - status (str): "ok", "nonzero", "timeout", "missing", or "error"

    example:
            >>> code, out, err, elapsed, status = run(["python3", "solution.py"], job)
            >>> print(status)
            "ok"
            >>> print(elapsed)
            0.123
            >>> print(out)
            "Hello, world!"
            >>> print(err)
            ""
    """
    start = time.time()
    try:
        p = subprocess.Popen(argv, cwd=WORKDIR, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             preexec_fn=lambda: set_limits(job))
        try:
            out, err = p.communicate(timeout=job["wall_timeout_s"])
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
            return 124, "", f"Time limit exceeded ({job['wall_timeout_s']}s)\n", time.time()-start, "timeout"
        status = "ok" if p.returncode == 0 else "nonzero"
        return p.returncode, out, err, time.time()-start, status
    except FileNotFoundError as e:
        return 127, "", f"{e}\n", time.time()-start, "missing"
    except Exception as e:
        return 1, "", f"{e}\n", time.time()-start, "error"

def copy_job():
    """
    Copy job files from /job to /work directory.
    """
    os.makedirs(WORKDIR, exist_ok=True)
    for root, _, files in os.walk("/job"):
        rel = os.path.relpath(root, "/job")
        dst = os.path.join(WORKDIR, "" if rel == "." else rel)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(dst, f))

def main():
    """
    Main function to execute the job defined in /job/job.json.

    returns:
        Exit code 0 on success, 2 if job file is missing.
    """
    if not os.path.exists(JOB_FILE):
        print("Missing /job/job.json", file=sys.stderr); return 2
    with open(JOB_FILE, "r", encoding="utf-8") as f:
        job = json.load(f)

    copy_job()

    # optional quick syntax check for Python
    if job["language"] == "python":
        _, _, err, _, _ = run([sys.executable, "-m", "py_compile", "solution.py"], job)
        if err:
            print(json.dumps({"status":"compile_error","stdout":"","stderr":err,"elapsed":0.0,"exit_code":1}))
            return 0

    if job.get("compile"):
        code, out, err, el, _ = run(job["compile"]["argv"], job)
        if code != 0:
            print(json.dumps({"status":"compile_error","stdout":out,"stderr":err,"elapsed":el,"exit_code":code}))
            return 0

    code, out, err, el, status = run(job["run"]["argv"], job)
    payload = {
        "status": "ok" if code == 0 else ("timeout" if status == "timeout" else "runtime_error"),
        "stdout": out, "stderr": err, "elapsed": el, "exit_code": code
    }
    print(json.dumps(payload))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
