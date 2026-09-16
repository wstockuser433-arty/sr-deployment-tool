"""
job_manager.py
==============
Manages long-running subprocess jobs (training, inference, preprocessing).
Streams stdout/stderr as Server-Sent Events.
"""
import asyncio
import os
import re
import signal
import subprocess
import sys
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Deque, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    cmd: list
    cwd: str
    status: JobStatus = JobStatus.PENDING
    process: Optional[subprocess.Popen] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    return_code: Optional[int] = None
    output_dir: Optional[str] = None  # set for jobs that write preview images (see preprocessing.py)
    meta: Optional[dict] = None       # arbitrary metadata (e.g. output_dir for raw inference)


# Global in-memory job store
_jobs: Dict[str, Job] = {}


def create_job(
    cmd: list,
    cwd: str,
    output_dir: Optional[str] = None,
    meta: Optional[dict] = None,
) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = Job(
        job_id=job_id, cmd=cmd, cwd=cwd,
        output_dir=output_dir, meta=meta,
    )
    return job_id


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def list_jobs() -> list:
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "return_code": j.return_code,
            "log_lines": len(j.logs),
        }
        for j in _jobs.values()
    ]


async def launch_job(job_id: str) -> None:
    """Launch the subprocess for a given job and stream its output into job.logs.

    Uses subprocess.Popen + asyncio.to_thread so it works on Windows where
    asyncio.create_subprocess_exec requires ProactorEventLoop (not available
    when uvicorn uses SelectorEventLoop).
    """
    job = _jobs.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job.status = JobStatus.RUNNING
    env = {**os.environ}

    # On Windows, spawn the child in its own process group so that signals
    # (Ctrl+C / SIGTERM) sent to the uvicorn parent do NOT propagate into the
    # child and interrupt numpy/torch imports with a KeyboardInterrupt.
    extra_kwargs: dict = {}
    if sys.platform == "win32":
        extra_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(
            job.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=job.cwd,
            env=env,
            **extra_kwargs,
        )
        job.process = proc

        def _stream_output() -> int:
            assert proc.stdout is not None, "stdout pipe was not created"
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                job.logs.append(line)
            proc.wait()
            return proc.returncode or 0

        return_code = await asyncio.to_thread(_stream_output)
        job.return_code = return_code
        # Don't overwrite an explicit CANCELLED status set by cancel_job()
        if job.status != JobStatus.CANCELLED:
            job.status = JobStatus.COMPLETED if return_code == 0 else JobStatus.FAILED

    except asyncio.CancelledError:
        job.status = JobStatus.CANCELLED
        if job.process and job.process.returncode is None:
            job.process.terminate()
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        job.logs.append(f"[job_manager] Error launching job: {type(exc).__name__}: {exc}")
        for line in tb.splitlines():
            job.logs.append(line)
        job.status = JobStatus.FAILED


_TERMINAL_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


async def stream_logs(job_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator for SSE streaming.
    Yields new log lines as they arrive, then sends a final 'status' event on completion.
    Emits non-terminal 'status_update' events for PAUSED <-> RUNNING transitions
    without closing the stream.
    """
    job = _jobs.get(job_id)
    if not job:
        yield f"data: [ERROR] Job {job_id} not found\n\n"
        return

    sent = 0
    last_status: Optional[JobStatus] = None

    while True:
        logs_snapshot = list(job.logs)
        new_lines = logs_snapshot[sent:]
        for line in new_lines:
            yield f"data: {line}\n\n"
        sent += len(new_lines)

        current_status = job.status

        # Emit interim status changes (PENDING/RUNNING/PAUSED) without closing the stream
        if current_status != last_status and current_status not in _TERMINAL_STATUSES:
            yield f"event: status_update\ndata: {current_status.value}\n\n"
            last_status = current_status

        if current_status in _TERMINAL_STATUSES:
            # Flush any remaining lines then close
            logs_snapshot = list(job.logs)
            for line in logs_snapshot[sent:]:
                yield f"data: {line}\n\n"
            yield f"event: status\ndata: {current_status.value}\n\n"
            break

        await asyncio.sleep(0.3)


# ── Process suspend / resume / kill helpers ───────────────────────────────────

def _suspend_process(proc: subprocess.Popen) -> None:
    """Suspend the process and all its children."""
    try:
        import psutil
        p = psutil.Process(proc.pid)
        children = p.children(recursive=True)
        for child in reversed(children):
            try:
                child.suspend()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        p.suspend()
        return
    except ImportError:
        pass
    # Fallback: Windows-only NtSuspendProcess via ctypes (parent process only)
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_SUSPEND_RESUME = 0x0800
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, proc.pid)
            if handle:
                ctypes.windll.ntdll.NtSuspendProcess(handle)
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass


def _resume_process(proc: subprocess.Popen) -> None:
    """Resume the process and all its children."""
    try:
        import psutil
        p = psutil.Process(proc.pid)
        p.resume()
        children = p.children(recursive=True)
        for child in children:
            try:
                child.resume()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_SUSPEND_RESUME = 0x0800
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, proc.pid)
            if handle:
                ctypes.windll.ntdll.NtResumeProcess(handle)
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass


def _kill_process(proc: subprocess.Popen) -> None:
    """Force-kill the process and all its children (works on suspended processes)."""
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for child in reversed(p.children(recursive=True)):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        p.kill()
        return
    except (psutil.NoSuchProcess, psutil.AccessDenied, ImportError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def pause_job(job_id: str) -> bool:
    """Suspend the running subprocess. Sets status to PAUSED."""
    job = _jobs.get(job_id)
    if not job or job.status != JobStatus.RUNNING:
        return False
    if job.process and job.process.returncode is None:
        _suspend_process(job.process)
    job.status = JobStatus.PAUSED
    job.logs.append("[gui] Job paused")
    return True


def resume_job(job_id: str) -> bool:
    """Resume a paused subprocess. Sets status back to RUNNING."""
    job = _jobs.get(job_id)
    if not job or job.status != JobStatus.PAUSED:
        return False
    if job.process and job.process.returncode is None:
        _resume_process(job.process)
    job.status = JobStatus.RUNNING
    job.logs.append("[gui] Job resumed")
    return True


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if not job:
        return False
    if job.process and job.process.returncode is None:
        if job.status == JobStatus.PAUSED:
            # Suspended processes on Windows can't receive signals — force kill instead
            _kill_process(job.process)
        else:
            try:
                if sys.platform == "win32":
                    # SIGTERM doesn't exist on Windows; use CTRL_BREAK_EVENT which
                    # is compatible with CREATE_NEW_PROCESS_GROUP children.
                    os.kill(job.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.kill(job.process.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    job.status = JobStatus.CANCELLED
    return True


def get_job_summary(job_id: str) -> Optional[dict]:
    job = _jobs.get(job_id)
    if not job:
        return None
    return {
        "job_id":       job.job_id,
        "status":       job.status,
        "return_code":  job.return_code,
        "logs":         list(job.logs)[-200:],  # last 200 lines for polling
        "meta":         job.meta,
    }


def get_recent_lines(
    job_id: str,
    max_lines: int = 200,
    skip_debug: bool = False,
) -> list[str]:
    """Return the most recent log lines for a job (never raises).

    If skip_debug is True, DEBUG-level lines are filtered out before the
    last `max_lines` are returned. This is important for jobs that emit
    heavy GDAL/rasterio DEBUG chatter (preprocessing) — otherwise the
    interesting INFO lines get pushed out of the window entirely.
    """
    job = _jobs.get(job_id)
    if job is None:
        return []
    try:
        lines = list(job.logs)
    except Exception:
        return []

    if skip_debug:
        # Match lines that start with "DEBUG" either immediately or after a
        # leading timestamp ("11:22:16  DEBUG      ...") — both forms occur
        # because the pipeline logs via both print() and logging.
        _dbg_re = re.compile(r'^\s*(?:\d{2}:\d{2}:\d{2}\s+)?DEBUG\b')
        lines = [ln for ln in lines if not _dbg_re.match(ln)]

    return lines[-max_lines:]