from __future__ import annotations

import ctypes as ct
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import psutil


@dataclass(frozen=True)
class ProcessRun:
    status: str
    returncode: int | None
    elapsed_s: float
    peak_rss_bytes: int | None
    diagnostic: str


Progress = Callable[[str], None]


class WindowsJob:
    """A kill-on-close Job Object also owns descendants created by the engine."""

    def __init__(self) -> None:
        from ctypes import wintypes as wt

        class BasicLimits(ct.Structure):
            _fields_ = [("per_process_time", ct.c_int64), ("per_job_time", ct.c_int64),
                        ("flags", wt.DWORD), ("min_working_set", ct.c_size_t),
                        ("max_working_set", ct.c_size_t), ("active_processes", wt.DWORD),
                        ("affinity", ct.c_size_t), ("priority", wt.DWORD), ("schedule", wt.DWORD)]

        class ExtendedLimits(ct.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ct.c_uint64 * 6),
                        ("process_memory", ct.c_size_t), ("job_memory", ct.c_size_t),
                        ("peak_process", ct.c_size_t), ("peak_job", ct.c_size_t)]

        self.api = ct.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
        self.api.CreateJobObjectW.restype = wt.HANDLE
        self.api.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
        self.api.SetInformationJobObject.restype = wt.BOOL
        self.api.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.api.OpenProcess.restype = wt.HANDLE
        self.api.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        self.api.AssignProcessToJobObject.restype = wt.BOOL
        self.api.CloseHandle.argtypes = [wt.HANDLE]
        self.api.CloseHandle.restype = wt.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ct.byref(limits), ct.sizeof(limits)):
            error = ct.WinError(ct.get_last_error())
            self.close()
            raise error

    def assign(self, pid: int) -> None:
        handle = self.api.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
        if not handle:
            raise ct.WinError(ct.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, handle):
                raise ct.WinError(ct.get_last_error())
        finally:
            self.api.CloseHandle(handle)

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _rss(process: psutil.Process) -> int | None:
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.Error:
        processes = [process]
    values = []
    for member in processes:
        try:
            values.append(member.memory_info().rss)
        except psutil.Error:
            continue
    return sum(values) if values else None


def _terminate(proc: subprocess.Popen, job: WindowsJob | None, assigned: bool) -> None:
    if job:
        job.close()
        if not assigned and proc.poll() is None:
            proc.kill()
    elif os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


def log_tail(path: Path, limit: int = 12_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read(limit).decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    except OSError:
        return ""


def run_process(command: list[str], *, cwd: Path, log: Path, timeout_s: float,
                cancel: Event, progress: Progress | None = None) -> ProcessRun:
    started = time.monotonic()
    peak = None
    status = "failed"
    diagnostic = ""
    proc = None
    job = None
    assigned = False
    with log.open("wb") as output:
        if cancel.is_set():
            return ProcessRun("cancelled", None, 0.0, None, "任务启动前已取消。")
        try:
            if os.name == "nt":
                job = WindowsJob()
                # Keep the main thread suspended until the job owns the process.
                options = {"creationflags": subprocess.CREATE_NO_WINDOW | 0x00000004}
            else:
                options = {"start_new_session": True}
            proc = subprocess.Popen(command, executable=command[0], cwd=cwd, stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, shell=False, **options)
            if job:
                job.assign(proc.pid)
                assigned = True
            try:
                monitored = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                monitored = None
            if job and monitored is not None:
                monitored.resume()
            with log.open("rb") as reader:
                last_message = None
                while True:
                    if monitored is not None and (sample := _rss(monitored)) is not None:
                        peak = sample if peak is None else max(peak, sample)
                    if progress:
                        data = reader.read(65_536)
                        lines = [line.strip() for line in re.split(rb"[\r\n]+", data) if line.strip()]
                        if lines and lines[-1] != last_message:
                            last_message = lines[-1]
                            progress(last_message[-400:].decode("utf-8", errors="replace"))
                    if cancel.is_set():
                        status, diagnostic = "cancelled", "用户已取消，已回收进程树。"
                        break
                    if proc.poll() is not None:
                        status = "success" if proc.returncode == 0 else "failed"
                        diagnostic = "" if proc.returncode == 0 else f"原生进程退出码 {proc.returncode}。"
                        break
                    if time.monotonic() - started >= timeout_s:
                        status, diagnostic = "timed_out", f"原生进程超过 {timeout_s:g} 秒，已回收进程树。"
                        break
                    cancel.wait(0.08)
        except (OSError, psutil.Error) as error:
            diagnostic = f"无法运行或监控原生进程 ({error})"
        finally:
            if proc is not None:
                _terminate(proc, job, assigned)
            elif job is not None:
                job.close()
    return ProcessRun(status, proc.returncode if proc else None, time.monotonic() - started, peak, diagnostic)
