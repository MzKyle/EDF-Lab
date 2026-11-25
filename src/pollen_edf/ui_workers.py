from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot


class JobSignals(QObject):
    complete = Signal(object, object)
    progress = Signal(object)


class Job(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.cancel = Event()
        self.signals = JobSignals()

    @Slot()
    def run(self):
        try:
            result = self.operation(self.cancel, self.signals.progress.emit)
        except Exception as error:
            self.signals.complete.emit(None, error)
        else:
            self.signals.complete.emit(result, None)


class WorkSession(QObject):
    busy_changed = Signal(bool)
    complete = Signal(object, object)
    progress = Signal(object)

    def __init__(self, pool: QThreadPool, parent: QObject | None = None):
        super().__init__(parent)
        self.pool = pool
        self.job: Job | None = None

    @property
    def busy(self) -> bool:
        return self.job is not None

    def start(self, operation) -> None:
        if self.busy:
            raise RuntimeError("已有后台任务正在执行。")
        self.job = Job(operation)
        self.job.signals.complete.connect(self._complete, Qt.ConnectionType.QueuedConnection)
        self.job.signals.progress.connect(self.progress, Qt.ConnectionType.QueuedConnection)
        self.busy_changed.emit(True)
        self.pool.start(self.job)

    def stop(self) -> None:
        if self.job:
            self.job.cancel.set()

    @Slot(object, object)
    def _complete(self, value, error) -> None:
        self.job = None
        self.complete.emit(value, error)
        if not self.busy:
            self.busy_changed.emit(False)


class LatestTask(QObject):
    """Keep at most one active and one replacement task per visual surface."""

    complete = Signal(object, object)

    def __init__(self, pool: QThreadPool, parent: QObject | None = None):
        super().__init__(parent)
        self.session = WorkSession(pool, self)
        self.session.complete.connect(self._complete)
        self.generation = 0
        self.active = 0
        self.pending = None

    def invalidate(self) -> None:
        self.generation += 1
        self.pending = None
        self.session.stop()

    def submit(self, operation) -> None:
        self.invalidate()
        self.pending = operation
        self._start_pending()

    def _start_pending(self) -> None:
        if self.pending is not None and not self.session.busy:
            operation, self.pending = self.pending, None
            self.active = self.generation
            self.session.start(operation)

    @Slot(object, object)
    def _complete(self, value, error) -> None:
        if self.active == self.generation:
            self.complete.emit(value, error)
        self._start_pending()
