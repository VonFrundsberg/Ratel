"""Manual body-centre trajectories, with explicit missing frames and resumable CSVs."""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class ManualTrack:
    def __init__(self, video, fps, count, width, height, start_s=0.0, duration_s=None):
        self.video = Path(video).resolve()
        self.fps, self.width, self.height = fps, width, height
        if not math.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
            raise ValueError("Video needs a positive frame rate, frame count and dimensions")
        if not math.isfinite(start_s) or start_s < 0 or (duration_s is not None and
                (not math.isfinite(duration_s) or duration_s <= 0)):
            raise ValueError("Scoring window needs a nonnegative start and positive duration")
        start = math.ceil(start_s * fps - 1e-9)
        end = count if duration_s is None else min(count, math.ceil((start_s + duration_s) * fps - 1e-9))
        if start >= end:
            raise ValueError("Scoring window contains no video frames")
        self.frames = np.arange(start, end)
        self.xy = np.full((len(self.frames), 2), np.nan)
        self.status = np.full(len(self.frames), "unscored", dtype=object)
        self.point_labels = np.full(len(self.frames), "", dtype=object)
        stat = self.video.stat()
        identity = f"{self.video}|{stat.st_size}|{stat.st_mtime_ns}|{fps}|{count}|{width}|{height}"
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.history = []
        self.stroke = None
        self.dirty = False

    def begin(self):
        self.end()
        self.stroke = {}

    def end(self):
        if self.stroke:
            self.history.append(self.stroke)
            self.history = self.history[-100:]
        self.stroke = None

    def mark(self, i, point=None, tracking_point="Body centre"):
        if point is not None:
            x, y = point
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < self.width and 0 <= y < self.height):
                raise ValueError("Position must be inside the source video")
            tracking_point = str(tracking_point).strip()
            if not tracking_point or len(tracking_point) > 100:
                raise ValueError("Tracked point needs a label of 1 to 100 characters")
        if self.stroke is None:
            self.begin()
        if i not in self.stroke:
            self.stroke[i] = (self.xy[i].copy(), self.status[i], self.point_labels[i])
        self.xy[i] = (np.nan, np.nan) if point is None else point
        self.status[i] = "missing" if point is None else "manual"
        self.point_labels[i] = "" if point is None else tracking_point
        self.dirty = True

    def undo(self):
        self.end()
        if self.history:
            changes = self.history.pop()
            for i, (xy, status, label) in changes.items():
                self.xy[i], self.status[i], self.point_labels[i] = xy, status, label
            self.dirty = True
            return min(changes)

    def save(self, path):
        path = Path(path)
        if path.resolve() == self.video:
            raise ValueError("Output cannot overwrite the video")
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(dict(frame=self.frames, time_s=self.frames / self.fps,
                                  x_px=self.xy[:, 0], y_px=self.xy[:, 1],
                                  tracking_status=self.status, tracking_point=self.point_labels,
                                  source_fingerprint=self.identity))
        temp = path.with_name(path.name + ".tmp")
        table.to_csv(temp, index=False)
        os.replace(temp, path)
        self.dirty = False

    def load(self, path):
        table = pd.read_csv(path, dtype={"tracking_point": str}, keep_default_na=False,
                            na_values={"x_px": [""], "y_px": [""]})
        required = {"frame", "time_s", "x_px", "y_px", "tracking_status", "source_fingerprint"}
        if not required.issubset(table) or len(table) != len(self.frames):
            raise ValueError("Resume CSV is not a manual track for this scoring window")
        if not (table.source_fingerprint == self.identity).all() or not np.array_equal(table.frame, self.frames):
            raise ValueError("Resume CSV belongs to a different video or scoring window")
        if not np.allclose(table.time_s, self.frames / self.fps, atol=1e-8, rtol=0):
            raise ValueError("Resume timestamps do not match the source video")
        xy = table[["x_px", "y_px"]].to_numpy(float)
        manual = table.tracking_status.eq("manual").to_numpy()
        valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < self.width) & (xy[:, 1] >= 0) & (xy[:, 1] < self.height)
        if not table.tracking_status.isin(["manual", "missing", "unscored"]).all() or not np.array_equal(manual, valid) or not np.isnan(xy[~manual]).all():
            raise ValueError("Resume CSV contains inconsistent positions or scoring states")
        if "tracking_point" in table:
            labels = table.tracking_point.astype(str).str.strip().to_numpy(object)
            if any(not label or len(label) > 100 for label in labels[manual]) or any(labels[~manual]):
                raise ValueError("Resume CSV contains inconsistent tracked-point labels")
        else:
            labels = np.where(manual, "Unspecified", "").astype(object)
        self.xy, self.status = xy, table.tracking_status.to_numpy(object)
        self.point_labels = labels
        self.history, self.stroke, self.dirty = [], None, False


def run(video, output, cfg=None, speed=0.25):
    """Open the optional Qt viewer; no video is modified or re-encoded."""
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise SystemExit('Manual tracking requires PyQt5. Use the .venv-oft environment or install ".[oft]".') from exc
    if speed not in (0.1, 0.25, 0.5, 1.0):
        raise ValueError("Playback speed must be 0.1, 0.25, 0.5 or 1.0")
    output = Path(output)
    if output.suffix.lower() != ".csv":
        raise ValueError("Manual tracking output must be a .csv file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_lock = QtCore.QLockFile(str(output.resolve()) + ".lock")
    if not output_lock.tryLock(0):
        raise ValueError("This output is already open in another manual-tracking viewer")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count, width, height = [int(cap.get(prop)) for prop in
                                (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        session = cfg.session(Path(video).stem) if cfg else None
        track = ManualTrack(video, fps, count, width, height,
                            session.start_s if session else 0.0, session.duration_s if session else None)
        if output.exists():
            track.load(output)

        class Canvas(QtWidgets.QWidget):
            def __init__(self, owner):
                super().__init__()
                self.owner, self.picture = owner, None
                self.setMinimumSize(480, 300)
                self.setFocusPolicy(QtCore.Qt.StrongFocus)
                self.setMouseTracking(True)
                self.setAttribute(QtCore.Qt.WA_TabletTracking)

            def image_rect(self):
                scale = min(self.width() / width, self.height() / height)
                return QtCore.QRectF((self.width() - width * scale) / 2,
                                     (self.height() - height * scale) / 2, width * scale, height * scale)

            def source_point(self, pos):
                rect = self.image_rect()
                if not rect.contains(pos):
                    return None
                x, y = (pos.x() - rect.x()) * width / rect.width(), (pos.y() - rect.y()) * height / rect.height()
                return (x, y) if 0 <= x < width and 0 <= y < height else None

            def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                painter.fillRect(self.rect(), QtGui.QColor("#151b21"))
                rect = self.image_rect()
                if self.picture is not None:
                    painter.drawImage(rect, self.picture)
                xy = track.xy[self.owner.index]
                if np.isfinite(xy).all():
                    x, y = rect.x() + xy[0] * rect.width() / width, rect.y() + xy[1] * rect.height() / height
                    painter.setPen(QtGui.QPen(QtGui.QColor("#34edc4"), 2))
                    painter.drawEllipse(QtCore.QPointF(x, y), 7, 7)
                    painter.drawLine(QtCore.QPointF(x - 12, y), QtCore.QPointF(x + 12, y))
                    painter.drawLine(QtCore.QPointF(x, y - 12), QtCore.QPointF(x, y + 12))

            def pointer(self, pos, pressed=False, released=False):
                if released:
                    self.owner.release()
                    return
                point = self.source_point(pos)
                if pressed and point is not None:
                    self.setFocus()
                    track.begin()
                    self.owner.contact = True
                if self.owner.contact:
                    self.owner.point = point
                    if point is not None:
                        track.mark(self.owner.index, point, self.owner.tracking_label())
                        self.owner.refresh()

            def tabletEvent(self, event):
                # Accept tablet events so Qt does not also synthesize mouse marks.
                self.pointer(event.posF(), event.type() == QtCore.QEvent.TabletPress,
                             event.type() == QtCore.QEvent.TabletRelease)
                event.accept()

            def mousePressEvent(self, event):
                if event.button() == QtCore.Qt.LeftButton:
                    self.pointer(event.localPos(), pressed=True)

            def mouseMoveEvent(self, event):
                self.pointer(event.localPos())

            def mouseReleaseEvent(self, event):
                if event.button() == QtCore.Qt.LeftButton:
                    self.pointer(event.localPos(), released=True)

            def leaveEvent(self, event):
                self.owner.release()

        class Window(QtWidgets.QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle(f"RATEL · Manual tracking · {Path(video).name}")
                self.index, self.contact, self.point = 0, False, None
                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self.tick)
                self.autosave = QtCore.QTimer(self)
                self.autosave.timeout.connect(lambda: self.save() if track.dirty else None)
                self.autosave.start(30000)
                body = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(body)
                layout.addWidget(QtWidgets.QLabel("Hold the pen on the selected tracking point during playback. Tap when paused. Lift to stop marking."))
                point_controls = QtWidgets.QHBoxLayout()
                point_controls.addWidget(QtWidgets.QLabel("Tracked point:"))
                self.tracked_point = QtWidgets.QComboBox()
                self.tracked_point.setEditable(True)
                self.tracked_point.addItems(["Body centre", "Head", "Tail", "Tail base", "Tail tip", "Nose", "Unspecified"])
                self.tracked_point.lineEdit().setMaxLength(100)
                labelled = np.flatnonzero(track.status == "manual")
                if len(labelled):
                    self.tracked_point.setCurrentText(str(track.point_labels[labelled[-1]]))
                point_controls.addWidget(self.tracked_point)
                point_controls.addWidget(QtWidgets.QLabel("Choose or type a label. Applies to new marks only."))
                point_controls.addStretch()
                layout.addLayout(point_controls)
                self.canvas = Canvas(self)
                layout.addWidget(self.canvas, 1)
                self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                self.slider.setRange(0, len(track.frames) - 1)
                self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
                self.slider.valueChanged.connect(self.seek)
                layout.addWidget(self.slider)
                controls = QtWidgets.QHBoxLayout()
                self.play = QtWidgets.QPushButton("Play (Space)")
                buttons = [(self.play, self.toggle), (QtWidgets.QPushButton("Previous (←)"), lambda: self.step(-1)),
                           (QtWidgets.QPushButton("Next (→)"), lambda: self.step(1)),
                           (QtWidgets.QPushButton("Missing + next (N)"), self.missing),
                           (QtWidgets.QPushButton("Undo (Ctrl+Z)"), self.undo),
                           (QtWidgets.QPushButton("Save (Ctrl+S)"), self.save)]
                for button, callback in buttons:
                    button.setFocusPolicy(QtCore.Qt.NoFocus)
                    button.clicked.connect(callback)
                    controls.addWidget(button)
                self.speed = QtWidgets.QComboBox()
                for value in (0.1, 0.25, 0.5, 1.0):
                    self.speed.addItem(f"{value:g}×", value)
                self.speed.setCurrentIndex((0.1, 0.25, 0.5, 1.0).index(speed))
                self.speed.currentIndexChanged.connect(self.interval)
                controls.addWidget(self.speed)
                layout.addLayout(controls)
                self.info = QtWidgets.QLabel()
                layout.addWidget(self.info)
                layout.addWidget(QtWidgets.QLabel("Space: play/pause · Arrows: one frame · N: missing + next · Ctrl+Z: undo stroke · Ctrl+S: save"))
                self.setCentralWidget(body)
                self.shortcuts = []
                for key, callback in [("Space", self.toggle), ("Left", lambda: self.step(-1)),
                                      ("Right", lambda: self.step(1)), ("N", self.missing),
                                      ("Ctrl+Z", self.undo), ("Ctrl+S", self.save)]:
                    # Space/arrows/N must remain available while typing a custom label.
                    text_key = key in {"Space", "Left", "Right", "N"}
                    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self.canvas if text_key else self)
                    if text_key:
                        shortcut.setContext(QtCore.Qt.WidgetShortcut)
                    shortcut.activated.connect(callback)
                    self.shortcuts.append(shortcut)
                self.interval()
                pending = np.flatnonzero(track.status == "unscored")
                self.show_frame(int(pending[0]) if len(pending) else 0)
                self.canvas.setFocus()
                self.resize(1100, 800)

            def interval(self):
                self.timer.setInterval(max(1, round(1000 / (fps * self.speed.currentData()))))

            def tracking_label(self):
                return self.tracked_point.currentText().strip() or "Unspecified"

            def release(self):
                self.contact, self.point = False, None
                track.end()

            def pause(self):
                self.timer.stop()
                self.play.setText("Play (Space)")
                self.release()

            def toggle(self):
                if self.timer.isActive():
                    self.pause()
                elif self.index < len(track.frames) - 1:
                    self.timer.start()
                    self.play.setText("Pause (Space)")

            def show_frame(self, i):
                target = int(track.frames[i])
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != target:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != target + 1:
                    self.pause()
                    QtWidgets.QMessageBox.warning(self, "Video read failed", f"Cannot read source frame {target}. No position recorded.")
                    return False
                self.index = i
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.canvas.picture = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)
                self.refresh()
                return True

            def refresh(self):
                marked = np.count_nonzero(track.status == "manual")
                missing = np.count_nonzero(track.status == "missing")
                self.info.setText(f"Frame {track.frames[self.index]} · {track.frames[self.index] / fps:.3f} s · "
                                  f"{track.status[self.index]}" +
                                  (f" ({track.point_labels[self.index]})" if track.point_labels[self.index] else "") +
                                  f" · {marked} marked / {missing} missing / "
                                  f"{len(track.frames) - marked - missing} unscored" + (" · unsaved" if track.dirty else ""))
                self.canvas.update()

            def tick(self):
                if self.index >= len(track.frames) - 1:
                    self.pause()
                    return
                if self.show_frame(self.index + 1) and self.contact and self.point is not None:
                    track.mark(self.index, self.point, self.tracking_label())
                    self.refresh()

            def seek(self, i):
                self.pause()
                self.show_frame(i)

            def step(self, delta):
                self.seek(max(0, min(len(track.frames) - 1, self.index + delta)))

            def missing(self):
                self.pause()
                track.mark(self.index)
                track.end()
                self.step(1)

            def undo(self):
                self.pause()
                i = track.undo()
                if i is not None:
                    self.show_frame(i)

            def save(self):
                try:
                    track.save(output)
                except OSError as exc:
                    self.pause()
                    QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
                    return False
                self.statusBar().showMessage(f"Saved {output}", 5000)
                self.refresh()
                return True

            def changeEvent(self, event):
                if event.type() == QtCore.QEvent.ActivationChange and not self.isActiveWindow():
                    self.pause()
                super().changeEvent(event)

            def closeEvent(self, event):
                self.pause()
                if track.dirty and not self.save():
                    event.ignore()
                else:
                    event.accept()

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        window = Window()
        window.show()
        app.exec_()
    finally:
        cap.release()
        output_lock.unlock()
