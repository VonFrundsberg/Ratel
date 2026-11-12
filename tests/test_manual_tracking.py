"""Focused checks for manual source-frame alignment, resume and pen input."""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from behaviour.manual_tracking import ManualTrack, run
from behaviour.trajectory import load_trajectory


class ManualTrackingTests(unittest.TestCase):
    def test_source_frames_resume_missing_and_undo(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "source.avi"
            video.write_bytes(b"synthetic identity fixture")
            track = ManualTrack(video, 10.0, 20, 100, 60, 0.25, 0.5)
            np.testing.assert_array_equal(track.frames, [3, 4, 5, 6, 7])
            track.mark(0, (10, 20), "Head")
            track.mark(1, (12, 20), "Tail")
            track.end()
            track.mark(2)
            track.end()
            output = Path(folder) / "track.csv"
            track.save(output)
            restored = ManualTrack(video, 10.0, 20, 100, 60, 0.25, 0.5)
            restored.load(output)
            np.testing.assert_allclose(restored.xy, track.xy, equal_nan=True)
            self.assertEqual(list(restored.status), ["manual", "manual", "missing", "unscored", "unscored"])
            self.assertEqual(list(restored.point_labels), ["Head", "Tail", "", "", ""])
            restored.mark(0, (11, 20), "Nose")
            restored.end()
            restored.undo()
            self.assertEqual(restored.point_labels[0], "Head")
            legacy = Path(folder) / "legacy.csv"
            pd.read_csv(output).drop(columns="tracking_point").to_csv(legacy, index=False)
            restored.load(legacy)
            self.assertEqual(list(restored.point_labels), ["Unspecified", "Unspecified", "", "", ""])
            restored.mark(3, (15, 20), "001")
            restored.mark(4, (16, 20), "NA")
            restored.end()
            labelled = Path(folder) / "labelled.csv"
            restored.save(labelled)
            restored.load(labelled)
            self.assertEqual(list(restored.point_labels[-2:]), ["001", "NA"])
            table = load_trajectory(output)
            np.testing.assert_allclose(table.time_s, table.frame / 10)
            self.assertTrue(table.x_px.iloc[2:].isna().all())
            self.assertEqual(track.undo(), 2)
            self.assertEqual(track.status[2], "unscored")
            self.assertEqual(track.undo(), 0)
            self.assertTrue(np.isnan(track.xy).all())
            self.assertTrue(all(label == "" for label in track.point_labels))
            with self.assertRaises(ValueError):
                track.mark(0, (100, 5))
            with self.assertRaises(ValueError):
                ManualTrack(video, 10.0, 20, 100, 60, 0.3, 0.4).load(output)
            video.write_bytes(b"changed source")
            with self.assertRaises(ValueError):
                ManualTrack(video, 10.0, 20, 100, 60, 0.25, 0.5).load(output)

    def test_viewer_tablet_scaling_playback_and_save(self):
        from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
            app.setFont(QtGui.QFont("Segoe UI", 10))
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 60))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                writer.write(np.full((60, 100, 3), i * 20, dtype=np.uint8))
            writer.release()
            output = Path(folder) / "manual.csv"
            failures = []

            def exercise():
                window = next(w for w in app.topLevelWidgets() if w.windowTitle().startswith("RATEL"))
                try:
                    window.resize(950, 750)
                    app.processEvents()
                    canvas = window.canvas
                    rect = canvas.image_rect()
                    pos = QtCore.QPointF(rect.x() + rect.width() * 0.3, rect.y() + rect.height() * 0.4)
                    np.testing.assert_allclose(canvas.source_point(pos), [30, 24])
                    self.assertIsNone(canvas.source_point(QtCore.QPointF(-1, -1)))
                    QtTest.QTest.mouseClick(canvas, QtCore.Qt.LeftButton, pos=pos.toPoint())
                    self.assertIn("manual", window.info.text())
                    window.undo()

                    def pen(kind, pressure, buttons):
                        event = QtGui.QTabletEvent(kind, pos, pos, QtGui.QTabletEvent.Stylus,
                                                   QtGui.QTabletEvent.Pen, pressure, 0, 0, 0, 0, 0,
                                                   QtCore.Qt.NoModifier, 1, QtCore.Qt.LeftButton, buttons)
                        app.sendEvent(canvas, event)
                        self.assertTrue(event.isAccepted())

                    pen(QtCore.QEvent.TabletPress, 0.5, QtCore.Qt.LeftButton)
                    window.tick()  # source frame 1, still in contact
                    pen(QtCore.QEvent.TabletRelease, 0, QtCore.Qt.NoButton)
                    window.tick()  # source frame 2 must stay unscored
                    window.step(1)
                    window.missing()  # source frame 3 is explicitly missing
                    self.assertTrue(window.save())
                    table = pd.read_csv(output)
                    self.assertEqual(list(table.tracking_status), ["manual", "manual", "unscored", "missing", "unscored", "unscored"])
                    np.testing.assert_allclose(table.x_px.iloc[:2], [30, 30])
                    np.testing.assert_allclose(table.y_px.iloc[:2], [24, 24])
                    self.assertTrue(table.x_px.iloc[2:].isna().all())
                    np.testing.assert_allclose(table.time_s, np.arange(6) / 10)
                    window.undo()
                    self.assertEqual(window.index, 3)
                    window.undo()
                    self.assertEqual(window.index, 0)
                    window.refresh()
                    app.processEvents()
                    if os.environ.get("RATEL_SMOKE_PREVIEW"):
                        window.grab().save(os.environ["RATEL_SMOKE_PREVIEW"])
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    window.close()
                    app.quit()

            QtCore.QTimer.singleShot(0, exercise)
            run(video, output)
            if failures:
                raise failures[0]
            saved = pd.read_csv(output)
            self.assertTrue(saved.tracking_status.eq("unscored").all())


if __name__ == "__main__":
    unittest.main()
