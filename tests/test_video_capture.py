"""
Unit tests for deep-live-cam video capture module.

Run with:
    cd ~/Desktop/deep-live-cam
    .venv/bin/python -m pytest tests/ -v

NOTE: Camera-hardware tests are skipped if no camera permission is granted.
"""

import sys
import platform
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_cap(opened=True, ret=True, frame=None):
    """Return a mock cv2.VideoCapture-like object."""
    if frame is None:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (ret, frame if ret else None)
    return cap


# ---------------------------------------------------------------------------
# Tests: VideoCapturer class
# ---------------------------------------------------------------------------

class TestVideoCapturerInit(unittest.TestCase):
    """Test VideoCapturer initialisation."""

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    def test_init_macos_does_not_import_pygrabber(self, _mock_sys):
        """On macOS, pygrabber must never be imported (Windows-only lib)."""
        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        self.assertEqual(vc.device_index, 0)
        self.assertFalse(vc.is_running)
        self.assertIsNone(vc.cap)

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    def test_init_stores_device_index(self, _):
        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(2)
        self.assertEqual(vc.device_index, 2)


class TestVideoCapturerStart(unittest.TestCase):
    """Test VideoCapturer.start() on macOS path."""

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    @patch("modules.video_capture.cv2.VideoCapture")
    def test_start_success(self, mock_vc_cls, _mock_sys):
        mock_cap = make_mock_cap(opened=True)
        mock_vc_cls.return_value = mock_cap

        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        result = vc.start()

        self.assertTrue(result)
        self.assertTrue(vc.is_running)
        # Should configure resolution/fps
        mock_cap.set.assert_called()

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    @patch("modules.video_capture.cv2.VideoCapture")
    def test_start_camera_not_opened(self, mock_vc_cls, _mock_sys):
        """start() returns False if camera can't be opened."""
        mock_cap = make_mock_cap(opened=False)
        mock_vc_cls.return_value = mock_cap

        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        result = vc.start()

        self.assertFalse(result)
        self.assertFalse(vc.is_running)

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    @patch("modules.video_capture.cv2.VideoCapture")
    def test_start_exception_returns_false(self, mock_vc_cls, _mock_sys):
        """start() returns False (doesn't raise) if VideoCapture raises."""
        mock_vc_cls.side_effect = RuntimeError("camera exploded")

        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        result = vc.start()

        self.assertFalse(result)


class TestVideoCapturerRead(unittest.TestCase):
    """Test VideoCapturer.read()."""

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    @patch("modules.video_capture.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_vc_cls, _mock_sys):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        mock_cap = make_mock_cap(opened=True, ret=True, frame=frame)
        mock_vc_cls.return_value = mock_cap

        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        vc.start()
        ok, got_frame = vc.read()

        self.assertTrue(ok)
        self.assertIsNotNone(got_frame)

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    def test_read_not_running_returns_false(self, _mock_sys):
        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        # Don't start — cap is None
        ok, frame = vc.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    @patch("modules.video_capture.cv2.VideoCapture")
    def test_read_invokes_frame_callback(self, mock_vc_cls, _mock_sys):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        mock_cap = make_mock_cap(opened=True, ret=True, frame=frame)
        mock_vc_cls.return_value = mock_cap

        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        vc.start()

        callback = MagicMock()
        vc.set_frame_callback(callback)
        vc.read()

        callback.assert_called_once()


class TestVideoCapturerRelease(unittest.TestCase):

    @patch("modules.video_capture.platform.system", return_value="Darwin")
    @patch("modules.video_capture.cv2.VideoCapture")
    def test_release_clears_state(self, mock_vc_cls, _mock_sys):
        mock_cap = make_mock_cap(opened=True)
        mock_vc_cls.return_value = mock_cap

        from modules.video_capture import VideoCapturer
        vc = VideoCapturer(0)
        vc.start()
        self.assertTrue(vc.is_running)

        vc.release()
        self.assertFalse(vc.is_running)
        self.assertIsNone(vc.cap)
        mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: get_available_cameras (macOS path)
# ---------------------------------------------------------------------------

class TestGetAvailableCamerasMacOS(unittest.TestCase):

    @patch("modules.ui.platform.system", return_value="Darwin")
    def test_falls_back_when_cv2_enumerate_not_installed(self, _):
        """When cv2_enumerate_cameras is absent, fallback to [0,1]."""
        with patch.dict("sys.modules", {"cv2_enumerate_cameras": None}):
            from importlib import reload
            import modules.ui as ui_mod
            # Patch the inner import inside get_available_cameras
            with patch("builtins.__import__", side_effect=ImportError):
                indices, names = ui_mod.get_available_cameras()
        # Should either return cameras or "No cameras found" — never crash
        self.assertIsInstance(indices, list)
        self.assertIsInstance(names, list)
        self.assertTrue(len(names) >= 1)

    @patch("modules.ui.platform.system", return_value="Darwin")
    def test_returns_no_cameras_when_enumeration_empty(self, _):
        """Empty device list returns ['No cameras found']."""
        fake_module = MagicMock()
        fake_module.enumerate_cameras.return_value = []
        fake_module.CAP_AVFOUNDATION = 1200

        with patch.dict("sys.modules", {"cv2_enumerate_cameras": fake_module}):
            from importlib import reload
            import modules.ui as ui_mod
            indices, names = ui_mod.get_available_cameras()

        self.assertEqual(names, ["No cameras found"])
        self.assertEqual(indices, [])

    @patch("modules.ui.platform.system", return_value="Darwin")
    def test_returns_correct_camera_list(self, _):
        """Correctly maps device index and name from enumeration."""
        dev0 = MagicMock()
        dev0.index = 0
        dev0.name = "FaceTime HD Camera"
        dev1 = MagicMock()
        dev1.index = 1
        dev1.name = "OBS Virtual Camera"

        fake_module = MagicMock()
        fake_module.enumerate_cameras.return_value = [dev0, dev1]
        fake_module.CAP_AVFOUNDATION = 1200

        with patch.dict("sys.modules", {"cv2_enumerate_cameras": fake_module}):
            import modules.ui as ui_mod
            indices, names = ui_mod.get_available_cameras()

        self.assertEqual(indices, [0, 1])
        self.assertIn("FaceTime HD Camera", names)
        self.assertIn("OBS Virtual Camera", names)


# ---------------------------------------------------------------------------
# Integration smoke test (skipped if camera not authorised)
# ---------------------------------------------------------------------------

@unittest.skipUnless(platform.system() == "Darwin", "macOS only")
class TestCameraPermissionSmoke(unittest.TestCase):

    def test_camera_authorization_status(self):
        """
        Smoke test: checks whether the current Python process is authorised
        to use the camera.  This test PASSES even when the camera is denied —
        it just prints the status so CI can see it.

        ACTUAL FIX NEEDED: The venv python3.13 binary is NOT code-signed,
        so macOS refuses to remember camera permission for it.
        See CAMERA_FIX.md for the fix.
        """
        import cv2
        cap = cv2.VideoCapture(0)
        authorized = cap.isOpened()
        cap.release()
        if not authorized:
            self.skipTest(
                "Camera not authorized for this Python binary (unsigned binary). "
                "See CAMERA_FIX.md for fix."
            )
        self.assertTrue(authorized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
