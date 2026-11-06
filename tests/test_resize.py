"""Run the native CUDA regression suite through pytest when pytest is available."""
import os
from pathlib import Path
import subprocess


def test_cuda_resize():
    """Check device copy, batched resize, CPU reference and preprocessing allocation."""
    root = Path(__file__).resolve().parents[1]
    config = os.environ.get("CUIMG_BUILD_CONFIG", "release")
    executable = root / "build" / config / "resize_test.exe"
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_real_file_batches():
    """Verify distinct files, cached input, changed output, reload and error recovery."""
    root = Path(__file__).resolve().parents[1]
    config = os.environ.get("CUIMG_BUILD_CONFIG", "release")
    env = os.environ.copy()
    qt = root / "externals" / "Qt" / "6.8.3" / "msvc2022_64"
    env["PATH"] = str(qt / "bin") + os.pathsep + env["PATH"]
    env["QT_PLUGIN_PATH"] = str(qt / "plugins")
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [str(root / "cuda.exe"), "--batch-test", str(root / "build" / config)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
