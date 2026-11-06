# CUDA batch resize GUI

Run `make run release` from the project directory (GNU make must be on PATH).
The GUI executable is `cuda.exe` in the project root; generated files stay in `build/`.

- **Open batch images** selects up to 64 actual image files. All images must have
  equal width and height; the selected file count becomes the batch size.
- **5120 x 5120 patterns** generates distinct images with the chosen batch count.
- **Run on GPU** uploads a changed batch and processes it in one kernel launch.
  Repeated runs retain GPU input. Reselect files to reload changed file contents.
- The batch selector shows each image's matching original/output. Save result
  exports the selected output as PNG.
- GPU input/output allocations, pinned transfer buffers, stream and timing events
  are reused. Source/target shape changes replace the affected device buffers.
- Timings separate CPU preparation, H2D upload, resize and D2H download. Total also
  includes CPU result-image copies but excludes GUI rendering. Cached-input timing
  is not throughput for new images: new batches still need decoding and uploading.
- `preprocessYolo` mode currently performs bilinear resize only, not letterbox or inference.

`make gui-test release` verifies distinct real-file batches, cached repeats, output
shape changes, refreshed input, mixed-dimension rejection and recovery. It also
runs two 5120-to-2048 batch-3 GUI passes and saves the preview and timings under
`build/release`. `make test release` runs CUDA/CPU pixel comparisons.
Optional `make pytest release` requires pytest.
