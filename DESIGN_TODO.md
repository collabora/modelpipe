# DESIGN_TODO

A terse catalogue of open tasks only. Gaps whose fix sits in glass2glass are
tracked in that repo's `design/TODO.md`, not here.

## GPU effects and device paths

- **Feed `mp_overlay` from the Vulkan decoders.** `vulkanh264dec !
  vulkancolorconvert` crashes on GStreamer 1.26 and 1.28 (upstream #4623). The
  decoder emits NV12 as one multi-planar image, and `calculate_reorder_indexes`
  reads a view for each plane. The upstream fix is per-plane views of that
  image. Until then, decoded frames reach `mp_overlay` only through
  `vulkandownload`.

- **Sync `mp_cudatovulkan` on the GPU.** It waits on the host for each copy
  with `cuStreamSynchronize`. Export a Vulkan semaphore to CUDA and signal it
  after the copy instead.

- **Route `mp_overlay` to the GPU overlay on glass2glass.** `mp-launch` maps
  `mp_overlay` to `analyticsoverlay`. For a CUDA decode, emit `cudatowgpu !
  wgpucolorconvert ! vellooverlay` instead, once glass2glass registers those
  elements for launch lines.

- **Finish `mp_anonymize` validation on a CUDA host.** Run the GStreamer and
  glass2glass CUDA tests and the documented detector pipeline under both
  backends. Do not describe the face pipeline as fully GPU-resident while
  `FaceEngine` passes NumPy frames to InsightFace.

- **Measure the complete path before adding another GPU backend.** Benchmark
  decode, conversion, inference, anonymization and encode together. Record the
  negotiated memory type and format at every video pad, count frame uploads and
  downloads, and record explicit stream synchronizations. Add a regression that
  fails when the tested CUDA pipeline maps frame pixels into system memory.

- **Add an `mp-launch` device-path report.** Read the typed frame capabilities
  and negotiated caps. Report memory types, formats, frame uploads and
  downloads, and explicit stream synchronizations for every video link. Refuse
  an unsupported path instead of inserting an implicit download. Keep
  scheduling in GStreamer and glass2glass.
