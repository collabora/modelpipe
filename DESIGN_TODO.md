# DESIGN_TODO

A terse catalogue of open tasks only. Gaps whose fix sits in glass2glass are
tracked in that repo's `design/TODO.md`, not here.

## GPU effects and device paths

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

- **Replace the incomplete Vulkan overlay with packaged shader assets.** Stop
  advertising `VulkanMemory` until an integration test proves that pixels are
  drawn. Evaluate Slang against direct SPIR-V and the available GStreamer shader
  elements before adding a dependency. If Slang is selected, keep the source in
  the repository, compile and validate SPIR-V during the build, ship the same
  binary tested in CI, and add DXIL only with a tested D3D12 memory path. Reuse
  the shader layer for anonymization and overlay drawing. Runtime user shaders
  and a shader editor are outside the first implementation.
