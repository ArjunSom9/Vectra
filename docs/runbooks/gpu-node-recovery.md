# Runbook: GPU inference node recovery

**When:** `GpuXidErrors`, a gateway that will not start, or a GPU host
reboot.

1. **Health.** `nvidia-smi` must list the H100 on the driver of
   `deploy/gpu/driver_matrix.md` (R570), with persistence mode on and MIG
   disabled. XID 79 (fell off the bus) or repeated XID 48/63/64 (memory
   errors): drain the host and open a hardware ticket.
2. **Driver or toolkit drift.** If the driver changed, re-run
   `sudo deploy/gpu/h100_setup.sh` and reboot.
3. **Container.** Start the pinned image:
   ```bash
   docker run --gpus all -p 7000:7000 -p 9464:9464 \
     -v /srv/vectra/dist/compiled_model:/srv/vectra/dist/compiled_model:ro \
     vectra-inference:1.0.0
   ```
4. **The gateway refuses to start.** That is by design when the bundle
   cannot be trusted:
   * `does not match its manifest checksum`: the bundle is corrupt; copy it
     again from the release.
   * `toolchain ... differs from the export toolchain`: the image and the
     bundle were built with different versions. Re-export inside the
     serving image:
     ```bash
     docker run --gpus all --entrypoint python vectra-inference:1.0.0 \
       -m models.export.compile_entry --checkpoint_dir=... --out=...
     ```
   * `PatternNotFoundError` or `GraphBreakError`: the model and the
     compiler disagree. Do not serve eager; roll back to the previous
     release and run `make test-compiler`.
   * `NumericsError`: the compiled model left the rtol 1e-3 band. Roll back
     and run `make test-kernels` on the host.
5. **Verify.** One request through the gateway, then watch the inference
   panels and `GpuDramActivityDropped` for 15 minutes.
