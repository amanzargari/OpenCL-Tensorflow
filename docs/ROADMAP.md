# Roadmap

The end goal is a fully OpenCL-backed training and inference pipeline for
this Keras model:

```python
def build_model():
    inp = layers.Input(shape=(T_CTX, 6, 3, N_BINS_KEPT, 2), name='stacked_iq')
    x = layers.Permute((1, 4, 2, 3, 5), name='to_tcrange')(inp)
    x = layers.Reshape((T_CTX, N_BINS_KEPT, 36), name='merge_sensors')(x)

    x = conv_bn_relu(x, 48,  (3, 7), strides=(1, 2), name='stem')
    x = dsconv_block(x, 96,  strides=(2, 2), name='b1')
    x = dsconv_block(x, 144, strides=(2, 2), name='b2')
    x = dsconv_block(x, 192, strides=(2, 2), name='b3')

    x = conv_bn_relu(x, 64, (1, 1), name='neck')

    trunk = layers.Flatten(name='flatten')(x)

    h = layers.Dense(9 * 6 * 8, activation='relu', name='hm_dense')(trunk)
    h = layers.Reshape((9, 6, 8), name='hm_reshape')(h)
    h = layers.UpSampling2D((2, 2), interpolation='bilinear', name='hm_up1')(h)
    h = conv_bn_relu(h, 16, (3, 3), name='hm_refine1')
    h = layers.UpSampling2D((2, 2), interpolation='bilinear', name='hm_up2')(h)
    h = conv_bn_relu(h, 16, (3, 3), name='hm_refine2')
    heatmap = layers.Conv2D(1, (1, 1), activation='sigmoid', name='heatmap')(h)
    return Model(inp, heatmap, name='uwb_loc_v3')
```

---

## Phase 1 — Conv2D ✅

- [x] Conv2D forward
- [x] Conv2D dL/dx
- [x] Conv2D dL/dw
- [x] Gradient registration through `tf.GradientTape`
- [x] Keras `OpenCLConv2D` wrapper
- [x] Correctness tests vs `tf.nn.conv2d` (SAME/VALID, stride 1 & 2)

## Phase 2 — `dsconv_block` and `conv_bn_relu` end-to-end ✅

- [x] **DepthwiseConv2D** — forward, dL/dx, dL/dw. Filter
  `[kH, kW, C, depth_multiplier]`. Tested across SAME/VALID, stride 1 & 2,
  depth_multiplier ∈ {1, 2}.
- [x] **BatchNormalization** — train + inference forward, plus the
  three-term backward (∂γ, ∂β, ∂x). Two reduction passes (`bn_reduce_stats`
  and `bn_backward_reduce`) and two elementwise passes (`bn_normalize`,
  `bn_backward_dx`). Keras layer maintains moving stats via EMA.
- [x] **ReLU** — elementwise forward and backward. Backward uses
  TF's `(gradients, features)` convention.

**Milestone hit:** stem + b1 + b2 + b3 + neck portion of the target model
trains end-to-end on the OpenCL backend.

## Phase 3 — Head and upsampling ✅

- [x] **Dense** — standalone GEMM kernel (one work-item per output element).
  Forward + BackpropInput + BackpropWeight + BackpropBias. 13 tests pass.
- [x] **UpSampling2D (bilinear)** — half-pixel-centre mapping matches
  `tf.image.resize` default. Backward uses float `atomic_add` via uint CAS
  loop (OpenCL 1.2 portable) and `clEnqueueFillBuffer` for zero-init.
  14 tests pass; parity with `tf.keras.layers.UpSampling2D` atol=1e-4.
- [x] **Sigmoid** — elementwise forward + backward. Backward takes the
  forward output y (not x) to avoid recomputing sigmoid. 10 tests pass.

**Milestone hit:** the entire `build_model()` graph runs forward and trains
on the OpenCL backend. See [`examples/train_full_model.py`](../examples/train_full_model.py).

## Phase 4 — Performance

Naive direct convolution and per-channel serial reductions will leave a
lot on the table. Order of attack:

- [ ] **Buffer pool** keyed by `(size, role)` on `CLBackend`. Eliminates
  per-call `cl_mem` alloc/release in the hot path.
- [ ] **Persistent host-pinned staging** with `clEnqueueMapBuffer` so the
  H↔D copies are zero-copy when the runtime supports it.
- [ ] **Tree-reduction kernels** for BN stats (one work-group per channel,
  local-memory accumulator) to replace the current "one work-item per
  channel" pattern.
- [ ] **Tiled / vectorized Conv2D** kernel (`float4` loads, LDS staging,
  one work-group per output tile). Target: ≥ 50% of device peak GFLOPS
  for the `(3×3, C=144)` block.
- [ ] **Winograd F(2×2, 3×3)** for the 3×3 conv layers. Optional;
  evaluate after the tiled kernel.
- [ ] **Asynchronous queues / overlap** of H→D copies with compute on the
  next op. Requires a queue per inter-op thread and per-tensor events.

Milestone: full training step within 2× of the reference time for the
same problem size on the same hardware via Mesa's OpenCL.

## Phase 5 — Polish

- [ ] CI on GitHub Actions with `pocl` as a software ICD (wired up in
  `.github/workflows/build.yml`).
- [ ] Optional `pip install .` packaging once the API is stable.
- [ ] FP16 path for inference (forward only, post-training).
