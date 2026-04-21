# ABFT and AByzFT Options in MLonMCU

This is a quick reference for MLonMCU backend options when using IREE ABFT/AByzFT compiler modes.

## Scope

These options are read from `mlonmcu/flow/iree/backend/backend.py` and forwarded to the IREE plugin passes.

Use backend-prefixed config keys, for example:
- `ireellvmc.compiler_mode=abft`
- `ireellvmc.compiler_mode=abyzft`

## Compiler Mode

| Key | Type | Default | Allowed |
|---|---|---|---|
| `compiler_mode` | string | `baseline` | `baseline`, `abft`, `abyzft`, `freivald`, `freivalds` |

## ABFT Options

| Key | Type | Default | Notes |
|---|---|---|---|
| `abft_enable_fuc` | bool | `false` | Enables ABFT FUC mode (pass flag `--abft-enable-fuc`). |
| `abft_enable_analysis` | bool | `true` | Controls ABFT analysis logging insertion (pass flag `--abft-enable-analysis-log=false` when disabled). |
| `abft_enable_matmul_tiling` | bool | `true` | Enables transform-spec tiling wiring in backend compile pipeline. |
| `abft_inject_fault` | bool | `false` | Enables synthetic fault injection. |
| `abft_inject_fault_delta` | int | `1` | Additive fault delta. |
| `abft_inject_fault_layer` | int | `-1` | Target layer ordinal for fault injection (`-1` = inject all layers). |
| `abft_inject_fault_pattern` | string | `single_point` | `single_point`, `trivial`, `checkered`. |

## AByzFT Options

| Key | Type | Default | Notes |
|---|---|---|---|
| `abyzft_enable_analysis` | bool | `true` | Analysis toggle for AByzFT mode (mapped via shared ABFT analysis control path). |
| `abyzft_inject_fault` | bool | `false` | Enables synthetic fault injection. |
| `abyzft_inject_fault_delta` | int | `1` | Additive fault delta. |
| `abyzft_inject_fault_layer` | int | `-1` | Target layer ordinal for fault injection (`-1` = inject all layers). |
| `abyzft_inject_fault_pattern` | string | `single_point` | `single_point`, `trivial`, `checkered`. |
| `abyzft_scale_sampling_mode` | int | `1` | `1`, `2`, `3` (type-aware behavior, see below). |
| `abyzft_float_disjoint_min_abs` | float | `0.5` | Float mode 1 lower abs bound. |
| `abyzft_float_disjoint_max_abs` | float | `2.0` | Float mode 1 upper abs bound. |
| `abyzft_float_range_min` | float | `-2.0` | Float mode 2 lower bound. |
| `abyzft_float_range_max` | float | `2.0` | Float mode 2 upper bound. |
| `abyzft_float_discrete_list` | string | `-8,-4,-2,2,4,8` | Float mode 3 sample set (CSV). |
| `abyzft_int_discrete_list` | string | `1,2,4` | Int/uint mode 1 sample set (CSV). |
| `abyzft_int_range_min` | int | `1` | Int/uint mode 2 lower bound. |
| `abyzft_int_range_max` | int | `8` | Int/uint mode 2 upper bound. |
| `abyzft_int_bits_max` | int | `2` | Int/uint mode 3 samples in `[1, 2^bits-1]`. |

## AByzFT Scale Sampling Semantics

The same `abyzft_scale_sampling_mode` is interpreted by tensor type.

For floating-point tensors:
- Mode `1`: continuous random from `[-max_abs, -min_abs] U [min_abs, max_abs]`.
- Mode `2`: continuous random from `[range_min, range_max]`.
- Mode `3`: random pick from `float_discrete_list`.

For integer and unsigned integer tensors:
- Mode `1`: random pick from `int_discrete_list`.
- Mode `2`: random integer from `[int_range_min, int_range_max]` (non-zero safe handling).
- Mode `3`: random integer from `[1, 2^int_bits_max - 1]`.

## Fault Injection Quick Reference

ABFT keys:
- `abft_inject_fault=true|false`
- `abft_inject_fault_delta=<int>`
- `abft_inject_fault_pattern=single_point|trivial|checkered`

AByzFT keys:
- `abyzft_inject_fault=true|false`
- `abyzft_inject_fault_delta=<int>`
- `abyzft_inject_fault_pattern=single_point|trivial|checkered`

## Example Commands

ABFT:

```bash
mlonmcu flow run aww --backend ireellvmc --target host_x86 \
  --config ireellvmc.compiler_mode=abft \
  --config ireellvmc.abft_enable_analysis=true \
  --config ireellvmc.abft_inject_fault=true \
  --config ireellvmc.abft_inject_fault_pattern=single_point \
  --config ireellvmc.abft_inject_fault_delta=8
```

AByzFT float mode 1 (disjoint continuous):

```bash
mlonmcu flow run aww --backend ireellvmc --target host_x86 \
  --config ireellvmc.compiler_mode=abyzft \
  --config ireellvmc.abyzft_scale_sampling_mode=1 \
  --config ireellvmc.abyzft_float_disjoint_min_abs=0.5 \
  --config ireellvmc.abyzft_float_disjoint_max_abs=2.0
```

AByzFT float mode 3 (discrete list):

```bash
mlonmcu flow run aww --backend ireellvmc --target host_x86 \
  --config ireellvmc.compiler_mode=abyzft \
  --config ireellvmc.abyzft_scale_sampling_mode=3 \
  --config ireellvmc.abyzft_float_discrete_list=-8,-4,-2,2,4,8
```

AByzFT int/uint mode 2 (range):

```bash
mlonmcu flow run aww --backend ireellvmc --target host_x86 \
  --config ireellvmc.compiler_mode=abyzft \
  --config ireellvmc.abyzft_scale_sampling_mode=2 \
  --config ireellvmc.abyzft_int_range_min=1 \
  --config ireellvmc.abyzft_int_range_max=8
```

AByzFT with fault injection:

```bash
mlonmcu flow run aww --backend ireellvmc --target host_x86 \
  --config ireellvmc.compiler_mode=abyzft \
  --config ireellvmc.abyzft_inject_fault=true \
  --config ireellvmc.abyzft_inject_fault_pattern=trivial \
  --config ireellvmc.abyzft_inject_fault_delta=8
```
