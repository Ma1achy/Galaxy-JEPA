# The Plate's data file (`plate.bin`), format version 2

*Written by `artifacts/plate_export.py` (Kickoff C). `plate.bin`: 29,000 galaxies (cluster-stratified plus
rare-class oversampling, `plate_subset.json`), 5.07 MB raw, 3.81 MB gzip-9, 3.63 MB brotli-11. Version 2
replaces v1's float16 PCA-32 with int8 coordinates on a 48-D basis built from the probes themselves.
Encoder M only (`runs/m/encoder.pt`, block-11 mean-pooled embeddings in
`artifacts/out/c2_m1_embeddings.npz`). Raw GZ2 votes only: `_fraction` and
`_count`, never `_debiased`. No images; the page fetches cutouts by RA/Dec.*

## Container

| bytes | content |
|---|---|
| 0–7 | magic `ALMGPLT1` (ASCII) |
| 8–11 | `uint32` LE: header length `H` in bytes (already padded) |
| 12–15 | `uint32` 0 (reserved) |
| 16 … 16+H−1 | UTF-8 JSON header, right-padded with spaces so that 16+H is a multiple of 8 |
| 16+H … | body: the sections, each starting at `body + section.offset`, offsets multiples of 8 |

Everything is **little-endian**. Every section starts 8-byte aligned in the file, so on a
little-endian host (every browser) `new Uint16Array(buf, 16+H+offset, count)` and the like work
without copying. `float16` needs `Float16Array` (recent browsers only) or the manual decode below.

**Planar layout.** A field with shape `[cols, n]` is stored column by column: all galaxies' column 0,
then all galaxies' column 1, and so on. Value (galaxy `i`, column `j`) = `a[j*n + i]`. Planar
compresses better and gives a whole answer's column as one contiguous `subarray` for colouring.
The Plate's own synthetic arrays are galaxy-major (`E[i*D+d]`), so transpose `coords` once on load if
you keep its `knn()` / `dot()` code unchanged.

**Row order** is a seeded shuffle of the subset, so any prefix is a fair sample (the 500-galaxy
`plate_test.bin` is the first 500 rows). Row index `i` is the galaxy's identity inside the file,
and the `knn` entries point at it.

## Per-galaxy sections (length `n`)

| name | dtype | shape | decode | meaning |
|---|---|---|---|---|
| `objid` | uint64 | [n] | `getBigUint64(…, true).toString()` | SDSS DR8+ `objID` (the repo's `object_id`), **not** GZ2's `dr7objid` |
| `ra`, `dec` | float32 | [n] | as is | degrees, J2000 (float32 is 0.1″ at worst) |
| `readout` | uint8 | [37, n] | `d = lo[j] + v*(hi[j]-lo[j])/255` | M's probe for answer `j`: decision `d` (a logit); `P = 1/(1+e^-d)`. See below |
| `map` | float16 | [2, n] | half → float | UMAP x, y (bounds in `header.map.min/max`) |
| `coords` | int8 | [48, n] | `z = q * header.coords.scale[k]` | the galaxy's coordinates `z` on the 48-D basis (standardised space). See *Coordinates* below |
| `knn` | uint16 | [6, n] | row index | the 6 nearest neighbours in the **full 384-D** embedding (Euclidean, among the file's rows, self excluded), nearest first |
| `knn_d6` | float16 | [n] | half → float | 384-D Euclidean distance to the 6th neighbour |
| `votes` | uint8 | [37, n] | `v/255` | raw GZ2 vote fraction. `v >= 128` ⇔ fraction ≥ 0.5 **exactly** (bytes are `floor(255f + 0.5)`) |
| `counts` | uint8 | [11, n] | as is, 255 = "255 or more" | question total: sum of that question's answers' `_count` columns |
| `mag_r` | uint16 | [n] | `10 + v/1000`; 65535 missing | `modelMag_r` |
| `ba` | uint8 | [n] | `v/254`; 255 missing | `expAB_r`, the exponential-fit axis ratio b/a in r (the repo's primary b/a, see `v_findings.md`) |
| `flags` | uint8 | [n] | bits | bit 0: 1 = probe-test, 0 = probe-train (**read-out in-sample**). Bits 1–3: inclusion reason, 0 = cluster-stratified, 1–7 = oversampled rare class (`header.flags`) |

## Model sections

| name | dtype | shape | meaning |
|---|---|---|---|
| `std_mean` | float32 | [384] | standardisation mean `μ` (probe 0's StandardScaler) |
| `std_scale` | float32 | [384] | standardisation scale `σ` (the same scaler) |
| `basis` | float32 | [48, 384] | basis `B`, orthonormal rows, in the standardised space |
| `centre` | float32 | [48] | `c`, subtracted after projecting |

For a 384-D embedding `x`: `s = (x − μ)/σ`, `z = B s − c`. These are binary sections rather than JSON
fields: 18,432 floats in JSON would be several times the bytes for the same values. float32 was kept
(the basis is 68 KB gzip-9 of the file's 3.81 MB).

## Header (JSON)

- `n`, `version`, `encoder` ("M"), `provenance` (checkpoint, code SHA, seed, pool size, `limit` — a
  non-zero `limit` means a development file, not the real one).
- `answers[37]`: `{id, plate_id, question, q}`. Order is the GZ2 tree (t01, t02, …, t11), which is
  **not** the Plate's TASKS order: match on `plate_id` (the Plate's own `t01_smooth`-style ids). `q`
  indexes `questions[11]` and the rows of `counts`.
- `readout`: `lo[37]`, `hi[37]` (each answer's 0.1 and 99.9 percentiles of the decision over the file's
  galaxies), `fitted[37]` (false: no probe, the row is 0), the probe definition.
- `probe_basis`: `w[37][48]`, `b[37]`. Each probe on the coordinates: `d = w[j]·z + b[j]`. **Exact on
  the float coordinates for every galaxy**, not only on a reconstruction, because every probe direction
  is itself in the basis (see below); only the int8 rounding of `z` separates it from the read-out
  (`reference_auc`: `auc_basis_i8` vs `auc_384`). Use `readout` for display; use `probe_basis` where you
  need the direction itself (walks, sliders): answer `j`'s direction in coordinate space is `w[j]`.
- `coords`: `n_components` (48), `n_probe` (37), `n_residual` (11), `qr_order[37]` (answer ids, the
  order the probe directions were orthonormalised in: the file's answer order), `scale[48]` (int8 step
  per coordinate), `quantisation`, `standardise`, `construction`, `project` (the rules, as text),
  `variance_kept`, `variance_kept_probe_span`.
- `map`: method `umap`, its parameters and version, `min`/`max`.
- `knn`, `flags`, `layout`: descriptions.
- `reference_auc[37]`: on this file's probe-test rows eligible for the answer. `auc_384` (full-precision
  probe), `auc_u8` (the stored byte), `auc_basis_i8` (the `probe_basis` score on the stored, dequantised
  `z`), `n_pos`, `n_neg`; in `plate.bin` also `auc_basis` (the same on the float `z`, before int8) and
  `auc_probe_test_full` (the probe on the whole 34,829-galaxy test split).
- `sections[]`: `{name, dtype, shape, offset, bytes, meaning}`. `offset` is relative to the body start.

## The rules the page has to reproduce

- **Label** (as the probes were trained): positive ⇔ `votes >= 128`.
- **Eligible** for answer `j` (the probe's `vote_count_min = 1`): `counts[q_j] >= 1`. GZ2 writes an
  unreached question's fractions as 0.0, so a 0 byte with `counts = 0` means "never asked", not "nobody
  said yes". The Plate's own `reached()` uses `n >= 5`, which is the page's choice, not the probe's.
- A read-out on a `flags & 1 == 0` galaxy is in-sample: the probe was fitted on it.
- `readout` saturates at 0 and 255 outside the answer's [0.1, 99.9] percentile range. Order inside the
  range is kept to 1/255 of it. `sigmoid × 255` was measured and rejected: it puts most of a rare
  answer's galaxies on a dozen byte values and cost up to 0.41 AUC in development.

## Coordinates: the 48-D basis (int8)

**Space.** The standardised embedding `s = (x − μ)/σ`, the space the probes read. The 37 probes were
each fitted with their own StandardScaler (on their own answer's eligible train galaxies, so the scalers
differ slightly); the file uses probe 0's (t01, fitted on the whole 160,857-galaxy train split) and every
other probe is re-expressed in it exactly — a change of scaler is affine, nothing is lost.

**Basis.** Rows 0–36: the 37 probe weight vectors in that space, orthonormalised by QR in `qr_order`
(the file's answer order), so row `j` is answer `j`'s direction with the earlier answers' directions
removed. Rows 37–47: the top 11 principal components of the file's standardised embeddings after the
probe span is projected out — they carry the neighbours, not the probes. The 48 dimensions keep
69.8% of the standardised embedding's total variance over the file's galaxies (the 37 probe
directions alone: 0.39% — the probes read low-variance directions, which is why a plain PCA of up to 128 components could not carry them (`plate_pca_sweep.json`)).

**Quantisation.** `coords` holds `q`, int8 in [−127, 127]; `z[k] = q[k] * header.coords.scale[k]`.
Scale rule: `scale[k] = max |z[:, k]|` over the file's galaxies `/ 127`, per coordinate, from the float
coordinates before quantising; `q = round(z / scale)`. Nothing clips (the largest |z| maps to exactly
±127) and −128 never occurs.

**Cost.** Against the full 384-D probe, on this file's probe-test galaxies: mean 0.0002, median 0.00005, worst 0.0039 AUC (t11 four arms); per answer in `reference_auc`. On the float
coordinates the loss is zero by construction; all of it is int8 rounding. `plate_test.bin` carries its
parent's `std_*`, `basis`, `centre` and `scale` unchanged: it is a slice, not a refit.

**Neighbours.** `knn` is still computed in the raw (unstandardised) 384-D embedding. Nearest neighbours
recomputed on the stored coordinates share 3.18 on average (PCA-32 in the v1 file: 5.22 on the 40,000-galaxy file; the 48-D basis on those same 40,000: 3.04). **The basis is chosen for the probes, not for neighbour search:** use `knn` for neighbours, never distances in `coords` of the 6 with it (1,000 seeded queries).

## Serving

Serve with `Content-Encoding: br` (or gzip); `fetch().arrayBuffer()` returns the decoded bytes.

## Reference decoder (JavaScript, DataView)

```js
// const P = decodePlate(await (await fetch('plate.bin')).arrayBuffer());
function decodePlate(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  if (new TextDecoder().decode(u8.subarray(0, 8)) !== 'ALMGPLT1') throw new Error('not a Plate file');
  const H = dv.getUint32(8, true), base = 16 + H;
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(16, base)));
  const half = (h) => {                        // IEEE 754 binary16 -> number
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  };
  const size = { int8: 1, uint8: 1, uint16: 2, float16: 2, float32: 4, uint64: 8 };
  const S = {};
  for (const s of header.sections) {
    const o = base + s.offset, len = s.bytes / size[s.dtype];
    let a;
    if (s.dtype === 'uint8') a = u8.subarray(o, o + len);
    else if (s.dtype === 'int8') a = new Int8Array(buf, o, len);
    else if (s.dtype === 'uint16') { a = new Uint16Array(len); for (let i = 0; i < len; i++) a[i] = dv.getUint16(o + 2 * i, true); }
    else if (s.dtype === 'float16') { a = new Float32Array(len); for (let i = 0; i < len; i++) a[i] = half(dv.getUint16(o + 2 * i, true)); }
    else if (s.dtype === 'float32') { a = new Float32Array(len); for (let i = 0; i < len; i++) a[i] = dv.getFloat32(o + 4 * i, true); }
    else if (s.dtype === 'uint64') { a = new Array(len); for (let i = 0; i < len; i++) a[i] = dv.getBigUint64(o + 8 * i, true).toString(); }
    S[s.name] = a;
  }
  const n = header.n, { lo, hi } = header.readout, { scale } = header.coords, { w, b } = header.probe_basis;
  const at = (name, j, i) => S[name][j * n + i];                               // planar access
  const decision = (j, i) => lo[j] + at('readout', j, i) * (hi[j] - lo[j]) / 255; // logit
  const z = (k, i) => at('coords', k, i) * scale[k];                           // coordinate
  return { header, S, n, at, decision, z,
    probeZ: (j, i) => w[j].reduce((s, wk, k) => s + wk * z(k, i), b[j]),     // probe on z
    prob: (j, i) => 1 / (1 + Math.exp(-decision(j, i))),
    vote: (j, i) => at('votes', j, i) / 255,
    eligible: (j, i) => at('counts', header.answers[j].q, i) >= 1,
    mag: (i) => (S.mag_r[i] === 65535 ? NaN : 10 + S.mag_r[i] / 1000),
    ba: (i) => (S.ba[i] === 255 ? NaN : S.ba[i] / 254),
    isTest: (i) => (S.flags[i] & 1) === 1 };
}
```

The decoder copies through `DataView`, so it runs on any host and any alignment (`Int8Array` and
`subarray` are single bytes: no alignment or byte order to care about). On a little-endian page with
an aligned buffer, `new Uint16Array(buf, o, len)` and `new Float32Array(buf, o, len)` are zero-copy
equivalents. For the kNN / walk code, dequantise `coords` once into a galaxy-major `Float32Array`
(`E[i*D+k] = z(k, i)`) rather than calling `z` in an inner loop.

Verified: `node` running this decoder (extracted verbatim from this file) against `read_plate` in
Python on both `plate.bin` and `plate_test.bin` — every field on six rows (first, second, third,
middle, last two) plus whole-section sums; exact on every integer and float field, ≤ 1e-15 relative on the derived `prob` and `probeZ` (2026-09-28, node v22.14.0).
