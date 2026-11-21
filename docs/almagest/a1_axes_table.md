## Per answer

| answer | label | label (transfer v1) | label (Σw rule) | raw AUC | cleaned | drop | cos raw–cl | QR / w[j] (Plate, raw) | bins | max off chord A1 (untr.) | 1−chord/arc A1 (untr.) | uncleaned 1−c/a | cos(chord,Σw) | cos(chord,w) | tail flag | retained w / Σw | split-half cos w (shuf. max) | MLP>lin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **t01_smooth_or_features** — 3 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| or_features_a01_smooth | STRONG | STRONG | PROXY of t04_spiral_a09_no_spiral | 0.872 | 0.758 | 0.115 | 0.716 | 0.149 / 0.851 | 20 | 0.175 (0.243–0.268) | 0.165 (0.351–0.381) | 0.339 | 0.992 | 0.062 | — | 0.81 / 0.99 | 0.575 (0.072) | no |
| or_features_a02_features_or_disk | STRONG | STRONG | PROXY of t04_spiral_a08_spiral | 0.875 | 0.763 | 0.112 | 0.697 | 0.536 / 0.856 | 20 | 0.167 (0.198–0.229) | 0.161 (0.259–0.320) | 0.334 | 0.993 | 0.063 | — | 0.80 / 0.99 | 0.513 (0.014) | no |
| or_features_a03_star_or_artifact | PROXY of t08_odd_feature_a21_disturbed | PROXY of t08_odd_feature_a21_disturbed | WEAK | 0.588 | 0.606 | -0.017 | 0.953 | 0.755 / 0.756 | 20 | 0.231 (0.316–0.374) | 0.505 (0.462–0.529) | 0.502 | 0.919 | 0.249 | — | 0.63 / 0.99 | 0.364 (0.132) | **YES** |
| **t02_edgeon** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | | | |
| a04_yes | SEPARATE | PROXY of t07_rounded_a18_cigar_shaped | PROXY of t07_rounded_a18_cigar_shaped | 0.765 | 0.796 | -0.031 | 0.888 | 0.752 / 0.758 | 20 | 0.176 (0.204–0.225) | 0.143 (0.375–0.407) | 0.257 | 0.990 | 0.047 | — | 0.92 / 0.98 | 0.787 (0.050) | no |
| a05_no | SEPARATE | PROXY of t07_rounded_a18_cigar_shaped | PROXY of t07_rounded_a18_cigar_shaped | 0.790 | 0.821 | -0.031 | 0.873 | 0.359 / 0.780 | 20 | 0.178 (0.166–0.185) | 0.115 (0.309–0.324) | 0.244 | 0.991 | 0.049 | **FLAG** | 0.92 / 0.98 | 0.730 (0.051) | no |
| **t03_bar** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | | | |
| a06_bar | STRONG | STRONG | PROXY of t07_rounded_a16_completely_round | 0.656 | 0.620 | 0.037 | 0.889 | 0.438 / 0.641 | 20 | 0.287 (0.188–0.197) | 0.338 (0.358–0.411) | 0.391 | 0.965 | 0.045 | — | 0.78 / 0.96 | 0.301 (-0.003) | no |
| a07_no_bar | STRONG | STRONG | PROXY of t01_smooth_or_features_a02_features_or_disk | 0.700 | 0.641 | 0.059 | 0.871 | 0.474 / 0.687 | 20 | 0.266 (0.149–0.161) | 0.296 (0.329–0.372) | 0.340 | 0.973 | 0.046 | — | 0.78 / 0.97 | 0.430 (0.040) | no |
| **t04_spiral** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | | | |
| a08_spiral | SEPARATE | PROXY of t01_smooth_or_features_a01_smooth | PROXY of t01_smooth_or_features_a01_smooth | 0.806 | 0.746 | 0.060 | 0.827 | 0.649 / 0.797 | 20 | 0.182 (0.263–0.287) | 0.134 (0.397–0.407) | 0.370 | 0.992 | 0.058 | **FLAG** | 0.84 / 0.99 | 0.442 (0.093) | no |
| a09_no_spiral | SEPARATE | PROXY of t01_smooth_or_features_a02_features_or_disk | PROXY of t01_smooth_or_features_a01_smooth | 0.836 | 0.772 | 0.064 | 0.796 | 0.487 / 0.823 | 20 | 0.180 (0.267–0.285) | 0.130 (0.391–0.410) | 0.353 | 0.992 | 0.058 | — | 0.83 / 0.99 | 0.532 (0.036) | no |
| **t05_bulge_prominence** — 4 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| prominence_a10_no_bulge | STRONG | STRONG | STRONG | 0.822 | 0.660 | 0.162 | 0.823 | 0.234 / 0.804 | 20 | 0.292 (0.130–0.139) | 0.351 (0.380–0.431) | 0.220 | 0.963 | 0.045 | — | 0.82 / 0.97 | 0.598 (0.131) | no |
| prominence_a11_just_noticeable | STRONG | STRONG | PROXY of t01_smooth_or_features_a01_smooth | 0.687 | 0.623 | 0.064 | 0.887 | 0.495 / 0.663 | 20 | 0.238 (0.295–0.337) | 0.228 (0.415–0.451) | 0.304 | 0.995 | 0.042 | **FLAG** | 0.84 / 0.99 | 0.258 (0.065) | no |
| prominence_a12_obvious | STRONG | STRONG | PROXY of t04_spiral_a08_spiral | 0.780 | 0.670 | 0.110 | 0.886 | 0.419 / 0.750 | 20 | 0.150 (0.305–0.371) | 0.122 (0.498–0.565) | 0.299 | 0.996 | 0.038 | — | 0.87 / 0.99 | 0.472 (0.076) | no |
| prominence_a13_dominant | SEPARATE | PROXY of t04_spiral_a08_spiral | PROXY of t01_smooth_or_features_a01_smooth | 0.743 | 0.594 | 0.149 | 0.788 | 0.667 / 0.763 | 20 | 0.368 (0.300–0.352) | 0.507 (0.417–0.496) | 0.295 | 0.968 | 0.044 | — | 0.59 / 0.99 | 0.218 (0.035) | no |
| **t06_odd** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | | | |
| a14_yes | STRONG | STRONG | PROXY of t07_rounded_a18_cigar_shaped | 0.693 | 0.706 | -0.013 | 0.877 | 0.530 / 0.681 | 20 | 0.353 (0.362–0.406) | 0.376 (0.576–0.638) | 0.315 | 0.957 | 0.041 | — | 0.83 / 0.98 | 0.566 (0.055) | no |
| a15_no | STRONG | STRONG | PROXY of t07_rounded_a18_cigar_shaped | 0.692 | 0.707 | -0.015 | 0.880 | 0.562 / 0.679 | 20 | 0.356 (0.353–0.407) | 0.381 (0.558–0.639) | 0.329 | 0.959 | 0.042 | — | 0.83 / 0.98 | 0.531 (0.140) | no |
| **t07_rounded** — 3 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| a16_completely_round | STRONG | STRONG | PROXY of t03_bar_a06_bar | 0.725 | 0.708 | 0.017 | 0.910 | 0.377 / 0.711 | 20 | 0.226 (0.190–0.214) | 0.230 (0.273–0.288) | 0.464 | 0.993 | 0.047 | — | 0.87 / 0.97 | 0.642 (0.042) | no |
| a17_in_between | STRONG | STRONG | PROXY of t04_spiral_a09_no_spiral | 0.595 | 0.617 | -0.022 | 0.934 | 0.558 / 0.597 | 20 | 0.308 (0.248–0.265) | 0.344 (0.508–0.530) | 0.394 | 0.983 | 0.045 | — | 0.86 / 0.99 | 0.405 (0.070) | no |
| a18_cigar_shaped | STRONG | STRONG | PROXY of t02_edgeon_a05_no | 0.813 | 0.828 | -0.015 | 0.831 | 0.372 / 0.794 | 20 | 0.171 (0.148–0.172) | 0.129 (0.236–0.257) | 0.277 | 0.982 | 0.052 | — | 0.92 / 0.97 | 0.792 (0.112) | no |
| **t08_odd_feature** — 7 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| feature_a19_ring | STRONG | STRONG | PROXY of t03_bar_a06_bar | 0.692 | 0.678 | 0.014 | 0.819 | 0.367 / 0.705 | 20 | 0.158 (0.074–0.093) | 0.174 (0.034–0.051) | 0.434 | 0.983 | 0.051 | — | 0.75 / 0.98 | 0.382 (0.124) | no |
| feature_a20_lens_or_arc | SEPARATE | PROXY of t04_spiral_a09_no_spiral | PROXY of t07_rounded_a17_in_between | 0.705 | 0.670 | 0.034 | 0.930 | 0.679 / 0.711 | 20 | 0.319 (0.359–0.379) | 0.351 (0.421–0.520) | 0.344 | 0.959 | 0.046 | — | 0.71 / 0.99 | 0.270 (0.098) | no |
| feature_a21_disturbed | STRONG | STRONG | STRONG | 0.643 | 0.625 | 0.018 | 0.947 | 0.377 / 0.653 | 20 | 0.351 (0.075–0.091) | 0.437 (0.019–0.028) | 0.438 | 0.976 | 0.034 | **FLAG** | 0.73 / 0.99 | 0.270 (0.081) | no |
| feature_a22_irregular | STRONG | STRONG | STRONG | 0.744 | 0.663 | 0.081 | 0.953 | 0.676 / 0.748 | 20 | 0.292 (0.077–0.094) | 0.265 (0.056–0.095) | 0.299 | 0.984 | 0.034 | — | 0.84 / 0.99 | 0.347 (0.201) | no |
| feature_a23_other | STRONG | STRONG | PROXY of t04_spiral_a08_spiral | 0.661 | 0.645 | 0.016 | 0.939 | 0.507 / 0.656 | 20 | 0.331 (0.109–0.124) | 0.396 (0.124–0.213) | 0.325 | 0.959 | 0.039 | — | 0.81 / 1.00 | 0.235 (0.110) | no |
| feature_a24_merger | STRONG | STRONG | STRONG | 0.640 | 0.643 | -0.003 | 0.838 | 0.610 / 0.666 | 20 | 0.346 (0.434–0.470) | 0.398 (0.704–0.733) | 0.315 | 0.970 | 0.041 | — | 0.72 / 0.99 | 0.205 (0.065) | no |
| feature_a38_dust_lane | SEPARATE | PROXY of t02_edgeon_a05_no | STRONG | 0.698 | 0.762 | -0.063 | 0.843 | 0.295 / 0.757 | 20 | 0.238 (0.108–0.141) | 0.363 (0.228–0.254) | 0.553 | 0.948 | 0.090 | — | 0.67 / 0.98 | 0.272 (0.086) | no |
| **t09_bulge_shape** — 3 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| shape_a25_rounded | STRONG | STRONG | STRONG | 0.669 | 0.620 | 0.049 | 0.837 | 0.443 / 0.680 | 20 | 0.272 (0.205–0.232) | 0.272 (0.496–0.561) | 0.348 | 0.984 | 0.050 | — | 0.77 / 0.96 | 0.355 (0.048) | no |
| shape_a26_boxy | NOISE | NOISE | NOISE | 0.570 | 0.574 | -0.004 | 0.939 | 0.536 / 0.575 | 20 | 0.335 (0.170–0.222) | 0.526 (0.560–0.654) | 0.590 | 0.982 | 0.050 | — | 0.62 / 0.99 | 0.057 (0.101) | no |
| shape_a27_no_bulge | STRONG | STRONG | STRONG | 0.700 | 0.628 | 0.072 | 0.825 | 0.534 / 0.710 | 20 | 0.273 (0.210–0.233) | 0.327 (0.469–0.477) | 0.264 | 0.984 | 0.052 | — | 0.76 / 0.96 | 0.341 (0.108) | no |
| **t10_arms_winding** — 3 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| winding_a28_tight | STRONG | STRONG | STRONG | 0.576 | 0.569 | 0.007 | 0.969 | 0.548 / 0.587 | 20 | 0.382 (0.132–0.167) | 0.550 (0.150–0.245) | 0.576 | 0.927 | 0.044 | — | 0.75 / 0.98 | 0.277 (-0.030) | no |
| winding_a29_medium | WEAK | STRONG | STRONG | 0.530 | 0.514 | 0.016 | 0.980 | 0.516 / 0.540 | 20 | 0.400 (0.230–0.283) | 0.681 (0.532–0.631) | 0.536 | 0.975 | 0.041 | — | 0.59 / 0.98 | 0.081 (0.074) | no |
| winding_a30_loose | SEPARATE | PROXY of t05_bulge_prominence_a11_just_noticeable | PROXY of t05_bulge_prominence_a11_just_noticeable | 0.627 | 0.618 | 0.010 | 0.972 | 0.506 / 0.621 | 20 | 0.346 (0.310–0.372) | 0.431 (0.465–0.569) | 0.592 | 0.980 | 0.039 | — | 0.81 / 0.99 | 0.342 (0.070) | no |
| **t11_arms_number** — 6 exclusive alternatives | | | | | | | | | | | | | | | | | | |
| number_a31_1 | PROXY of t05_bulge_prominence_a11_just_noticeable | PROXY of t05_bulge_prominence_a11_just_noticeable | STRONG | 0.614 | 0.568 | 0.046 | 0.956 | 0.472 / 0.605 | 20 | 0.329 (0.367–0.427) | 0.462 (0.618–0.666) | 0.500 | 0.985 | 0.039 | — | 0.58 / 0.99 | 0.195 (0.072) | no |
| number_a32_2 | SEPARATE | PROXY of t03_bar_a06_bar | STRONG | 0.573 | 0.568 | 0.005 | 0.970 | 0.516 / 0.577 | 20 | 0.373 (0.429–0.451) | 0.550 (0.640–0.712) | 0.565 | 0.951 | 0.046 | — | 0.73 / 0.94 | 0.155 (0.085) | no |
| number_a33_3 | NOISE | NOISE | NOISE | 0.518 | 0.521 | -0.003 | 0.984 | 0.551 / 0.488 | 20 | 0.292 (0.153–0.176) | 0.512 (0.272–0.321) | 0.534 | 0.979 | 0.056 | — | 0.66 / 0.99 | 0.087 (0.158) | no |
| number_a34_4 | PROXY of t01_smooth_or_features_a02_features_or_disk | PROXY of t01_smooth_or_features_a02_features_or_disk | STRONG | 0.569 | 0.565 | 0.003 | 0.984 | 0.388 / 0.633 | 20 | 0.277 (0.202–0.269) | 0.579 (0.522–0.642) | 0.624 | 0.970 | 0.090 | — | 0.57 / 0.99 | 0.086 (0.055) | no |
| number_a36_more_than_4 | NOISE | NOISE | NOISE | 0.657 | 0.593 | 0.065 | 0.942 | 0.426 / 0.675 | 20 | 0.341 (0.405–0.419) | 0.504 (0.639–0.713) | 0.452 | 0.982 | 0.066 | — | 0.63 / 0.98 | 0.028 (0.087) | no |
| number_a37_cant_tell | SEPARATE | PROXY of t01_smooth_or_features_a02_features_or_disk | PROXY of t07_rounded_a17_in_between | 0.601 | 0.597 | 0.004 | 0.936 | 0.502 / 0.591 | 20 | 0.314 (0.401–0.433) | 0.332 (0.670–0.694) | 0.304 | 0.973 | 0.044 | — | 0.75 / 0.99 | 0.298 (0.068) | no |

## Proxy candidates: conditional test and chains

| answer | label | candidate proxy of | transfer ratio | own AUC | conditional AUC [95% CI] | shuffled max | root | chain |
|---|---|---|---|---|---|---|---|---|
| t01_smooth_or_features_a03_star_or_artifact | PROXY of t08_odd_feature_a21_disturbed | t08_odd_feature_a21_disturbed | 2.48 | 0.606 | 0.530 [0.369, 0.728] | 0.661 | t08_odd_feature_a21_disturbed | t01_smooth_or_features_a03_star_or_artifact → t08_odd_feature_a21_disturbed |
| t02_edgeon_a04_yes | SEPARATE | t07_rounded_a18_cigar_shaped | 0.97 | 0.796 | 0.604 [0.594, 0.616] | 0.518 | — | — |
| t02_edgeon_a05_no | SEPARATE | t07_rounded_a18_cigar_shaped | 0.98 | 0.821 | 0.590 [0.578, 0.604] | 0.508 | — | — |
| t04_spiral_a08_spiral | SEPARATE | t01_smooth_or_features_a01_smooth | 0.95 | 0.746 | 0.599 [0.591, 0.606] | 0.511 | — | — |
| t04_spiral_a09_no_spiral | SEPARATE | t01_smooth_or_features_a02_features_or_disk | 0.95 | 0.772 | 0.608 [0.600, 0.618] | 0.512 | — | — |
| t05_bulge_prominence_a13_dominant | SEPARATE | t04_spiral_a08_spiral | 1.05 | 0.594 | 0.547 [0.533, 0.560] | 0.511 | — | — |
| t08_odd_feature_a20_lens_or_arc | SEPARATE | t04_spiral_a09_no_spiral | 0.98 | 0.670 | 0.562 [0.547, 0.577] | 0.514 | — | — |
| t08_odd_feature_a38_dust_lane | SEPARATE | t02_edgeon_a05_no | 0.98 | 0.762 | 0.619 [0.573, 0.659] | 0.549 | — | — |
| t10_arms_winding_a30_loose | SEPARATE | t05_bulge_prominence_a11_just_noticeable | 0.91 | 0.618 | 0.567 [0.556, 0.579] | 0.512 | — | — |
| t11_arms_number_a31_1 | PROXY of t05_bulge_prominence_a11_just_noticeable | t05_bulge_prominence_a11_just_noticeable | 1.36 | 0.568 | 0.534 [0.518, 0.551] | 0.529 | t05_bulge_prominence_a11_just_noticeable | t11_arms_number_a31_1 → t05_bulge_prominence_a11_just_noticeable |
| t11_arms_number_a32_2 | SEPARATE | t03_bar_a06_bar | 0.94 | 0.568 | 0.539 [0.531, 0.547] | 0.506 | — | — |
| t11_arms_number_a34_4 | PROXY of t01_smooth_or_features_a02_features_or_disk | t01_smooth_or_features_a02_features_or_disk | 1.19 | 0.565 | 0.533 [0.479, 0.584] | 0.583 | t01_smooth_or_features_a02_features_or_disk | t11_arms_number_a34_4 → t01_smooth_or_features_a02_features_or_disk |
| t11_arms_number_a37_cant_tell | SEPARATE | t01_smooth_or_features_a02_features_or_disk | 0.95 | 0.597 | 0.549 [0.541, 0.557] | 0.505 | — | — |

## Below the ratio floor (AUC − 0.5 < 0.05): WEAK, transfer reported, not used

| answer | own AUC | top transfer into it (ratio, AUC) |
|---|---|---|
| t10_arms_winding_a29_medium | 0.514 | t05_bulge_prominence_a13_dominant (0.74, 0.511) |

## Mutual transfers (both directions ≥ 0.9)

| a | b | ratio a→b | ratio b→a | kept STRONG | by |
|---|---|---|---|---|---|
| t01_smooth_or_features_a01_smooth | t04_spiral_a08_spiral | 0.95 | 0.94 | t01_smooth_or_features_a01_smooth | ancestor |
| t01_smooth_or_features_a01_smooth | t04_spiral_a09_no_spiral | 0.95 | 0.94 | t01_smooth_or_features_a01_smooth | ancestor |
| t01_smooth_or_features_a02_features_or_disk | t04_spiral_a08_spiral | 0.95 | 0.93 | t01_smooth_or_features_a02_features_or_disk | ancestor |
| t01_smooth_or_features_a02_features_or_disk | t04_spiral_a09_no_spiral | 0.95 | 0.94 | t01_smooth_or_features_a02_features_or_disk | ancestor |
| t02_edgeon_a04_yes | t07_rounded_a18_cigar_shaped | 0.97 | 0.97 | t07_rounded_a18_cigar_shaped | higher AUC |
| t02_edgeon_a05_no | t07_rounded_a18_cigar_shaped | 0.98 | 0.98 | t07_rounded_a18_cigar_shaped | higher AUC |

## Candidate two-concept planes (both STRONG or SEPARATE, |r| < shuffled-null max + 0.1)

| a | b | r(d_a, d_b) | null max \|r\| | n both | same question |
|---|---|---|---|---|---|
| t04_spiral_a08_spiral | t09_bulge_shape_a27_no_bulge | +0.001 | 0.064 | 15420 | no |
| t07_rounded_a18_cigar_shaped | t08_odd_feature_a24_merger | -0.003 | 0.077 | 31464 | no |
| t02_edgeon_a05_no | t05_bulge_prominence_a13_dominant | +0.004 | 0.090 | 33078 | no |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a20_lens_or_arc | +0.004 | 0.096 | 30799 | no |
| t08_odd_feature_a24_merger | t09_bulge_shape_a27_no_bulge | +0.004 | 0.084 | 15491 | no |
| t04_spiral_a08_spiral | t07_rounded_a18_cigar_shaped | +0.004 | 0.098 | 32313 | no |
| t03_bar_a07_no_bar | t06_odd_a14_yes | +0.004 | 0.072 | 33078 | no |
| t05_bulge_prominence_a11_just_noticeable | t11_arms_number_a32_2 | +0.005 | 0.069 | 20241 | no |
| t05_bulge_prominence_a10_no_bulge | t06_odd_a14_yes | +0.007 | 0.080 | 33080 | no |
| t08_odd_feature_a21_disturbed | t11_arms_number_a32_2 | -0.007 | 0.116 | 19368 | no |
| t05_bulge_prominence_a10_no_bulge | t06_odd_a15_no | -0.011 | 0.090 | 33080 | no |
| t03_bar_a07_no_bar | t06_odd_a15_no | -0.011 | 0.106 | 33078 | no |
| t02_edgeon_a05_no | t05_bulge_prominence_a12_obvious | +0.013 | 0.093 | 33078 | no |
| t05_bulge_prominence_a13_dominant | t11_arms_number_a32_2 | -0.013 | 0.071 | 20241 | no |
| t09_bulge_shape_a25_rounded | t10_arms_winding_a30_loose | +0.015 | 0.069 | 10383 | no |
| t05_bulge_prominence_a12_obvious | t08_odd_feature_a19_ring | -0.017 | 0.069 | 30799 | no |
| t04_spiral_a09_no_spiral | t07_rounded_a18_cigar_shaped | +0.018 | 0.087 | 32313 | no |
| t06_odd_a15_no | t08_odd_feature_a22_irregular | -0.019 | 0.107 | 32213 | no |
| t05_bulge_prominence_a10_no_bulge | t10_arms_winding_a30_loose | +0.020 | 0.116 | 20243 | no |
| t02_edgeon_a04_yes | t05_bulge_prominence_a12_obvious | +0.021 | 0.099 | 33078 | no |
| t06_odd_a14_yes | t08_odd_feature_a22_irregular | +0.021 | 0.082 | 32213 | no |
| t02_edgeon_a04_yes | t05_bulge_prominence_a13_dominant | +0.023 | 0.032 | 33078 | no |
| t07_rounded_a18_cigar_shaped | t08_odd_feature_a20_lens_or_arc | -0.023 | 0.114 | 31464 | no |
| t07_rounded_a18_cigar_shaped | t10_arms_winding_a30_loose | -0.023 | 0.091 | 19476 | no |
| t03_bar_a07_no_bar | t10_arms_winding_a28_tight | +0.024 | 0.081 | 20243 | no |
| t03_bar_a06_bar | t10_arms_winding_a28_tight | +0.027 | 0.060 | 20243 | no |
| t04_spiral_a09_no_spiral | t09_bulge_shape_a27_no_bulge | +0.031 | 0.081 | 15420 | no |
| t08_odd_feature_a20_lens_or_arc | t09_bulge_shape_a27_no_bulge | -0.034 | 0.090 | 15491 | no |
| t05_bulge_prominence_a13_dominant | t08_odd_feature_a21_disturbed | -0.035 | 0.062 | 30799 | no |
| t07_rounded_a16_completely_round | t08_odd_feature_a22_irregular | +0.037 | 0.112 | 31464 | no |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a24_merger | -0.044 | 0.045 | 30799 | no |
| t08_odd_feature_a21_disturbed | t11_arms_number_a37_cant_tell | +0.044 | 0.071 | 19368 | no |
| t04_spiral_a08_spiral | t05_bulge_prominence_a10_no_bulge | -0.047 | 0.133 | 33078 | no |
| t05_bulge_prominence_a12_obvious | t11_arms_number_a32_2 | +0.049 | 0.079 | 20241 | no |
| t08_odd_feature_a22_irregular | t09_bulge_shape_a27_no_bulge | -0.049 | 0.072 | 15491 | no |
| t03_bar_a07_no_bar | t08_odd_feature_a38_dust_lane | -0.052 | 0.156 | 30798 | no |
| t05_bulge_prominence_a11_just_noticeable | t08_odd_feature_a38_dust_lane | -0.053 | 0.055 | 30799 | no |
| t05_bulge_prominence_a13_dominant | t09_bulge_shape_a27_no_bulge | -0.055 | 0.051 | 15419 | no |
| t03_bar_a07_no_bar | t08_odd_feature_a22_irregular | +0.055 | 0.106 | 30798 | no |
| t08_odd_feature_a20_lens_or_arc | t10_arms_winding_a28_tight | +0.056 | 0.072 | 19371 | no |
| t02_edgeon_a05_no | t08_odd_feature_a24_merger | -0.057 | 0.125 | 31820 | no |
| t05_bulge_prominence_a10_no_bulge | t05_bulge_prominence_a13_dominant | -0.057 | 0.059 | 33080 | yes |
| t08_odd_feature_a23_other | t11_arms_number_a32_2 | -0.057 | 0.066 | 19368 | no |
| t08_odd_feature_a38_dust_lane | t10_arms_winding_a30_loose | +0.059 | 0.173 | 19371 | no |
| t04_spiral_a08_spiral | t10_arms_winding_a28_tight | -0.063 | 0.084 | 20244 | no |
| t03_bar_a06_bar | t05_bulge_prominence_a12_obvious | -0.065 | 0.107 | 33076 | no |
| t02_edgeon_a04_yes | t08_odd_feature_a24_merger | +0.067 | 0.085 | 31820 | no |
| t08_odd_feature_a20_lens_or_arc | t09_bulge_shape_a25_rounded | -0.067 | 0.118 | 15491 | no |
| t05_bulge_prominence_a12_obvious | t10_arms_winding_a28_tight | -0.068 | 0.121 | 20243 | no |
| t08_odd_feature_a24_merger | t10_arms_winding_a28_tight | -0.068 | 0.107 | 19371 | no |
| t05_bulge_prominence_a12_obvious | t06_odd_a15_no | +0.072 | 0.050 | 33080 | no |
| t05_bulge_prominence_a10_no_bulge | t05_bulge_prominence_a11_just_noticeable | +0.072 | 0.117 | 33080 | yes |
| t05_bulge_prominence_a13_dominant | t09_bulge_shape_a25_rounded | -0.072 | 0.079 | 15419 | no |
| t05_bulge_prominence_a12_obvious | t06_odd_a14_yes | -0.073 | 0.126 | 33080 | no |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a21_disturbed | +0.073 | 0.091 | 30799 | no |
| t04_spiral_a09_no_spiral | t05_bulge_prominence_a10_no_bulge | +0.073 | 0.069 | 33078 | no |
| t02_edgeon_a04_yes | t05_bulge_prominence_a11_just_noticeable | +0.074 | 0.063 | 33078 | no |
| t01_smooth_or_features_a01_smooth | t07_rounded_a18_cigar_shaped | -0.075 | 0.075 | 34055 | no |
| t08_odd_feature_a24_merger | t09_bulge_shape_a25_rounded | -0.076 | 0.079 | 15491 | no |
| t01_smooth_or_features_a01_smooth | t09_bulge_shape_a27_no_bulge | +0.076 | 0.103 | 16692 | no |
| t02_edgeon_a04_yes | t03_bar_a07_no_bar | -0.076 | 0.083 | 33076 | no |
| t01_smooth_or_features_a02_features_or_disk | t10_arms_winding_a28_tight | -0.077 | 0.069 | 20244 | no |
| t01_smooth_or_features_a02_features_or_disk | t07_rounded_a18_cigar_shaped | +0.077 | 0.087 | 34055 | no |
| t01_smooth_or_features_a02_features_or_disk | t09_bulge_shape_a27_no_bulge | -0.078 | 0.059 | 16692 | no |
| t02_edgeon_a04_yes | t08_odd_feature_a19_ring | +0.079 | 0.077 | 31820 | no |
| t07_rounded_a17_in_between | t11_arms_number_a32_2 | +0.083 | 0.073 | 19474 | no |
| t09_bulge_shape_a27_no_bulge | t11_arms_number_a32_2 | -0.083 | 0.101 | 10381 | no |
| t04_spiral_a09_no_spiral | t10_arms_winding_a28_tight | +0.084 | 0.079 | 20244 | no |
| t07_rounded_a16_completely_round | t11_arms_number_a37_cant_tell | -0.087 | 0.013 | 19474 | no |
| t09_bulge_shape_a27_no_bulge | t10_arms_winding_a30_loose | -0.090 | 0.102 | 10383 | no |
| t06_odd_a15_no | t08_odd_feature_a23_other | +0.091 | 0.089 | 32213 | no |
| t01_smooth_or_features_a01_smooth | t10_arms_winding_a28_tight | +0.092 | 0.116 | 20244 | no |
| t04_spiral_a08_spiral | t11_arms_number_a32_2 | +0.092 | 0.106 | 20241 | no |
| t08_odd_feature_a19_ring | t08_odd_feature_a22_irregular | +0.092 | 0.092 | 32213 | yes |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a38_dust_lane | +0.093 | 0.130 | 30799 | no |
| t07_rounded_a17_in_between | t08_odd_feature_a21_disturbed | +0.094 | 0.062 | 31464 | no |
| t02_edgeon_a05_no | t03_bar_a07_no_bar | +0.094 | 0.053 | 33076 | no |
| t02_edgeon_a05_no | t08_odd_feature_a19_ring | -0.094 | 0.052 | 31820 | no |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a23_other | +0.096 | 0.110 | 30799 | no |
| t08_odd_feature_a21_disturbed | t08_odd_feature_a38_dust_lane | +0.098 | 0.145 | 32213 | yes |
| t02_edgeon_a05_no | t10_arms_winding_a30_loose | -0.099 | 0.074 | 20243 | no |
| t06_odd_a15_no | t11_arms_number_a32_2 | +0.099 | 0.093 | 20242 | no |
| t05_bulge_prominence_a13_dominant | t08_odd_feature_a38_dust_lane | +0.100 | 0.230 | 30799 | no |
| t03_bar_a06_bar | t06_odd_a14_yes | -0.101 | 0.084 | 33078 | no |
| t09_bulge_shape_a25_rounded | t11_arms_number_a32_2 | +0.101 | 0.069 | 10381 | no |
| t06_odd_a14_yes | t08_odd_feature_a23_other | -0.101 | 0.064 | 32213 | no |
| t04_spiral_a09_no_spiral | t11_arms_number_a32_2 | -0.102 | 0.079 | 20241 | no |
| t06_odd_a14_yes | t11_arms_number_a32_2 | -0.102 | 0.041 | 20242 | no |
| t08_odd_feature_a23_other | t10_arms_winding_a28_tight | +0.103 | 0.076 | 19371 | no |
| t08_odd_feature_a20_lens_or_arc | t08_odd_feature_a22_irregular | -0.105 | 0.062 | 32213 | yes |
| t03_bar_a06_bar | t06_odd_a15_no | +0.106 | 0.125 | 33078 | no |
| t05_bulge_prominence_a10_no_bulge | t07_rounded_a17_in_between | -0.108 | 0.142 | 32312 | no |
| t02_edgeon_a05_no | t05_bulge_prominence_a11_just_noticeable | -0.109 | 0.109 | 33078 | no |
| t08_odd_feature_a22_irregular | t09_bulge_shape_a25_rounded | +0.110 | 0.081 | 15491 | no |
| t06_odd_a14_yes | t10_arms_winding_a30_loose | +0.111 | 0.129 | 20244 | no |
| t05_bulge_prominence_a13_dominant | t06_odd_a15_no | +0.112 | 0.100 | 33080 | no |
| t10_arms_winding_a30_loose | t11_arms_number_a32_2 | +0.112 | 0.119 | 20232 | no |
| t01_smooth_or_features_a02_features_or_disk | t02_edgeon_a05_no | +0.113 | 0.065 | 34350 | no |
| t08_odd_feature_a24_merger | t11_arms_number_a32_2 | -0.115 | 0.103 | 19368 | no |
| t02_edgeon_a04_yes | t08_odd_feature_a21_disturbed | +0.115 | 0.080 | 31820 | no |
| t01_smooth_or_features_a01_smooth | t02_edgeon_a05_no | -0.116 | 0.048 | 34350 | no |
| t05_bulge_prominence_a11_just_noticeable | t06_odd_a14_yes | -0.117 | 0.088 | 33080 | no |
| t04_spiral_a08_spiral | t09_bulge_shape_a25_rounded | +0.118 | 0.038 | 15420 | no |
| t05_bulge_prominence_a13_dominant | t10_arms_winding_a28_tight | -0.118 | 0.088 | 20243 | no |
| t06_odd_a15_no | t10_arms_winding_a30_loose | -0.119 | 0.126 | 20244 | no |
| t03_bar_a07_no_bar | t05_bulge_prominence_a12_obvious | +0.120 | 0.068 | 33076 | no |
| t05_bulge_prominence_a13_dominant | t06_odd_a14_yes | -0.120 | 0.087 | 33080 | no |
| t05_bulge_prominence_a11_just_noticeable | t06_odd_a15_no | +0.122 | 0.076 | 33080 | no |
| t02_edgeon_a05_no | t08_odd_feature_a21_disturbed | -0.123 | 0.177 | 31820 | no |
| t02_edgeon_a05_no | t08_odd_feature_a20_lens_or_arc | -0.127 | 0.095 | 31820 | no |
| t02_edgeon_a04_yes | t10_arms_winding_a30_loose | +0.127 | 0.114 | 20243 | no |
| t03_bar_a06_bar | t08_odd_feature_a22_irregular | -0.129 | 0.064 | 30798 | no |
| t08_odd_feature_a22_irregular | t10_arms_winding_a28_tight | -0.133 | 0.086 | 19371 | no |
| t01_smooth_or_features_a01_smooth | t05_bulge_prominence_a10_no_bulge | +0.134 | 0.082 | 33080 | no |
| t08_odd_feature_a19_ring | t08_odd_feature_a20_lens_or_arc | -0.134 | 0.092 | 32213 | yes |
| t08_odd_feature_a21_disturbed | t09_bulge_shape_a25_rounded | -0.135 | 0.089 | 15491 | no |
| t05_bulge_prominence_a11_just_noticeable | t09_bulge_shape_a25_rounded | -0.135 | 0.090 | 15419 | no |
| t05_bulge_prominence_a13_dominant | t07_rounded_a18_cigar_shaped | -0.137 | 0.084 | 32312 | no |
| t03_bar_a06_bar | t10_arms_winding_a30_loose | -0.139 | 0.079 | 20243 | no |
| t07_rounded_a18_cigar_shaped | t08_odd_feature_a19_ring | +0.144 | 0.083 | 31464 | no |
| t01_smooth_or_features_a02_features_or_disk | t02_edgeon_a04_yes | -0.148 | 0.083 | 34350 | no |
| t04_spiral_a09_no_spiral | t09_bulge_shape_a25_rounded | -0.149 | 0.103 | 15420 | no |
| t05_bulge_prominence_a10_no_bulge | t10_arms_winding_a28_tight | +0.151 | 0.084 | 20243 | no |
| t04_spiral_a08_spiral | t06_odd_a15_no | -0.151 | 0.058 | 33081 | no |
| t01_smooth_or_features_a01_smooth | t02_edgeon_a04_yes | +0.151 | 0.086 | 34350 | no |
| t01_smooth_or_features_a02_features_or_disk | t05_bulge_prominence_a10_no_bulge | -0.152 | 0.087 | 33080 | no |
| t08_odd_feature_a21_disturbed | t09_bulge_shape_a27_no_bulge | +0.153 | 0.076 | 15491 | no |
| t08_odd_feature_a20_lens_or_arc | t11_arms_number_a32_2 | -0.153 | 0.119 | 19368 | no |
| t08_odd_feature_a20_lens_or_arc | t08_odd_feature_a21_disturbed | -0.153 | 0.060 | 32213 | yes |
| t07_rounded_a18_cigar_shaped | t08_odd_feature_a21_disturbed | +0.155 | 0.105 | 31464 | no |
| t08_odd_feature_a38_dust_lane | t11_arms_number_a32_2 | +0.155 | 0.071 | 19368 | no |
| t02_edgeon_a04_yes | t08_odd_feature_a20_lens_or_arc | +0.156 | 0.088 | 31820 | no |
| t01_smooth_or_features_a01_smooth | t08_odd_feature_a21_disturbed | -0.163 | 0.074 | 32213 | no |
| t03_bar_a07_no_bar | t07_rounded_a18_cigar_shaped | -0.163 | 0.097 | 32310 | no |
| t07_rounded_a16_completely_round | t10_arms_winding_a30_loose | +0.170 | 0.136 | 19476 | no |
| t01_smooth_or_features_a02_features_or_disk | t06_odd_a15_no | -0.171 | 0.071 | 34828 | no |
| t04_spiral_a09_no_spiral | t06_odd_a15_no | +0.173 | 0.101 | 33081 | no |
| t08_odd_feature_a23_other | t09_bulge_shape_a27_no_bulge | +0.174 | 0.092 | 15491 | no |
| t07_rounded_a16_completely_round | t10_arms_winding_a28_tight | -0.176 | 0.106 | 19476 | no |
| t01_smooth_or_features_a02_features_or_disk | t06_odd_a14_yes | +0.176 | 0.165 | 34828 | no |
| t05_bulge_prominence_a10_no_bulge | t11_arms_number_a32_2 | -0.177 | 0.104 | 20241 | no |
| t04_spiral_a09_no_spiral | t06_odd_a14_yes | -0.180 | 0.131 | 33081 | no |
| t01_smooth_or_features_a02_features_or_disk | t09_bulge_shape_a25_rounded | +0.181 | 0.108 | 16692 | no |
| t08_odd_feature_a24_merger | t11_arms_number_a37_cant_tell | +0.182 | 0.091 | 19368 | no |
| t08_odd_feature_a24_merger | t08_odd_feature_a38_dust_lane | +0.185 | 0.135 | 32213 | yes |
| t07_rounded_a16_completely_round | t08_odd_feature_a24_merger | +0.187 | 0.093 | 31464 | no |
| t04_spiral_a09_no_spiral | t08_odd_feature_a21_disturbed | -0.190 | 0.112 | 30800 | no |
| t10_arms_winding_a28_tight | t11_arms_number_a32_2 | -0.191 | 0.127 | 20232 | no |
| t05_bulge_prominence_a13_dominant | t08_odd_feature_a24_merger | +0.198 | 0.105 | 30799 | no |
| t08_odd_feature_a22_irregular | t11_arms_number_a37_cant_tell | -0.200 | 0.112 | 19368 | no |
| t04_spiral_a08_spiral | t08_odd_feature_a21_disturbed | +0.204 | 0.104 | 30800 | no |
| t02_edgeon_a04_yes | t03_bar_a06_bar | +0.211 | 0.137 | 33076 | no |
| t07_rounded_a17_in_between | t10_arms_winding_a28_tight | -0.213 | 0.211 | 19476 | no |
| t03_bar_a06_bar | t09_bulge_shape_a27_no_bulge | -0.217 | 0.117 | 15417 | no |
| t05_bulge_prominence_a11_just_noticeable | t09_bulge_shape_a27_no_bulge | +0.229 | 0.145 | 15419 | no |
| t05_bulge_prominence_a10_no_bulge | t11_arms_number_a37_cant_tell | +0.230 | 0.155 | 20241 | no |
| t08_odd_feature_a20_lens_or_arc | t08_odd_feature_a38_dust_lane | +0.239 | 0.153 | 32213 | yes |

## A1 bends more than every untrained seed (1 − chord/arc)

| answer | label | 1−chord/arc A1 (untrained) | max off chord A1 (untrained) | uncleaned 1−c/a | tail flag |
|---|---|---|---|---|---|
| t05_bulge_prominence_a13_dominant | SEPARATE | 0.507 (0.417–0.496) | 0.368 (0.300–0.352) | 0.295 | — |
| t08_odd_feature_a19_ring | STRONG | 0.174 (0.034–0.051) | 0.158 (0.074–0.093) | 0.434 | — |
| t08_odd_feature_a21_disturbed | STRONG | 0.437 (0.019–0.028) | 0.351 (0.075–0.091) | 0.438 | **FLAG** |
| t08_odd_feature_a22_irregular | STRONG | 0.265 (0.056–0.095) | 0.292 (0.077–0.094) | 0.299 | — |
| t08_odd_feature_a23_other | STRONG | 0.396 (0.124–0.213) | 0.331 (0.109–0.124) | 0.325 | — |
| t08_odd_feature_a38_dust_lane | SEPARATE | 0.363 (0.228–0.254) | 0.238 (0.108–0.141) | 0.553 | — |
| t10_arms_winding_a28_tight | STRONG | 0.550 (0.150–0.245) | 0.382 (0.132–0.167) | 0.576 | — |
| t10_arms_winding_a29_medium | WEAK | 0.681 (0.532–0.631) | 0.400 (0.230–0.283) | 0.536 | — |
| t11_arms_number_a33_3 | NOISE | 0.512 (0.272–0.321) | 0.292 (0.153–0.176) | 0.534 | — |
