## Per answer

| answer | label | raw AUC | cleaned | drop | cos raw–cl | QR / w[j] (Plate, raw) | bins | max off chord A1 (untr.) | 1−chord/arc A1 (untr.) | uncleaned 1−c/a | cos(chord,Σw) | cos(chord,w) | tail flag | retained w / Σw | split-half cos w (shuf. max) | MLP>lin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **t01_smooth_or_features** — 3 exclusive alternatives | | | | | | | | | | | | | | | | |
| or_features_a01_smooth | PROXY of t04_spiral_a09_no_spiral | 0.872 | 0.758 | 0.115 | 0.716 | 0.149 / 0.851 | 20 | 0.175 (0.243–0.268) | 0.165 (0.351–0.381) | 0.339 | 0.992 | 0.062 | — | 0.81 / 0.99 | 0.575 (0.072) | no |
| or_features_a02_features_or_disk | PROXY of t04_spiral_a08_spiral | 0.875 | 0.763 | 0.112 | 0.697 | 0.536 / 0.856 | 20 | 0.167 (0.198–0.229) | 0.161 (0.259–0.320) | 0.334 | 0.993 | 0.063 | — | 0.80 / 0.99 | 0.513 (0.014) | no |
| or_features_a03_star_or_artifact | WEAK | 0.588 | 0.606 | -0.017 | 0.953 | 0.755 / 0.756 | 20 | 0.231 (0.316–0.374) | 0.505 (0.462–0.529) | 0.502 | 0.919 | 0.249 | — | 0.63 / 0.99 | 0.364 (0.132) | **YES** |
| **t02_edgeon** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | |
| a04_yes | PROXY of t07_rounded_a18_cigar_shaped | 0.765 | 0.796 | -0.031 | 0.888 | 0.752 / 0.758 | 20 | 0.176 (0.204–0.225) | 0.143 (0.375–0.407) | 0.257 | 0.990 | 0.047 | — | 0.92 / 0.98 | 0.787 (0.050) | no |
| a05_no | PROXY of t07_rounded_a18_cigar_shaped | 0.790 | 0.821 | -0.031 | 0.873 | 0.359 / 0.780 | 20 | 0.178 (0.166–0.185) | 0.115 (0.309–0.324) | 0.244 | 0.991 | 0.049 | **FLAG** | 0.92 / 0.98 | 0.730 (0.051) | no |
| **t03_bar** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | |
| a06_bar | PROXY of t07_rounded_a16_completely_round | 0.656 | 0.620 | 0.037 | 0.889 | 0.438 / 0.641 | 20 | 0.287 (0.188–0.197) | 0.338 (0.358–0.411) | 0.391 | 0.965 | 0.045 | — | 0.78 / 0.96 | 0.301 (-0.003) | no |
| a07_no_bar | PROXY of t01_smooth_or_features_a02_features_or_disk | 0.700 | 0.641 | 0.059 | 0.871 | 0.474 / 0.687 | 20 | 0.266 (0.149–0.161) | 0.296 (0.329–0.372) | 0.340 | 0.973 | 0.046 | — | 0.78 / 0.97 | 0.430 (0.040) | no |
| **t04_spiral** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | |
| a08_spiral | PROXY of t01_smooth_or_features_a01_smooth | 0.806 | 0.746 | 0.060 | 0.827 | 0.649 / 0.797 | 20 | 0.182 (0.263–0.287) | 0.134 (0.397–0.407) | 0.370 | 0.992 | 0.058 | **FLAG** | 0.84 / 0.99 | 0.442 (0.093) | no |
| a09_no_spiral | PROXY of t01_smooth_or_features_a01_smooth | 0.836 | 0.772 | 0.064 | 0.796 | 0.487 / 0.823 | 20 | 0.180 (0.267–0.285) | 0.130 (0.391–0.410) | 0.353 | 0.992 | 0.058 | — | 0.83 / 0.99 | 0.532 (0.036) | no |
| **t05_bulge_prominence** — 4 exclusive alternatives | | | | | | | | | | | | | | | | |
| prominence_a10_no_bulge | STRONG | 0.822 | 0.660 | 0.162 | 0.823 | 0.234 / 0.804 | 20 | 0.292 (0.130–0.139) | 0.351 (0.380–0.431) | 0.220 | 0.963 | 0.045 | — | 0.82 / 0.97 | 0.598 (0.131) | no |
| prominence_a11_just_noticeable | PROXY of t01_smooth_or_features_a01_smooth | 0.687 | 0.623 | 0.064 | 0.887 | 0.495 / 0.663 | 20 | 0.238 (0.295–0.337) | 0.228 (0.415–0.451) | 0.304 | 0.995 | 0.042 | **FLAG** | 0.84 / 0.99 | 0.258 (0.065) | no |
| prominence_a12_obvious | PROXY of t04_spiral_a08_spiral | 0.780 | 0.670 | 0.110 | 0.886 | 0.419 / 0.750 | 20 | 0.150 (0.305–0.371) | 0.122 (0.498–0.565) | 0.299 | 0.996 | 0.038 | — | 0.87 / 0.99 | 0.472 (0.076) | no |
| prominence_a13_dominant | PROXY of t01_smooth_or_features_a01_smooth | 0.743 | 0.594 | 0.149 | 0.788 | 0.667 / 0.763 | 20 | 0.368 (0.300–0.352) | 0.507 (0.417–0.496) | 0.295 | 0.968 | 0.044 | — | 0.59 / 0.99 | 0.218 (0.035) | no |
| **t06_odd** — complements (two answers; fractions sum to 1) | | | | | | | | | | | | | | | | |
| a14_yes | PROXY of t07_rounded_a18_cigar_shaped | 0.693 | 0.706 | -0.013 | 0.877 | 0.530 / 0.681 | 20 | 0.353 (0.362–0.406) | 0.376 (0.576–0.638) | 0.315 | 0.957 | 0.041 | — | 0.83 / 0.98 | 0.566 (0.055) | no |
| a15_no | PROXY of t07_rounded_a18_cigar_shaped | 0.692 | 0.707 | -0.015 | 0.880 | 0.562 / 0.679 | 20 | 0.356 (0.353–0.407) | 0.381 (0.558–0.639) | 0.329 | 0.959 | 0.042 | — | 0.83 / 0.98 | 0.531 (0.140) | no |
| **t07_rounded** — 3 exclusive alternatives | | | | | | | | | | | | | | | | |
| a16_completely_round | PROXY of t03_bar_a06_bar | 0.725 | 0.708 | 0.017 | 0.910 | 0.377 / 0.711 | 20 | 0.226 (0.190–0.214) | 0.230 (0.273–0.288) | 0.464 | 0.993 | 0.047 | — | 0.87 / 0.97 | 0.642 (0.042) | no |
| a17_in_between | PROXY of t04_spiral_a09_no_spiral | 0.595 | 0.617 | -0.022 | 0.934 | 0.558 / 0.597 | 20 | 0.308 (0.248–0.265) | 0.344 (0.508–0.530) | 0.394 | 0.983 | 0.045 | — | 0.86 / 0.99 | 0.405 (0.070) | no |
| a18_cigar_shaped | PROXY of t02_edgeon_a05_no | 0.813 | 0.828 | -0.015 | 0.831 | 0.372 / 0.794 | 20 | 0.171 (0.148–0.172) | 0.129 (0.236–0.257) | 0.277 | 0.982 | 0.052 | — | 0.92 / 0.97 | 0.792 (0.112) | no |
| **t08_odd_feature** — 7 exclusive alternatives | | | | | | | | | | | | | | | | |
| feature_a19_ring | PROXY of t03_bar_a06_bar | 0.692 | 0.678 | 0.014 | 0.819 | 0.367 / 0.705 | 20 | 0.158 (0.074–0.093) | 0.174 (0.034–0.051) | 0.434 | 0.983 | 0.051 | — | 0.75 / 0.98 | 0.382 (0.124) | no |
| feature_a20_lens_or_arc | PROXY of t07_rounded_a17_in_between | 0.705 | 0.670 | 0.034 | 0.930 | 0.679 / 0.711 | 20 | 0.319 (0.359–0.379) | 0.351 (0.421–0.520) | 0.344 | 0.959 | 0.046 | — | 0.71 / 0.99 | 0.270 (0.098) | no |
| feature_a21_disturbed | STRONG | 0.643 | 0.625 | 0.018 | 0.947 | 0.377 / 0.653 | 20 | 0.351 (0.075–0.091) | 0.437 (0.019–0.028) | 0.438 | 0.976 | 0.034 | **FLAG** | 0.73 / 0.99 | 0.270 (0.081) | no |
| feature_a22_irregular | STRONG | 0.744 | 0.663 | 0.081 | 0.953 | 0.676 / 0.748 | 20 | 0.292 (0.077–0.094) | 0.265 (0.056–0.095) | 0.299 | 0.984 | 0.034 | — | 0.84 / 0.99 | 0.347 (0.201) | no |
| feature_a23_other | PROXY of t04_spiral_a08_spiral | 0.661 | 0.645 | 0.016 | 0.939 | 0.507 / 0.656 | 20 | 0.331 (0.109–0.124) | 0.396 (0.124–0.213) | 0.325 | 0.959 | 0.039 | — | 0.81 / 1.00 | 0.235 (0.110) | no |
| feature_a24_merger | STRONG | 0.640 | 0.643 | -0.003 | 0.838 | 0.610 / 0.666 | 20 | 0.346 (0.434–0.470) | 0.398 (0.704–0.733) | 0.315 | 0.970 | 0.041 | — | 0.72 / 0.99 | 0.205 (0.065) | no |
| feature_a38_dust_lane | STRONG | 0.698 | 0.762 | -0.063 | 0.843 | 0.295 / 0.757 | 20 | 0.238 (0.108–0.141) | 0.363 (0.228–0.254) | 0.553 | 0.948 | 0.090 | — | 0.67 / 0.98 | 0.272 (0.086) | no |
| **t09_bulge_shape** — 3 exclusive alternatives | | | | | | | | | | | | | | | | |
| shape_a25_rounded | STRONG | 0.669 | 0.620 | 0.049 | 0.837 | 0.443 / 0.680 | 20 | 0.272 (0.205–0.232) | 0.272 (0.496–0.561) | 0.348 | 0.984 | 0.050 | — | 0.77 / 0.96 | 0.355 (0.048) | no |
| shape_a26_boxy | NOISE | 0.570 | 0.574 | -0.004 | 0.939 | 0.536 / 0.575 | 20 | 0.335 (0.170–0.222) | 0.526 (0.560–0.654) | 0.590 | 0.982 | 0.050 | — | 0.62 / 0.99 | 0.057 (0.101) | no |
| shape_a27_no_bulge | STRONG | 0.700 | 0.628 | 0.072 | 0.825 | 0.534 / 0.710 | 20 | 0.273 (0.210–0.233) | 0.327 (0.469–0.477) | 0.264 | 0.984 | 0.052 | — | 0.76 / 0.96 | 0.341 (0.108) | no |
| **t10_arms_winding** — 3 exclusive alternatives | | | | | | | | | | | | | | | | |
| winding_a28_tight | STRONG | 0.576 | 0.569 | 0.007 | 0.969 | 0.548 / 0.587 | 20 | 0.382 (0.132–0.167) | 0.550 (0.150–0.245) | 0.576 | 0.927 | 0.044 | — | 0.75 / 0.98 | 0.277 (-0.030) | no |
| winding_a29_medium | STRONG | 0.530 | 0.514 | 0.016 | 0.980 | 0.516 / 0.540 | 20 | 0.400 (0.230–0.283) | 0.681 (0.532–0.631) | 0.536 | 0.975 | 0.041 | — | 0.59 / 0.98 | 0.081 (0.074) | no |
| winding_a30_loose | PROXY of t05_bulge_prominence_a11_just_noticeable | 0.627 | 0.618 | 0.010 | 0.972 | 0.506 / 0.621 | 20 | 0.346 (0.310–0.372) | 0.431 (0.465–0.569) | 0.592 | 0.980 | 0.039 | — | 0.81 / 0.99 | 0.342 (0.070) | no |
| **t11_arms_number** — 6 exclusive alternatives | | | | | | | | | | | | | | | | |
| number_a31_1 | STRONG | 0.614 | 0.568 | 0.046 | 0.956 | 0.472 / 0.605 | 20 | 0.329 (0.367–0.427) | 0.462 (0.618–0.666) | 0.500 | 0.985 | 0.039 | — | 0.58 / 0.99 | 0.195 (0.072) | no |
| number_a32_2 | STRONG | 0.573 | 0.568 | 0.005 | 0.970 | 0.516 / 0.577 | 20 | 0.373 (0.429–0.451) | 0.550 (0.640–0.712) | 0.565 | 0.951 | 0.046 | — | 0.73 / 0.94 | 0.155 (0.085) | no |
| number_a33_3 | NOISE | 0.518 | 0.521 | -0.003 | 0.984 | 0.551 / 0.488 | 20 | 0.292 (0.153–0.176) | 0.512 (0.272–0.321) | 0.534 | 0.979 | 0.056 | — | 0.66 / 0.99 | 0.087 (0.158) | no |
| number_a34_4 | STRONG | 0.569 | 0.565 | 0.003 | 0.984 | 0.388 / 0.633 | 20 | 0.277 (0.202–0.269) | 0.579 (0.522–0.642) | 0.624 | 0.970 | 0.090 | — | 0.57 / 0.99 | 0.086 (0.055) | no |
| number_a36_more_than_4 | NOISE | 0.657 | 0.593 | 0.065 | 0.942 | 0.426 / 0.675 | 20 | 0.341 (0.405–0.419) | 0.504 (0.639–0.713) | 0.452 | 0.982 | 0.066 | — | 0.63 / 0.98 | 0.028 (0.087) | no |
| number_a37_cant_tell | PROXY of t07_rounded_a17_in_between | 0.601 | 0.597 | 0.004 | 0.936 | 0.502 / 0.591 | 20 | 0.314 (0.401–0.433) | 0.332 (0.670–0.694) | 0.304 | 0.973 | 0.044 | — | 0.75 / 0.99 | 0.298 (0.068) | no |

## Candidate two-concept planes (STRONG pairs, |cos Σw| < 0.3)

| a | b | cos(Σw) | same question |
|---|---|---|---|
| t09_bulge_shape_a27_no_bulge | t11_arms_number_a32_2 | +0.027 | no |
| t09_bulge_shape_a27_no_bulge | t11_arms_number_a31_1 | +0.036 | no |
| t09_bulge_shape_a25_rounded | t11_arms_number_a32_2 | +0.057 | no |
| t08_odd_feature_a21_disturbed | t11_arms_number_a31_1 | +0.078 | no |
| t08_odd_feature_a21_disturbed | t11_arms_number_a32_2 | +0.080 | no |
| t09_bulge_shape_a25_rounded | t11_arms_number_a31_1 | -0.102 | no |
| t09_bulge_shape_a25_rounded | t10_arms_winding_a29_medium | +0.102 | no |
| t08_odd_feature_a24_merger | t10_arms_winding_a29_medium | -0.102 | no |
| t08_odd_feature_a24_merger | t10_arms_winding_a28_tight | -0.103 | no |
| t08_odd_feature_a24_merger | t09_bulge_shape_a25_rounded | +0.105 | no |
| t05_bulge_prominence_a10_no_bulge | t11_arms_number_a31_1 | +0.112 | no |
| t10_arms_winding_a29_medium | t11_arms_number_a32_2 | +0.117 | no |
| t10_arms_winding_a28_tight | t11_arms_number_a31_1 | +0.131 | no |
| t09_bulge_shape_a27_no_bulge | t10_arms_winding_a29_medium | -0.165 | no |
| t08_odd_feature_a38_dust_lane | t11_arms_number_a32_2 | +0.170 | no |
| t08_odd_feature_a22_irregular | t08_odd_feature_a38_dust_lane | +0.171 | yes |
| t08_odd_feature_a22_irregular | t09_bulge_shape_a25_rounded | -0.199 | no |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a38_dust_lane | +0.202 | no |
| t08_odd_feature_a22_irregular | t11_arms_number_a32_2 | -0.220 | no |
| t08_odd_feature_a22_irregular | t08_odd_feature_a24_merger | -0.226 | yes |
| t08_odd_feature_a22_irregular | t11_arms_number_a31_1 | +0.232 | no |
| t08_odd_feature_a24_merger | t09_bulge_shape_a27_no_bulge | -0.240 | no |
| t08_odd_feature_a38_dust_lane | t10_arms_winding_a29_medium | -0.251 | no |
| t05_bulge_prominence_a10_no_bulge | t11_arms_number_a32_2 | -0.260 | no |
| t08_odd_feature_a21_disturbed | t11_arms_number_a34_4 | +0.266 | no |
| t10_arms_winding_a28_tight | t11_arms_number_a32_2 | +0.266 | no |
| t11_arms_number_a32_2 | t11_arms_number_a34_4 | -0.276 | yes |
| t08_odd_feature_a21_disturbed | t08_odd_feature_a38_dust_lane | +0.276 | yes |
| t05_bulge_prominence_a10_no_bulge | t08_odd_feature_a24_merger | -0.292 | no |
| t08_odd_feature_a24_merger | t08_odd_feature_a38_dust_lane | +0.294 | yes |

## Proxy readings

| answer | label | top match outside its question | cos | residual AUC | residual split-half cos w |
|---|---|---|---|---|---|
| t01_smooth_or_features_a01_smooth | PROXY of t04_spiral_a09_no_spiral | t04_spiral_a09_no_spiral | +0.989 | 0.556 | 0.022 |
| t01_smooth_or_features_a02_features_or_disk | PROXY of t04_spiral_a08_spiral | t04_spiral_a08_spiral | +0.986 | 0.563 | 0.038 |
| t01_smooth_or_features_a03_star_or_artifact | WEAK | t11_arms_number_a34_4 | +0.628 | — | — |
| t02_edgeon_a04_yes | PROXY of t07_rounded_a18_cigar_shaped | t07_rounded_a18_cigar_shaped | +0.918 | 0.594 | 0.153 |
| t02_edgeon_a05_no | PROXY of t07_rounded_a18_cigar_shaped | t07_rounded_a18_cigar_shaped | -0.945 | 0.594 | 0.054 |
| t03_bar_a06_bar | PROXY of t07_rounded_a16_completely_round | t07_rounded_a16_completely_round | -0.869 | 0.601 | 0.317 |
| t03_bar_a07_no_bar | PROXY of t01_smooth_or_features_a02_features_or_disk | t01_smooth_or_features_a02_features_or_disk | -0.879 | 0.606 | 0.326 |
| t04_spiral_a08_spiral | PROXY of t01_smooth_or_features_a01_smooth | t01_smooth_or_features_a01_smooth | -0.988 | 0.615 | 0.276 |
| t04_spiral_a09_no_spiral | PROXY of t01_smooth_or_features_a01_smooth | t01_smooth_or_features_a01_smooth | +0.989 | 0.630 | 0.304 |
| t05_bulge_prominence_a10_no_bulge | STRONG | t09_bulge_shape_a27_no_bulge | +0.704 | — | — |
| t05_bulge_prominence_a11_just_noticeable | PROXY of t01_smooth_or_features_a01_smooth | t01_smooth_or_features_a01_smooth | -0.952 | 0.584 | 0.295 |
| t05_bulge_prominence_a12_obvious | PROXY of t04_spiral_a08_spiral | t04_spiral_a08_spiral | -0.953 | 0.604 | 0.354 |
| t05_bulge_prominence_a13_dominant | PROXY of t01_smooth_or_features_a01_smooth | t01_smooth_or_features_a01_smooth | +0.902 | 0.564 | 0.231 |
| t06_odd_a14_yes | PROXY of t07_rounded_a18_cigar_shaped | t07_rounded_a18_cigar_shaped | -0.818 | 0.689 | 0.578 |
| t06_odd_a15_no | PROXY of t07_rounded_a18_cigar_shaped | t07_rounded_a18_cigar_shaped | +0.819 | 0.688 | 0.556 |
| t07_rounded_a16_completely_round | PROXY of t03_bar_a06_bar | t03_bar_a06_bar | -0.869 | 0.692 | 0.664 |
| t07_rounded_a17_in_between | PROXY of t04_spiral_a09_no_spiral | t04_spiral_a09_no_spiral | -0.893 | 0.580 | 0.388 |
| t07_rounded_a18_cigar_shaped | PROXY of t02_edgeon_a05_no | t02_edgeon_a05_no | -0.945 | 0.610 | 0.114 |
| t08_odd_feature_a19_ring | PROXY of t03_bar_a06_bar | t03_bar_a06_bar | +0.840 | 0.617 | 0.371 |
| t08_odd_feature_a20_lens_or_arc | PROXY of t07_rounded_a17_in_between | t07_rounded_a17_in_between | -0.860 | 0.628 | 0.366 |
| t08_odd_feature_a21_disturbed | STRONG | t07_rounded_a16_completely_round | -0.642 | — | — |
| t08_odd_feature_a22_irregular | STRONG | t11_arms_number_a33_3 | +0.871 | — | — |
| t08_odd_feature_a23_other | PROXY of t04_spiral_a08_spiral | t04_spiral_a08_spiral | -0.815 | 0.592 | 0.288 |
| t08_odd_feature_a24_merger | STRONG | t11_arms_number_a36_more_than_4 | +0.769 | — | — |
| t08_odd_feature_a38_dust_lane | STRONG | t10_arms_winding_a28_tight | +0.797 | — | — |
| t09_bulge_shape_a25_rounded | STRONG | t05_bulge_prominence_a10_no_bulge | -0.667 | — | — |
| t09_bulge_shape_a26_boxy | NOISE | t11_arms_number_a31_1 | +0.688 | — | — |
| t09_bulge_shape_a27_no_bulge | STRONG | t07_rounded_a18_cigar_shaped | +0.726 | — | — |
| t10_arms_winding_a28_tight | STRONG | t08_odd_feature_a38_dust_lane | +0.797 | — | — |
| t10_arms_winding_a29_medium | STRONG | t11_arms_number_a34_4 | -0.589 | — | — |
| t10_arms_winding_a30_loose | PROXY of t05_bulge_prominence_a11_just_noticeable | t05_bulge_prominence_a11_just_noticeable | -0.943 | 0.566 | 0.272 |
| t11_arms_number_a31_1 | STRONG | t09_bulge_shape_a26_boxy | +0.688 | — | — |
| t11_arms_number_a32_2 | STRONG | t06_odd_a14_yes | -0.602 | — | — |
| t11_arms_number_a33_3 | NOISE | t08_odd_feature_a22_irregular | +0.871 | — | — |
| t11_arms_number_a34_4 | STRONG | t08_odd_feature_a38_dust_lane | +0.662 | — | — |
| t11_arms_number_a36_more_than_4 | NOISE | t08_odd_feature_a24_merger | +0.769 | — | — |
| t11_arms_number_a37_cant_tell | PROXY of t07_rounded_a17_in_between | t07_rounded_a17_in_between | -0.827 | 0.564 | 0.300 |
