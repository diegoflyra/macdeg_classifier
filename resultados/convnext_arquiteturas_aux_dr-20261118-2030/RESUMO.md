# convnext_arquiteturas_aux_dr-20261118-2030

- commit: `c424c0107c921dd390b503a2dab9efa9d5c176b4`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-21 13:50

Referência: `convnext_arquiteturas_aux_dr-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| convnext_arquiteturas_aux_dr-02 | `{"model": "convnext_base", "augment": {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "rotate_interpolation": "bilinear", "color": 0.1}, "aux_dr": 0.5}` | 0.803 [0.785, 0.817] | 0.973 | 0.906 | 0.980 | 0.840 | 0.792 | empatou |
| convnext_arquiteturas_aux_dr-00 | `{"model": "convnext_tiny", "augment": {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "rotate_interpolation": "bilinear", "color": 0.1}, "aux_dr": 0.5}` | 0.802 [0.783, 0.819] | 0.974 | 0.893 | 0.983 | 0.861 | 0.792 | referência |
| convnext_arquiteturas_aux_dr-03 | `{"model": "convnext_large", "augment": {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "rotate_interpolation": "bilinear", "color": 0.1}, "aux_dr": 0.5}` | 0.793 [0.774, 0.808] | 0.974 | 0.907 | 0.977 | 0.832 | 0.792 | empatou |
| convnext_arquiteturas_aux_dr-01 | `{"model": "convnext_small", "augment": {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "rotate_interpolation": "bilinear", "color": 0.1}, "aux_dr": 0.5}` | 0.786 [0.767, 0.802] | 0.971 | 0.888 | 0.979 | 0.838 | 0.792 | perdeu |

**Aninhado** (escolha da melhor configuração feita só com validação interna): F2 médio por base 0.798 [0.7792, 0.8139]; escolhida por dobra: {0: 'convnext_arquiteturas_aux_dr-00', 1: 'convnext_arquiteturas_aux_dr-02', 2: 'convnext_arquiteturas_aux_dr-02', 3: 'convnext_arquiteturas_aux_dr-02', 4: 'convnext_arquiteturas_aux_dr-03'}

## convnext_arquiteturas_aux_dr-00

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.881 | 0.963 | 0.920 | 0.879 |
| brset | 261 | 8881 | 0.808 | 0.991 | 0.885 | 0.983 |
| hyamd | 532 | 560 | 0.853 | 0.833 | 0.959 | 0.371 |
| odir | 280 | 4362 | 0.676 | 0.924 | 0.771 | 0.940 |
| rfmid | 179 | 2059 | 0.794 | 0.973 | 0.860 | 0.951 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0001 | 0.0032 | 0.0167 |
| outra | 5507 | 0.0 | 0.0016 | 0.4586 | 0.1371 |
| drusas | 2832 | 0.0 | 0.0018 | 0.9654 | 0.2433 |
| dmri | 1515 | 0.1977 | 0.9969 | 1.0 | 0.8904 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.85 (n=206), descolamento 1.00 (n=25), edema_macular 0.97 (n=395), glaucoma 0.98 (n=671), hemorragia 0.85 (n=85), membrana_epirretiniana 0.86 (n=390), miopia 0.87 (n=616), nao_especificada 0.88 (n=662), nevo 0.96 (n=97), oclusao_vascular 0.93 (n=326), outra 0.87 (n=671), retinopatia_diabetica 0.47 (n=671), retinopatia_hipertensiva 0.96 (n=404)

## convnext_arquiteturas_aux_dr-01

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.861 | 0.951 | 0.905 | 0.861 |
| brset | 261 | 8881 | 0.799 | 0.989 | 0.897 | 0.979 |
| hyamd | 532 | 560 | 0.824 | 0.813 | 0.934 | 0.302 |
| odir | 280 | 4362 | 0.659 | 0.926 | 0.789 | 0.923 |
| rfmid | 179 | 2059 | 0.787 | 0.969 | 0.872 | 0.942 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0001 | 0.0029 | 0.0227 |
| outra | 5507 | 0.0 | 0.0012 | 0.3675 | 0.1622 |
| drusas | 2832 | 0.0 | 0.0019 | 0.8427 | 0.2768 |
| dmri | 1515 | 0.0916 | 0.9922 | 1.0 | 0.897 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.83 (n=206), descolamento 1.00 (n=25), edema_macular 0.95 (n=395), glaucoma 0.97 (n=671), hemorragia 0.84 (n=85), membrana_epirretiniana 0.86 (n=390), miopia 0.82 (n=616), nao_especificada 0.86 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.91 (n=326), outra 0.85 (n=671), retinopatia_diabetica 0.41 (n=671), retinopatia_hipertensiva 0.94 (n=404)

## convnext_arquiteturas_aux_dr-02

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.866 | 0.951 | 0.913 | 0.858 |
| brset | 261 | 8881 | 0.807 | 0.986 | 0.897 | 0.981 |
| hyamd | 532 | 560 | 0.842 | 0.827 | 0.955 | 0.318 |
| odir | 280 | 4362 | 0.672 | 0.930 | 0.814 | 0.920 |
| rfmid | 179 | 2059 | 0.828 | 0.974 | 0.911 | 0.949 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0001 | 0.0035 | 0.0221 |
| outra | 5507 | 0.0001 | 0.0015 | 0.4136 | 0.1705 |
| drusas | 2832 | 0.0001 | 0.0017 | 0.9276 | 0.2818 |
| dmri | 1515 | 0.1541 | 0.9946 | 1.0 | 0.9116 |

Especificidade por doença: catarata 0.99 (n=288), cicatriz 0.84 (n=206), descolamento 1.00 (n=25), edema_macular 0.95 (n=395), glaucoma 0.98 (n=671), hemorragia 0.88 (n=85), membrana_epirretiniana 0.85 (n=390), miopia 0.83 (n=616), nao_especificada 0.86 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.89 (n=326), outra 0.85 (n=671), retinopatia_diabetica 0.42 (n=671), retinopatia_hipertensiva 0.95 (n=404)

## convnext_arquiteturas_aux_dr-03

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.854 | 0.954 | 0.901 | 0.850 |
| brset | 261 | 8881 | 0.805 | 0.990 | 0.904 | 0.979 |
| hyamd | 532 | 560 | 0.845 | 0.829 | 0.962 | 0.304 |
| odir | 280 | 4362 | 0.666 | 0.934 | 0.807 | 0.920 |
| rfmid | 179 | 2059 | 0.796 | 0.973 | 0.911 | 0.930 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0002 | 0.0049 | 0.0211 |
| outra | 5507 | 0.0002 | 0.0027 | 0.4325 | 0.1656 |
| drusas | 2832 | 0.0001 | 0.002 | 0.8931 | 0.2549 |
| dmri | 1515 | 0.1992 | 0.9917 | 0.9997 | 0.9063 |

Especificidade por doença: catarata 0.99 (n=288), cicatriz 0.80 (n=206), descolamento 1.00 (n=25), edema_macular 0.96 (n=395), glaucoma 0.98 (n=671), hemorragia 0.86 (n=85), membrana_epirretiniana 0.85 (n=390), miopia 0.81 (n=616), nao_especificada 0.85 (n=662), nevo 0.96 (n=97), oclusao_vascular 0.89 (n=326), outra 0.85 (n=671), retinopatia_diabetica 0.41 (n=671), retinopatia_hipertensiva 0.95 (n=404)
