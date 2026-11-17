# rodada2-20261115-1245

- commit: `0aff867a4985a37474222743d4d4a4e3a3e0b219`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-17 14:38

Referência: `rodada2-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| rodada2-01 | `{"augment": {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "color": 0.1}}` | 0.776 [0.757, 0.795] | 0.970 | 0.864 | 0.986 | 0.858 | 0.792 | empatou |
| rodada2-04 | `{"aux_dr": 0.5}` | 0.775 [0.756, 0.794] | 0.966 | 0.865 | 0.984 | 0.850 | 0.792 | empatou |
| rodada2-03 | `{"preprocess": "clahe"}` | 0.769 [0.747, 0.788] | 0.968 | 0.830 | 0.990 | 0.875 | 0.792 | empatou |
| rodada2-00 | `{}` | 0.763 [0.743, 0.781] | 0.967 | 0.890 | 0.977 | 0.811 | 0.792 | referência |
| rodada2-02 | `{"oversample": {"retinopatia_diabetica": 3}}` | 0.763 [0.742, 0.782] | 0.964 | 0.830 | 0.988 | 0.862 | 0.792 | empatou |
| rodada2-05 | `{"image_size": 1024, "cache": false}` | 0.727 [0.705, 0.745] | 0.955 | 0.859 | 0.984 | 0.793 | 0.792 | perdeu |

**Aninhado** (escolha da melhor configuração feita só com validação interna): F2 médio por base 0.781 [0.7623, 0.7987]; escolhida por dobra: {0: 'rodada2-00', 1: 'rodada2-03', 2: 'rodada2-01', 3: 'rodada2-01', 4: 'rodada2-01'}

## rodada2-00

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.850 | 0.957 | 0.932 | 0.781 |
| brset | 261 | 8881 | 0.729 | 0.985 | 0.851 | 0.971 |
| hyamd | 532 | 560 | 0.842 | 0.773 | 0.970 | 0.248 |
| odir | 280 | 4362 | 0.640 | 0.915 | 0.754 | 0.927 |
| rfmid | 179 | 2059 | 0.756 | 0.957 | 0.860 | 0.928 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0005 | 0.0394 | 0.0134 |
| outra | 5507 | 0.0001 | 0.0219 | 0.522 | 0.1471 |
| drusas | 2832 | 0.0003 | 0.0209 | 0.7288 | 0.1949 |
| dmri | 1515 | 0.2669 | 0.9025 | 0.9993 | 0.8601 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.72 (n=206), descolamento 0.92 (n=25), edema_macular 0.93 (n=395), glaucoma 0.97 (n=671), hemorragia 0.86 (n=85), membrana_epirretiniana 0.87 (n=390), miopia 0.79 (n=616), nao_especificada 0.78 (n=662), nevo 0.95 (n=97), oclusao_vascular 0.90 (n=326), outra 0.84 (n=671), retinopatia_diabetica 0.37 (n=671), retinopatia_hipertensiva 0.94 (n=404)

## rodada2-01

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.841 | 0.955 | 0.890 | 0.840 |
| brset | 261 | 8881 | 0.762 | 0.989 | 0.820 | 0.984 |
| hyamd | 532 | 560 | 0.848 | 0.793 | 0.966 | 0.305 |
| odir | 280 | 4362 | 0.654 | 0.924 | 0.714 | 0.952 |
| rfmid | 179 | 2059 | 0.775 | 0.969 | 0.821 | 0.959 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0001 | 0.0035 | 0.0927 | 0.0113 |
| outra | 5507 | 0.0008 | 0.0596 | 0.6451 | 0.1237 |
| drusas | 2832 | 0.0033 | 0.0654 | 0.8471 | 0.1829 |
| dmri | 1515 | 0.4005 | 0.9629 | 0.9995 | 0.8422 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 72 | 33 | 134 | 16524 | 526 | 172 |
| base | adam | 263 | 12 | 1 | 17 | 662 | 71 | 32 |
| base | brset | 261 | 17 | 9 | 30 | 8881 | 138 | 28 |
| base | hyamd | 532 | 10 | 8 | 8 | 560 | 62 | 30 |
| base | odir | 280 | 22 | 11 | 58 | 4362 | 173 | 64 |
| base | rfmid | 179 | 11 | 4 | 21 | 2059 | 82 | 18 |
| estagio_dmri | dmri | 983 | 62 | 25 | 126 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 8 | 6 | 4 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 2 | 2 | 4 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 29 | 2 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 1 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 19 | 3 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 16 | 7 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 5 | 3 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 32 | 6 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 70 | 22 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 71 | 32 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 2 | 0 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 142 | 45 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 21 | 2 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 42 | 17 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 65 | 31 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 11 | 2 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.85 (n=206), descolamento 0.96 (n=25), edema_macular 0.97 (n=395), glaucoma 0.99 (n=671), hemorragia 0.88 (n=85), membrana_epirretiniana 0.94 (n=390), miopia 0.86 (n=616), nao_especificada 0.84 (n=662), nevo 0.97 (n=97), oclusao_vascular 0.96 (n=326), outra 0.88 (n=671), retinopatia_diabetica 0.42 (n=671), retinopatia_hipertensiva 0.96 (n=404)

## rodada2-02

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.856 | 0.948 | 0.920 | 0.819 |
| brset | 261 | 8881 | 0.748 | 0.987 | 0.812 | 0.982 |
| hyamd | 532 | 560 | 0.794 | 0.779 | 0.855 | 0.498 |
| odir | 280 | 4362 | 0.654 | 0.912 | 0.714 | 0.952 |
| rfmid | 179 | 2059 | 0.761 | 0.960 | 0.832 | 0.945 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0003 | 0.0042 | 0.1042 | 0.0121 |
| outra | 5507 | 0.0016 | 0.0762 | 0.5036 | 0.1424 |
| drusas | 2832 | 0.0032 | 0.0487 | 0.7264 | 0.1896 |
| dmri | 1515 | 0.2513 | 0.8611 | 0.9982 | 0.8337 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 128 | 38 | 129 | 16524 | 596 | 193 |
| base | adam | 263 | 10 | 7 | 11 | 662 | 53 | 28 |
| base | brset | 261 | 22 | 12 | 27 | 8881 | 136 | 42 |
| base | hyamd | 532 | 64 | 3 | 13 | 560 | 162 | 22 |
| base | odir | 280 | 23 | 12 | 57 | 4362 | 173 | 64 |
| base | rfmid | 179 | 9 | 4 | 21 | 2059 | 72 | 37 |
| estagio_dmri | dmri | 983 | 64 | 35 | 116 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 28 | 3 | 7 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 36 | 0 | 6 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 24 | 4 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 1 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 17 | 4 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 12 | 3 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 6 | 1 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 25 | 11 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 64 | 37 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 53 | 28 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 4 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 162 | 40 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 14 | 11 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 36 | 18 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 165 | 23 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 13 | 12 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.82 (n=206), descolamento 0.96 (n=25), edema_macular 0.96 (n=395), glaucoma 0.99 (n=671), hemorragia 0.92 (n=85), membrana_epirretiniana 0.91 (n=390), miopia 0.83 (n=616), nao_especificada 0.82 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.91 (n=326), outra 0.87 (n=671), retinopatia_diabetica 0.58 (n=671), retinopatia_hipertensiva 0.94 (n=404)

## rodada2-03

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.853 | 0.951 | 0.886 | 0.878 |
| brset | 261 | 8881 | 0.760 | 0.986 | 0.805 | 0.986 |
| hyamd | 532 | 560 | 0.811 | 0.773 | 0.895 | 0.409 |
| odir | 280 | 4362 | 0.649 | 0.919 | 0.689 | 0.960 |
| rfmid | 179 | 2059 | 0.769 | 0.962 | 0.816 | 0.958 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0002 | 0.0031 | 0.0576 | 0.0098 |
| outra | 5507 | 0.0017 | 0.0475 | 0.5029 | 0.1277 |
| drusas | 2832 | 0.0018 | 0.0285 | 0.757 | 0.1875 |
| dmri | 1515 | 0.2607 | 0.9224 | 0.9995 | 0.8356 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 132 | 42 | 125 | 16524 | 665 | 176 |
| base | adam | 263 | 16 | 4 | 14 | 662 | 80 | 16 |
| base | brset | 261 | 25 | 13 | 26 | 8881 | 162 | 33 |
| base | hyamd | 532 | 46 | 6 | 10 | 560 | 133 | 43 |
| base | odir | 280 | 31 | 13 | 56 | 4362 | 199 | 55 |
| base | rfmid | 179 | 14 | 6 | 19 | 2059 | 91 | 29 |
| estagio_dmri | dmri | 983 | 86 | 36 | 115 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 16 | 6 | 4 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 30 | 0 | 6 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 29 | 9 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 2 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 18 | 7 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 15 | 5 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 8 | 1 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 30 | 7 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 86 | 15 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 80 | 16 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 4 | 0 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 172 | 33 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 20 | 11 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 50 | 21 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 137 | 43 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 14 | 8 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.82 (n=206), descolamento 1.00 (n=25), edema_macular 0.96 (n=395), glaucoma 0.99 (n=671), hemorragia 0.94 (n=85), membrana_epirretiniana 0.93 (n=390), miopia 0.90 (n=616), nao_especificada 0.88 (n=662), nevo 0.99 (n=97), oclusao_vascular 0.93 (n=326), outra 0.88 (n=671), retinopatia_diabetica 0.51 (n=671), retinopatia_hipertensiva 0.96 (n=404)

## rodada2-04

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.863 | 0.951 | 0.916 | 0.843 |
| brset | 261 | 8881 | 0.769 | 0.983 | 0.858 | 0.979 |
| hyamd | 532 | 560 | 0.828 | 0.766 | 0.934 | 0.327 |
| odir | 280 | 4362 | 0.655 | 0.916 | 0.725 | 0.948 |
| rfmid | 179 | 2059 | 0.760 | 0.964 | 0.816 | 0.952 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0012 | 0.0439 | 0.0143 |
| outra | 5507 | 0.0002 | 0.0265 | 0.4903 | 0.1547 |
| drusas | 2832 | 0.0013 | 0.0392 | 0.7889 | 0.2214 |
| dmri | 1515 | 0.2434 | 0.9304 | 0.9998 | 0.8607 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 78 | 41 | 126 | 16524 | 507 | 214 |
| base | adam | 263 | 10 | 6 | 12 | 662 | 62 | 21 |
| base | brset | 261 | 11 | 13 | 26 | 8881 | 128 | 61 |
| base | hyamd | 532 | 23 | 4 | 12 | 560 | 76 | 32 |
| base | odir | 280 | 24 | 16 | 53 | 4362 | 166 | 76 |
| base | rfmid | 179 | 10 | 2 | 23 | 2059 | 75 | 24 |
| estagio_dmri | dmri | 983 | 55 | 37 | 114 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 9 | 3 | 7 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 14 | 1 | 5 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 1 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 24 | 10 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 19 | 5 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 14 | 9 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 5 | 2 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 20 | 15 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 69 | 22 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 62 | 21 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 4 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 147 | 69 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 12 | 12 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 40 | 8 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 78 | 32 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 13 | 7 |

Especificidade por doença: catarata 0.99 (n=288), cicatriz 0.79 (n=206), descolamento 0.92 (n=25), edema_macular 0.97 (n=395), glaucoma 0.98 (n=671), hemorragia 0.89 (n=85), membrana_epirretiniana 0.88 (n=390), miopia 0.86 (n=616), nao_especificada 0.84 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.90 (n=326), outra 0.89 (n=671), retinopatia_diabetica 0.43 (n=671), retinopatia_hipertensiva 0.96 (n=404)

## rodada2-05

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.846 | 0.942 | 0.962 | 0.711 |
| brset | 261 | 8881 | 0.694 | 0.977 | 0.774 | 0.976 |
| hyamd | 532 | 560 | 0.842 | 0.765 | 0.989 | 0.163 |
| odir | 280 | 4362 | 0.572 | 0.879 | 0.646 | 0.935 |
| rfmid | 179 | 2059 | 0.681 | 0.939 | 0.777 | 0.919 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0015 | 0.0255 | 0.1723 | 0.0162 |
| outra | 5507 | 0.0081 | 0.1269 | 0.7002 | 0.2065 |
| drusas | 2832 | 0.0212 | 0.1623 | 0.8746 | 0.2662 |
| dmri | 1515 | 0.3337 | 0.9003 | 0.9994 | 0.8587 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 101 | 54 | 113 | 16524 | 440 | 469 |
| base | adam | 263 | 3 | 11 | 7 | 662 | 20 | 66 |
| base | brset | 261 | 31 | 11 | 28 | 8881 | 151 | 104 |
| base | hyamd | 532 | 2 | 12 | 4 | 560 | 21 | 69 |
| base | odir | 280 | 44 | 14 | 55 | 4362 | 177 | 142 |
| base | rfmid | 179 | 21 | 6 | 19 | 2059 | 71 | 88 |
| estagio_dmri | dmri | 983 | 99 | 42 | 109 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 0 | 6 | 4 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 2 | 6 | 0 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 31 | 7 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 2 | 1 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 17 | 10 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 9 | 23 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 7 | 4 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 29 | 29 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 54 | 85 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 20 | 66 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 4 | 0 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 163 | 92 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 16 | 25 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 51 | 42 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 22 | 75 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 15 | 10 |

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.84 (n=206), descolamento 0.96 (n=25), edema_macular 0.95 (n=395), glaucoma 0.95 (n=671), hemorragia 0.89 (n=85), membrana_epirretiniana 0.87 (n=390), miopia 0.74 (n=616), nao_especificada 0.71 (n=662), nevo 0.99 (n=97), oclusao_vascular 0.87 (n=326), outra 0.85 (n=671), retinopatia_diabetica 0.29 (n=671), retinopatia_hipertensiva 0.95 (n=404)
