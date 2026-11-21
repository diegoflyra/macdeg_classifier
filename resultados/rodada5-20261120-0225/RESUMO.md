# rodada5-20261120-0225

- commit: `c0ce7aa26739be90a8b72416b19cb4c90a164827`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-21 17:33
- tarefas faltando: 31

Referência: `resultados/rodada4-20261119-0229/rodada4-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| rodada5-04 | `{"augment": {"crop_scale": [0.7, 1.0], "hflip": true, "rotate_deg": 30, "color": 0.25}}` | 0.811 [0.786, 0.835] | 0.976 | 0.863 | 0.990 | 0.895 | 0.793 | empatou |
| rodada5-02 | `{"model": "v2s_macula", "image_size": 1024, "cache": false}` | 0.811 [0.789, 0.832] | 0.978 | 0.872 | 0.988 | 0.889 | 0.792 | empatou |
| rodada5-01 | `{"aux_drusas": 0.5}` | 0.805 [0.785, 0.825] | 0.979 | 0.844 | 0.991 | 0.912 | 0.792 | empatou |

**Aninhado** (escolha da melhor configuração feita só com validação interna): F2 médio por base 0.814 [0.7832, 0.8418]; escolhida por dobra: {2: 'rodada5-02', 3: 'rodada5-01'}

## rodada5-01

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.868 | 0.965 | 0.886 | 0.914 |
| brset | 261 | 8881 | 0.822 | 0.993 | 0.858 | 0.989 |
| hyamd | 532 | 560 | 0.820 | 0.812 | 0.883 | 0.521 |
| odir | 280 | 4362 | 0.707 | 0.947 | 0.725 | 0.974 |
| rfmid | 179 | 2059 | 0.809 | 0.981 | 0.827 | 0.975 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0003 | 0.0803 | 0.0102 |
| outra | 5507 | 0.0002 | 0.054 | 0.5741 | 0.0933 |
| drusas | 2832 | 0.0003 | 0.116 | 0.9367 | 0.2225 |
| dmri | 1515 | 0.5062 | 0.9721 | 0.9976 | 0.8601 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 74 | 62 | 163 | 16524 | 216 | 138 |
| base | adam | 263 | 8 | 5 | 22 | 662 | 13 | 15 |
| base | brset | 261 | 13 | 20 | 24 | 8881 | 45 | 36 |
| base | hyamd | 532 | 29 | 18 | 33 | 560 | 96 | 48 |
| base | odir | 280 | 16 | 16 | 61 | 4362 | 44 | 27 |
| base | rfmid | 179 | 8 | 3 | 23 | 2059 | 18 | 12 |
| estagio_dmri | dmri | 983 | 45 | 44 | 130 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 12 | 8 | 23 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 17 | 10 | 10 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 12 | 8 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 4 | 6 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 3 | 0 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 1 | 4 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 9 | 3 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 18 | 8 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 13 | 15 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 2 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 37 | 27 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 2 | 4 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 13 | 13 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 96 | 49 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 6 | 0 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.849 (n=424 x 560, hyamd), dmri_precoce 0.667 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.91 (n=206), descolamento 1.00 (n=25), edema_macular 0.98 (n=395), glaucoma 0.99 (n=671), hemorragia 0.89 (n=85), membrana_epirretiniana 0.97 (n=390), miopia 0.95 (n=616), nao_especificada 0.91 (n=662), nevo 0.99 (n=97), oclusao_vascular 0.98 (n=326), outra 0.91 (n=671), retinopatia_diabetica 0.60 (n=671), retinopatia_hipertensiva 0.98 (n=404)

## rodada5-02

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 210 | 530 | 0.893 | 0.968 | 0.933 | 0.885 |
| brset | 209 | 7105 | 0.806 | 0.990 | 0.852 | 0.987 |
| hyamd | 426 | 448 | 0.835 | 0.834 | 0.923 | 0.429 |
| odir | 224 | 3489 | 0.692 | 0.939 | 0.746 | 0.959 |
| rfmid | 144 | 1647 | 0.829 | 0.983 | 0.861 | 0.971 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 8813 | 0.0 | 0.0006 | 0.0349 | 0.0121 |
| outra | 4406 | 0.0005 | 0.0241 | 0.5305 | 0.1164 |
| drusas | 2832 | 0.0009 | 0.0417 | 0.8766 | 0.2076 |
| dmri | 1213 | 0.4496 | 0.9702 | 0.9974 | 0.8928 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1213 | 44 | 56 | 111 | 13219 | 131 | 174 |
| base | adam | 210 | 2 | 7 | 12 | 530 | 7 | 22 |
| base | brset | 209 | 8 | 11 | 23 | 7105 | 32 | 27 |
| base | hyamd | 426 | 22 | 24 | 11 | 448 | 48 | 43 |
| base | odir | 224 | 7 | 12 | 50 | 3489 | 24 | 60 |
| base | rfmid | 144 | 5 | 2 | 15 | 1647 | 20 | 22 |
| estagio_dmri | dmri | 787 | 22 | 32 | 100 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 349 | 9 | 17 | 7 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 77 | 13 | 7 | 4 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 246 | 0 | 3 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 168 | 6 | 4 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 23 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 307 | 2 | 1 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 524 | 2 | 3 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 65 | 1 | 4 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 308 | 2 | 10 |
| doenca | miopia | 0 | 0 | 0 | 0 | 481 | 16 | 14 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 530 | 7 | 22 |
| doenca | nevo | 0 | 0 | 0 | 0 | 73 | 1 | 0 |
| doenca | normal | 0 | 0 | 0 | 0 | 8813 | 30 | 46 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 254 | 3 | 2 |
| doenca | outra | 0 | 0 | 0 | 0 | 553 | 9 | 17 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 543 | 49 | 44 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 331 | 3 | 4 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.860 (n=349 x 448, hyamd), dmri_precoce 0.716 (n=77 x 448, hyamd)

Especificidade por doença: catarata 0.99 (n=246), cicatriz 0.90 (n=168), descolamento 1.00 (n=23), edema_macular 0.99 (n=307), glaucoma 0.98 (n=524), hemorragia 0.86 (n=65), membrana_epirretiniana 0.92 (n=308), miopia 0.93 (n=481), nao_especificada 0.88 (n=530), nevo 0.99 (n=73), oclusao_vascular 0.99 (n=254), outra 0.90 (n=553), retinopatia_diabetica 0.53 (n=543), retinopatia_hipertensiva 0.96 (n=331)

## rodada5-04

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 158 | 397 | 0.860 | 0.970 | 0.880 | 0.907 |
| brset | 156 | 5329 | 0.840 | 0.994 | 0.891 | 0.988 |
| hyamd | 319 | 336 | 0.826 | 0.790 | 0.915 | 0.405 |
| odir | 168 | 2618 | 0.693 | 0.923 | 0.720 | 0.969 |
| rfmid | 107 | 1235 | 0.836 | 0.979 | 0.869 | 0.972 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 6611 | 0.0 | 0.0013 | 0.1003 | 0.0089 |
| outra | 3304 | 0.001 | 0.0639 | 0.5818 | 0.0917 |
| drusas | 2832 | 0.0014 | 0.0803 | 0.8314 | 0.1631 |
| dmri | 908 | 0.4871 | 0.964 | 0.9989 | 0.8458 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 908 | 24 | 45 | 100 | 9915 | 71 | 127 |
| base | adam | 158 | 4 | 2 | 15 | 397 | 8 | 21 |
| base | brset | 156 | 3 | 16 | 14 | 5329 | 18 | 26 |
| base | hyamd | 319 | 9 | 14 | 18 | 336 | 16 | 37 |
| base | odir | 168 | 7 | 10 | 40 | 2618 | 23 | 31 |
| base | rfmid | 107 | 1 | 3 | 13 | 1235 | 6 | 12 |
| estagio_dmri | dmri | 589 | 15 | 31 | 82 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 245 | 2 | 6 | 9 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 74 | 7 | 8 | 9 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 145 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 123 | 7 | 4 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 16 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 229 | 1 | 1 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 429 | 1 | 1 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 49 | 0 | 3 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 234 | 4 | 1 |
| doenca | miopia | 0 | 0 | 0 | 0 | 385 | 3 | 13 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 397 | 8 | 21 |
| doenca | nevo | 0 | 0 | 0 | 0 | 59 | 0 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 6611 | 20 | 25 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 193 | 1 | 3 |
| doenca | outra | 0 | 0 | 0 | 0 | 402 | 7 | 14 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 404 | 17 | 37 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 239 | 2 | 3 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.846 (n=245 x 336, hyamd), dmri_precoce 0.608 (n=74 x 336, hyamd)

Especificidade por doença: catarata 1.00 (n=145), cicatriz 0.93 (n=123), descolamento 1.00 (n=16), edema_macular 0.99 (n=229), glaucoma 1.00 (n=429), hemorragia 0.90 (n=49), membrana_epirretiniana 0.97 (n=234), miopia 0.92 (n=385), nao_especificada 0.91 (n=397), nevo 0.97 (n=59), oclusao_vascular 0.97 (n=193), outra 0.90 (n=402), retinopatia_diabetica 0.50 (n=404), retinopatia_hipertensiva 0.97 (n=239)
