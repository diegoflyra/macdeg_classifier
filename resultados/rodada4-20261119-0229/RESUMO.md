# rodada4-20261119-0229

- commit: `c756e834da3262b584ff2a73e405295a5103ed0d`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-19 20:59

Referência: `rodada4-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| rodada4-00 | `{}` | 0.804 [0.784, 0.822] | 0.977 | 0.851 | 0.990 | 0.899 | 0.792 | referência |
| rodada4-01 | `{"model": "inception_v3.tv_in1k"}` | 0.795 [0.775, 0.811] | 0.974 | 0.886 | 0.985 | 0.856 | 0.792 | empatou |
| rodada4-02 | `{"model": "vit_base_patch16_384.augreg_in21k_ft_in1k"}` | 0.465 [0.448, 0.479] | 0.826 | 0.933 | 0.464 | 0.218 | 0.792 | perdeu |

**Aninhado** (escolha da melhor configuração feita só com validação interna): F2 médio por base 0.797 [0.7768, 0.814]; escolhida por dobra: {0: 'rodada4-00', 1: 'rodada4-00', 2: 'rodada4-01', 3: 'rodada4-00', 4: 'rodada4-00'}

## rodada4-00

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.879 | 0.969 | 0.897 | 0.917 |
| brset | 261 | 8881 | 0.795 | 0.991 | 0.831 | 0.988 |
| hyamd | 532 | 560 | 0.822 | 0.810 | 0.904 | 0.436 |
| odir | 280 | 4362 | 0.699 | 0.940 | 0.725 | 0.970 |
| rfmid | 179 | 2059 | 0.826 | 0.982 | 0.855 | 0.972 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0009 | 0.0746 | 0.0091 |
| outra | 5507 | 0.0011 | 0.0619 | 0.5725 | 0.0944 |
| drusas | 2832 | 0.0009 | 0.0639 | 0.8296 | 0.1688 |
| dmri | 1515 | 0.4484 | 0.946 | 0.9948 | 0.8422 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.843 (n=424 x 560, hyamd), dmri_precoce 0.679 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.89 (n=206), descolamento 1.00 (n=25), edema_macular 0.99 (n=395), glaucoma 0.99 (n=671), hemorragia 0.93 (n=85), membrana_epirretiniana 0.95 (n=390), miopia 0.93 (n=616), nao_especificada 0.92 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.98 (n=326), outra 0.91 (n=671), retinopatia_diabetica 0.53 (n=671), retinopatia_hipertensiva 0.97 (n=404)

## rodada4-01

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.866 | 0.957 | 0.924 | 0.837 |
| brset | 261 | 8881 | 0.815 | 0.989 | 0.893 | 0.983 |
| hyamd | 532 | 560 | 0.840 | 0.815 | 0.945 | 0.350 |
| odir | 280 | 4362 | 0.684 | 0.933 | 0.761 | 0.948 |
| rfmid | 179 | 2059 | 0.768 | 0.975 | 0.838 | 0.947 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0013 | 0.0571 | 0.0106 |
| outra | 5507 | 0.0003 | 0.027 | 0.5117 | 0.1088 |
| drusas | 2832 | 0.0017 | 0.041 | 0.7969 | 0.19 |
| dmri | 1515 | 0.3568 | 0.9331 | 0.9965 | 0.8647 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 33 | 85 | 140 | 16524 | 117 | 414 |
| base | adam | 263 | 3 | 10 | 17 | 662 | 6 | 59 |
| base | brset | 261 | 6 | 22 | 22 | 8881 | 28 | 76 |
| base | hyamd | 532 | 7 | 29 | 22 | 560 | 35 | 83 |
| base | odir | 280 | 10 | 20 | 57 | 4362 | 33 | 128 |
| base | rfmid | 179 | 7 | 4 | 22 | 2059 | 15 | 68 |
| estagio_dmri | dmri | 983 | 26 | 56 | 118 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 0 | 16 | 15 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 7 | 13 | 7 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 6 | 24 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 1 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 2 | 9 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 4 | 3 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 1 | 6 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 4 | 19 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 14 | 46 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 6 | 59 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 0 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 29 | 86 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 3 | 20 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 11 | 43 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 35 | 87 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 2 | 10 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.851 (n=424 x 560, hyamd), dmri_precoce 0.672 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.81 (n=206), descolamento 0.96 (n=25), edema_macular 0.97 (n=395), glaucoma 0.99 (n=671), hemorragia 0.87 (n=85), membrana_epirretiniana 0.91 (n=390), miopia 0.88 (n=616), nao_especificada 0.84 (n=662), nevo 0.97 (n=97), oclusao_vascular 0.93 (n=326), outra 0.86 (n=671), retinopatia_diabetica 0.45 (n=671), retinopatia_hipertensiva 0.95 (n=404)

## rodada4-02

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.684 | 0.740 | 0.981 | 0.128 |
| brset | 261 | 8881 | 0.200 | 0.796 | 0.828 | 0.534 |
| hyamd | 532 | 560 | 0.819 | 0.549 | 0.987 | 0.014 |
| odir | 280 | 4362 | 0.248 | 0.559 | 0.857 | 0.201 |
| rfmid | 179 | 2059 | 0.372 | 0.827 | 0.978 | 0.291 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.1793 | 0.3432 | 0.556 | 0.4217 |
| outra | 5507 | 0.2512 | 0.4694 | 0.7103 | 0.7051 |
| drusas | 2832 | 0.2403 | 0.3214 | 0.4906 | 0.3309 |
| dmri | 1515 | 0.3832 | 0.638 | 0.7898 | 0.9096 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 67 | 191 | 34 | 16524 | 53 | 9601 |
| base | adam | 263 | 4 | 26 | 1 | 662 | 3 | 525 |
| base | brset | 261 | 28 | 27 | 17 | 8881 | 21 | 4053 |
| base | hyamd | 532 | 7 | 51 | 0 | 560 | 3 | 239 |
| base | odir | 280 | 27 | 64 | 13 | 4362 | 22 | 3377 |
| base | rfmid | 179 | 1 | 23 | 3 | 2059 | 4 | 1407 |
| estagio_dmri | dmri | 983 | 60 | 140 | 34 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 6 | 31 | 0 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 1 | 20 | 0 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 228 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 4 | 115 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 16 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 2 | 267 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 2 | 421 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 1 | 53 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 3 | 292 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 0 | 489 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 3 | 525 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 1 | 47 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 21 | 5815 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 0 | 262 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 11 | 479 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 3 | 328 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 2 | 264 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.550 (n=424 x 560, hyamd), dmri_precoce 0.545 (n=108 x 560, hyamd)

Especificidade por doença: catarata 0.21 (n=288), cicatriz 0.35 (n=206), descolamento 0.36 (n=25), edema_macular 0.32 (n=395), glaucoma 0.37 (n=671), hemorragia 0.32 (n=85), membrana_epirretiniana 0.21 (n=390), miopia 0.14 (n=616), nao_especificada 0.13 (n=662), nevo 0.51 (n=97), oclusao_vascular 0.18 (n=326), outra 0.21 (n=671), retinopatia_diabetica 0.04 (n=671), retinopatia_hipertensiva 0.32 (n=404)
