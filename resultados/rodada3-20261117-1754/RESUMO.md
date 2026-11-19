# rodada3-20261117-1754

- commit: `272148187ecec672863554fff1bf842770cdaa4d`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-18 21:23

Referência: `rodada3-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| rodada3-03 | `{"model": "tf_efficientnet_b4.ns_jft_in1k"}` | 0.810 [0.789, 0.827] | 0.975 | 0.880 | 0.985 | 0.878 | 0.792 | empatou |
| rodada3-02 | `{"model": "tf_efficientnetv2_s.in21k_ft_in1k"}` | 0.809 [0.791, 0.826] | 0.977 | 0.890 | 0.986 | 0.868 | 0.792 | empatou |
| rodada3-00 | `{}` | 0.801 [0.782, 0.818] | 0.973 | 0.883 | 0.986 | 0.867 | 0.792 | referência |
| rodada3-01 | `{"model": "densenet121.tv_in1k"}` | 0.793 [0.775, 0.811] | 0.973 | 0.885 | 0.985 | 0.853 | 0.792 | empatou |
| rodada3-04 | `{"crop": 0.6}` | 0.761 [0.741, 0.780] | 0.968 | 0.857 | 0.984 | 0.837 | 0.792 | perdeu |

**Aninhado** (escolha da melhor configuração feita só com validação interna): F2 médio por base 0.809 [0.7905, 0.8274]; escolhida por dobra: {0: 'rodada3-03', 1: 'rodada3-03', 2: 'rodada3-01', 3: 'rodada3-02', 4: 'rodada3-03'}

## rodada3-00

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.866 | 0.957 | 0.951 | 0.787 |
| brset | 261 | 8881 | 0.813 | 0.991 | 0.854 | 0.988 |
| hyamd | 532 | 560 | 0.848 | 0.809 | 0.955 | 0.355 |
| odir | 280 | 4362 | 0.670 | 0.932 | 0.732 | 0.953 |
| rfmid | 179 | 2059 | 0.808 | 0.968 | 0.849 | 0.965 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0001 | 0.0016 | 0.0654 | 0.0123 |
| outra | 5507 | 0.0006 | 0.0375 | 0.6109 | 0.128 |
| drusas | 2832 | 0.0013 | 0.0443 | 0.8624 | 0.1984 |
| dmri | 1515 | 0.4221 | 0.9569 | 0.9993 | 0.8772 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.849 (n=424 x 560, hyamd), dmri_precoce 0.650 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.89 (n=206), descolamento 1.00 (n=25), edema_macular 0.99 (n=395), glaucoma 0.99 (n=671), hemorragia 0.89 (n=85), membrana_epirretiniana 0.93 (n=390), miopia 0.90 (n=616), nao_especificada 0.79 (n=662), nevo 0.99 (n=97), oclusao_vascular 0.97 (n=326), outra 0.89 (n=671), retinopatia_diabetica 0.46 (n=671), retinopatia_hipertensiva 0.96 (n=404)

## rodada3-01

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.864 | 0.961 | 0.920 | 0.838 |
| brset | 261 | 8881 | 0.793 | 0.988 | 0.858 | 0.984 |
| hyamd | 532 | 560 | 0.839 | 0.799 | 0.955 | 0.304 |
| odir | 280 | 4362 | 0.682 | 0.927 | 0.771 | 0.943 |
| rfmid | 179 | 2059 | 0.790 | 0.977 | 0.844 | 0.957 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0001 | 0.002 | 0.0717 | 0.0134 |
| outra | 5507 | 0.0007 | 0.0474 | 0.6473 | 0.1327 |
| drusas | 2832 | 0.002 | 0.0431 | 0.8335 | 0.1836 |
| dmri | 1515 | 0.4316 | 0.9662 | 0.9989 | 0.8726 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 47 | 50 | 127 | 16524 | 184 | 280 |
| base | adam | 263 | 10 | 2 | 11 | 662 | 58 | 24 |
| base | brset | 261 | 8 | 9 | 29 | 8881 | 17 | 58 |
| base | hyamd | 532 | 12 | 12 | 12 | 560 | 30 | 59 |
| base | odir | 280 | 9 | 20 | 55 | 4362 | 57 | 101 |
| base | rfmid | 179 | 8 | 7 | 20 | 2059 | 22 | 38 |
| estagio_dmri | dmri | 983 | 35 | 38 | 115 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 5 | 7 | 8 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 7 | 5 | 4 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 1 | 1 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 2 | 19 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 1 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 0 | 3 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 1 | 5 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 3 | 1 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 4 | 23 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 21 | 38 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 58 | 24 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 0 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 46 | 66 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 6 | 11 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 6 | 22 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 31 | 59 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 5 | 6 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.848 (n=424 x 560, hyamd), dmri_precoce 0.608 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.81 (n=206), descolamento 0.96 (n=25), edema_macular 0.98 (n=395), glaucoma 0.99 (n=671), hemorragia 0.92 (n=85), membrana_epirretiniana 0.88 (n=390), miopia 0.87 (n=616), nao_especificada 0.84 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.96 (n=326), outra 0.87 (n=671), retinopatia_diabetica 0.42 (n=671), retinopatia_hipertensiva 0.96 (n=404)

## rodada3-02

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.871 | 0.967 | 0.901 | 0.893 |
| brset | 261 | 8881 | 0.835 | 0.992 | 0.908 | 0.984 |
| hyamd | 532 | 560 | 0.834 | 0.806 | 0.945 | 0.312 |
| odir | 280 | 4362 | 0.710 | 0.941 | 0.779 | 0.955 |
| rfmid | 179 | 2059 | 0.797 | 0.981 | 0.855 | 0.956 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0001 | 0.0506 | 0.0224 |
| outra | 5507 | 0.0001 | 0.0278 | 0.5337 | 0.175 |
| drusas | 2832 | 0.0004 | 0.0537 | 0.8129 | 0.2624 |
| dmri | 1515 | 0.4181 | 0.9614 | 0.9981 | 0.9228 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 49 | 59 | 118 | 16524 | 265 | 266 |
| base | adam | 263 | 15 | 2 | 11 | 662 | 81 | 11 |
| base | brset | 261 | 5 | 19 | 19 | 8881 | 33 | 68 |
| base | hyamd | 532 | 15 | 10 | 14 | 560 | 40 | 64 |
| base | odir | 280 | 10 | 23 | 52 | 4362 | 85 | 79 |
| base | rfmid | 179 | 4 | 5 | 22 | 2059 | 26 | 44 |
| estagio_dmri | dmri | 983 | 34 | 49 | 104 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 7 | 6 | 9 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 8 | 4 | 5 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 1 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 9 | 13 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 1 | 10 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 4 | 6 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 2 | 4 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 11 | 8 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 26 | 29 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 81 | 11 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 0 | 1 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 61 | 67 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 5 | 17 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 21 | 27 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 40 | 65 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 3 | 8 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.833 (n=424 x 560, hyamd), dmri_precoce 0.699 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.87 (n=206), descolamento 1.00 (n=25), edema_macular 0.97 (n=395), glaucoma 0.99 (n=671), hemorragia 0.87 (n=85), membrana_epirretiniana 0.94 (n=390), miopia 0.89 (n=616), nao_especificada 0.89 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.94 (n=326), outra 0.88 (n=671), retinopatia_diabetica 0.42 (n=671), retinopatia_hipertensiva 0.95 (n=404)

## rodada3-03

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.877 | 0.966 | 0.928 | 0.856 |
| brset | 261 | 8881 | 0.820 | 0.994 | 0.893 | 0.984 |
| hyamd | 532 | 560 | 0.825 | 0.795 | 0.912 | 0.414 |
| odir | 280 | 4362 | 0.705 | 0.937 | 0.771 | 0.955 |
| rfmid | 179 | 2059 | 0.822 | 0.973 | 0.866 | 0.965 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.001 | 0.0748 | 0.0103 |
| outra | 5507 | 0.0004 | 0.0495 | 0.62 | 0.0944 |
| drusas | 2832 | 0.0013 | 0.0786 | 0.8572 | 0.179 |
| dmri | 1515 | 0.502 | 0.9718 | 0.9988 | 0.8455 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 66 | 61 | 116 | 16524 | 337 | 288 |
| base | adam | 263 | 7 | 1 | 12 | 662 | 68 | 22 |
| base | brset | 261 | 8 | 18 | 20 | 8881 | 38 | 77 |
| base | hyamd | 532 | 29 | 6 | 18 | 560 | 97 | 64 |
| base | odir | 280 | 15 | 26 | 49 | 4362 | 98 | 90 |
| base | rfmid | 179 | 7 | 10 | 17 | 2059 | 36 | 35 |
| estagio_dmri | dmri | 983 | 37 | 55 | 98 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 15 | 5 | 10 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 14 | 1 | 8 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 1 | 1 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 11 | 15 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 3 | 7 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 3 | 5 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 1 | 5 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 13 | 11 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 37 | 38 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 68 | 22 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 0 | 5 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 62 | 76 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 7 | 11 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 25 | 23 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 99 | 66 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 7 | 3 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.841 (n=424 x 560, hyamd), dmri_precoce 0.614 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.87 (n=206), descolamento 1.00 (n=25), edema_macular 0.98 (n=395), glaucoma 0.99 (n=671), hemorragia 0.85 (n=85), membrana_epirretiniana 0.94 (n=390), miopia 0.90 (n=616), nao_especificada 0.86 (n=662), nevo 0.94 (n=97), oclusao_vascular 0.96 (n=326), outra 0.89 (n=671), retinopatia_diabetica 0.51 (n=671), retinopatia_hipertensiva 0.97 (n=404)

## rodada3-04

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.846 | 0.952 | 0.897 | 0.838 |
| brset | 261 | 8881 | 0.789 | 0.987 | 0.904 | 0.976 |
| hyamd | 532 | 560 | 0.820 | 0.782 | 0.923 | 0.329 |
| odir | 280 | 4362 | 0.595 | 0.916 | 0.664 | 0.941 |
| rfmid | 179 | 2059 | 0.757 | 0.960 | 0.832 | 0.942 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0002 | 0.004 | 0.1176 | 0.0182 |
| outra | 5507 | 0.0014 | 0.0801 | 0.7001 | 0.1743 |
| drusas | 2832 | 0.0027 | 0.0706 | 0.9098 | 0.2338 |
| dmri | 1515 | 0.4533 | 0.9501 | 0.9987 | 0.8805 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 78 | 38 | 139 | 16524 | 210 | 400 |
| base | adam | 263 | 15 | 1 | 12 | 662 | 56 | 22 |
| base | brset | 261 | 2 | 15 | 23 | 8881 | 13 | 124 |
| base | hyamd | 532 | 25 | 8 | 16 | 560 | 50 | 65 |
| base | odir | 280 | 28 | 9 | 66 | 4362 | 72 | 124 |
| base | rfmid | 179 | 8 | 5 | 22 | 2059 | 19 | 65 |
| estagio_dmri | dmri | 983 | 53 | 30 | 123 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 9 | 6 | 9 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 16 | 2 | 7 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 5 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 4 | 22 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 1 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 1 | 16 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 2 | 10 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 2 | 5 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 9 | 11 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 16 | 78 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 56 | 22 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 0 | 2 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 54 | 82 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 2 | 23 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 10 | 41 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 51 | 69 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 3 | 13 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.825 (n=424 x 560, hyamd), dmri_precoce 0.616 (n=108 x 560, hyamd)

Especificidade por doença: catarata 0.98 (n=288), cicatriz 0.81 (n=206), descolamento 0.96 (n=25), edema_macular 0.95 (n=395), glaucoma 0.98 (n=671), hemorragia 0.86 (n=85), membrana_epirretiniana 0.93 (n=390), miopia 0.80 (n=616), nao_especificada 0.84 (n=662), nevo 0.97 (n=97), oclusao_vascular 0.91 (n=326), outra 0.84 (n=671), retinopatia_diabetica 0.43 (n=671), retinopatia_hipertensiva 0.94 (n=404)
