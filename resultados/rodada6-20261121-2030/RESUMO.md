# rodada6-20261121-2030

- commit: `4e80f14fd26356f02f746b5d356510e3c47dbcd2`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-22 10:50

Referência: `resultados/rodada4-20261119-0229/rodada4-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| rodada6-01 | `{"model": "hibrida_v2s", "image_size": 1024, "cache": false, "aux_localization": 0.25, "aux_drusas": 0.5}` | 0.810 [0.791, 0.829] | 0.977 | 0.855 | 0.992 | 0.902 | 0.792 | empatou |

## rodada6-01

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.874 | 0.964 | 0.897 | 0.906 |
| brset | 261 | 8881 | 0.835 | 0.994 | 0.885 | 0.988 |
| hyamd | 532 | 560 | 0.821 | 0.804 | 0.897 | 0.464 |
| odir | 280 | 4362 | 0.700 | 0.941 | 0.714 | 0.975 |
| rfmid | 179 | 2059 | 0.822 | 0.979 | 0.844 | 0.975 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0 | 0.0005 | 0.0638 | 0.0083 |
| outra | 5507 | 0.0003 | 0.0457 | 0.5862 | 0.0928 |
| drusas | 2832 | 0.0006 | 0.0634 | 0.9377 | 0.1931 |
| dmri | 1515 | 0.4683 | 0.9703 | 0.9985 | 0.8442 |

Discordâncias com a referência (mesmas imagens). DMRI: só a referência detectou (perdida) / só esta detectou (ganha); sem DMRI: falso positivo só da referência (corrigido) / só desta (criado):

| grupo | valor | DMRI | perdidas | ganhas | nenhuma detectou | sem DMRI | FP corrigidos | FP criados |
|---|---|---|---|---|---|---|---|---|
| geral | todas | 1515 | 55 | 60 | 165 | 16524 | 176 | 145 |
| base | adam | 263 | 5 | 5 | 22 | 662 | 11 | 18 |
| base | brset | 261 | 5 | 19 | 25 | 8881 | 31 | 36 |
| base | hyamd | 532 | 22 | 18 | 33 | 560 | 62 | 46 |
| base | odir | 280 | 18 | 15 | 62 | 4362 | 46 | 24 |
| base | rfmid | 179 | 5 | 3 | 23 | 2059 | 26 | 21 |
| estagio_dmri | dmri | 983 | 33 | 42 | 132 | 0 | 0 | 0 |
| estagio_dmri | dmri_intermediaria_tardia | 424 | 7 | 15 | 16 | 0 | 0 | 0 |
| estagio_dmri | dmri_precoce | 108 | 15 | 3 | 17 | 0 | 0 | 0 |
| doenca | catarata | 0 | 0 | 0 | 0 | 288 | 0 | 0 |
| doenca | cicatriz | 0 | 0 | 0 | 0 | 206 | 4 | 13 |
| doenca | descolamento | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| doenca | edema_macular | 0 | 0 | 0 | 0 | 395 | 3 | 4 |
| doenca | glaucoma | 0 | 0 | 0 | 0 | 671 | 3 | 0 |
| doenca | hemorragia | 0 | 0 | 0 | 0 | 85 | 1 | 4 |
| doenca | membrana_epirretiniana | 0 | 0 | 0 | 0 | 390 | 8 | 1 |
| doenca | miopia | 0 | 0 | 0 | 0 | 616 | 21 | 14 |
| doenca | nao_especificada | 0 | 0 | 0 | 0 | 662 | 11 | 18 |
| doenca | nevo | 0 | 0 | 0 | 0 | 97 | 0 | 0 |
| doenca | normal | 0 | 0 | 0 | 0 | 11017 | 38 | 23 |
| doenca | oclusao_vascular | 0 | 0 | 0 | 0 | 326 | 2 | 4 |
| doenca | outra | 0 | 0 | 0 | 0 | 671 | 19 | 14 |
| doenca | retinopatia_diabetica | 0 | 0 | 0 | 0 | 671 | 62 | 47 |
| doenca | retinopatia_hipertensiva | 0 | 0 | 0 | 0 | 404 | 4 | 3 |

AUROC por estágio de DMRI, contra os sem DMRI das mesmas bases: dmri_intermediaria_tardia 0.859 (n=424 x 560, hyamd), dmri_precoce 0.585 (n=108 x 560, hyamd)

Especificidade por doença: catarata 1.00 (n=288), cicatriz 0.85 (n=206), descolamento 1.00 (n=25), edema_macular 0.98 (n=395), glaucoma 0.99 (n=671), hemorragia 0.89 (n=85), membrana_epirretiniana 0.97 (n=390), miopia 0.94 (n=616), nao_especificada 0.91 (n=662), nevo 0.98 (n=97), oclusao_vascular 0.98 (n=326), outra 0.92 (n=671), retinopatia_diabetica 0.55 (n=671), retinopatia_hipertensiva 0.97 (n=404)
