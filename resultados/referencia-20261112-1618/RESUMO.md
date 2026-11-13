# referencia-20261112-1618

- commit: `d0bd3f40186fac802f28f3b8894b07c1d6c82a01`
- assinatura dos dados: `72e45dfc1003894a`
- analisado: 2026-11-13 02:15

Referência: `referencia-00`

| configuração | mudanças | F2 médio por base [IC95] | AUROC | sens. | esp. normais | esp. outras | régua origem | veredito |
|---|---|---|---|---|---|---|---|---|
| referencia-00 | `{}` | 0.770 [0.749, 0.789] | 0.967 | 0.853 | 0.986 | 0.852 | 0.792 | referência |

## referencia-00

| base | DMRI | sem DMRI | F2 | AUROC | sens. | esp. |
|---|---|---|---|---|---|---|
| adam | 263 | 662 | 0.850 | 0.951 | 0.905 | 0.834 |
| brset | 261 | 8881 | 0.762 | 0.988 | 0.824 | 0.983 |
| hyamd | 532 | 560 | 0.814 | 0.774 | 0.915 | 0.327 |
| odir | 280 | 4362 | 0.646 | 0.924 | 0.718 | 0.946 |
| rfmid | 179 | 2059 | 0.778 | 0.961 | 0.844 | 0.950 |

Distribuição das notas (quantis) e fração acima do limiar:

| tipo | n | p10 | p50 | p90 | acima do limiar |
|---|---|---|---|---|---|
| normal | 11017 | 0.0001 | 0.0049 | 0.0995 | 0.0107 |
| outra | 5507 | 0.001 | 0.072 | 0.6022 | 0.1378 |
| drusas | 2832 | 0.0045 | 0.061 | 0.7799 | 0.1822 |
| dmri | 1515 | 0.3444 | 0.9314 | 0.9993 | 0.8521 |

Especificidade por doença: catarata 0.99 (n=288), cicatriz 0.81 (n=206), descolamento 1.00 (n=25), edema_macular 0.96 (n=395), glaucoma 0.98 (n=671), hemorragia 0.89 (n=85), membrana_epirretiniana 0.90 (n=390), miopia 0.89 (n=616), nao_especificada 0.83 (n=662), nevo 0.99 (n=97), oclusao_vascular 0.91 (n=326), outra 0.87 (n=671), retinopatia_diabetica 0.43 (n=671), retinopatia_hipertensiva 0.95 (n=404)
