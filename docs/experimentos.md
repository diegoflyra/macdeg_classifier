# Catálogo dos experimentos

Gerado por `python -m macdeg.catalogo` a partir da branch `resultados` (não edite à mão).
Código exato de cada lote: `git show <commit>:macdeg/train.py` (ou `git checkout <commit>`).
Métrica principal: média do F2 dentro de cada base, limiar escolhido em outras dobras.
Veredito: diferença pareada contra a referência do lote, IC95 por paciente.

| lote | criado | commit | experimento | configurações | assinatura dos dados |
|---|---|---|---|---|---|
| [referencia-20261112-1618](#referencia-20261112-1618) | 2026-11-12 16:18 | `d0bd3f40` | referencia | 1 | `72e45dfc1003894a` |
| [rodada2-20261115-1245](#rodada2-20261115-1245) | 2026-11-15 12:45 | `0aff867a` | rodada2 | 6 | `72e45dfc1003894a` |
| [rodada3-20261117-1754](#rodada3-20261117-1754) | 2026-11-17 17:54 | `27214818` | rodada3 | 5 | `72e45dfc1003894a` |
| [rodada4-20261119-0229](#rodada4-20261119-0229) | 2026-11-19 02:29 | `c756e834` | rodada4 | 3 | `72e45dfc1003894a` |

## referencia-20261112-1618

- Criado em 2026-11-12 16:18; commit `d0bd3f40186fac802f28f3b8894b07c1d6c82a01`.
- Tarefas: 20; faltando na análise: 0.
- Mudanças da base (`base_mudancas`): (base).
- Referência do veredito: `referencia-00`.
- Receita comum: `model: "resnet50"`, `pretrained: true`, `image_size: 512`, `batch_size: 16`, `epochs: 20`, `patience: 5`, `lr: 0.0003`, `weight_decay: 0.0001`, `warmup_epochs: 1`, `balance: "pos_weight"`.

| configuração | mudança | F2 médio por base [IC95] | Δ contra a referência [IC95] | AUROC | AUPRC | sensibilidade | esp. em DR | DMRI precoce × DR (AUROC) | melhor época (mediana) | GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| `referencia-00` | (base) | 0,770 [0,749; 0,789] | referência | 0,967 | 0,776 | 0,853 | 0,432 | — | 9,0 | 9,1 h (117 s/época) |

## rodada2-20261115-1245

- Criado em 2026-11-15 12:45; commit `0aff867a4985a37474222743d4d4a4e3a3e0b219`.
- Tarefas: 120; faltando na análise: 0.
- Mudanças da base (`base_mudancas`): (base).
- Referência do veredito: `rodada2-00`.
- Receita comum: `model: "resnet50"`, `pretrained: true`, `image_size: 512`, `batch_size: 16`, `epochs: 20`, `patience: 5`, `lr: 0.0003`, `weight_decay: 0.0001`, `warmup_epochs: 1`, `balance: "pos_weight"`.

| configuração | mudança | F2 médio por base [IC95] | Δ contra a referência [IC95] | AUROC | AUPRC | sensibilidade | esp. em DR | DMRI precoce × DR (AUROC) | melhor época (mediana) | GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| `rodada2-00` | (base) | 0,763 [0,743; 0,781] | referência | 0,967 | 0,768 | 0,890 | 0,365 | — | 10,0 | 11,8 h (136 s/época) |
| `rodada2-01` | `augment: {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "color": 0.1}` | 0,776 [0,757; 0,795] | 0,013 [-0,000; 0,028] **empatou** | 0,970 | 0,793 | 0,864 | 0,416 | — | 13,0 | 12,0 h (124 s/época) |
| `rodada2-02` | `oversample: {"retinopatia_diabetica": 3}` | 0,763 [0,742; 0,782] | -0,001 [-0,016; 0,015] **empatou** | 0,964 | 0,770 | 0,830 | 0,577 | — | 6,5 | 7,4 h (97 s/época) |
| `rodada2-03` | `preprocess: "clahe"` | 0,769 [0,747; 0,788] | 0,005 [-0,012; 0,021] **empatou** | 0,968 | 0,774 | 0,830 | 0,505 | — | 9,5 | 5,7 h (70 s/época) |
| `rodada2-04` | `aux_dr: 0.5` | 0,775 [0,756; 0,794] | 0,012 [-0,001; 0,025] **empatou** | 0,966 | 0,788 | 0,865 | 0,434 | — | 11,0 | 6,7 h (75 s/época) |
| `rodada2-05` | `image_size: 1024`; `cache: false` | 0,727 [0,705; 0,745] | -0,036 [-0,053; -0,021] **perdeu** | 0,955 | 0,711 | 0,859 | 0,286 | — | 8,0 | 17,1 h (213 s/época) |

Seleção aninhada (a melhor configuração escolhida só com a validação interna): F2 0,781 [0,762; 0,799]; escolhas: `rodada2-01` × 3, `rodada2-00` × 1, `rodada2-03` × 1.

Base deste lote (`rodada2-00`) contra a base do lote anterior (`referencia-00`): -0,007 [-0,023; 0,010], **empatou**.

## rodada3-20261117-1754

- Criado em 2026-11-17 17:54; commit `272148187ecec672863554fff1bf842770cdaa4d`.
- Tarefas: 100; faltando na análise: 0.
- Mudanças da base (`base_mudancas`): `augment: {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "color": 0.1}`; `aux_dr: 0.5`.
- Referência do veredito: `rodada3-00`.
- Receita comum: `model: "resnet50"`, `pretrained: true`, `image_size: 512`, `batch_size: 16`, `epochs: 20`, `patience: 5`, `lr: 0.0003`, `weight_decay: 0.0001`, `warmup_epochs: 1`, `balance: "pos_weight"`, `augment: {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "color": 0.1}`, `aux_dr: 0.5`.

| configuração | mudança | F2 médio por base [IC95] | Δ contra a referência [IC95] | AUROC | AUPRC | sensibilidade | esp. em DR | DMRI precoce × DR (AUROC) | melhor época (mediana) | GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| `rodada3-00` | (base) | 0,801 [0,782; 0,818] | referência | 0,973 | 0,804 | 0,883 | 0,459 | 0,650 | 12,5 | 7,2 h (73 s/época) |
| `rodada3-01` | `model: "densenet121.tv_in1k"` | 0,793 [0,775; 0,811] | -0,008 [-0,020; 0,005] **empatou** | 0,973 | 0,797 | 0,885 | 0,417 | 0,608 | 12,5 | 9,9 h (99 s/época) |
| `rodada3-02` | `model: "tf_efficientnetv2_s.in21k_ft_in1k"` | 0,809 [0,791; 0,826] | 0,008 [-0,005; 0,021] **empatou** | 0,977 | 0,816 | 0,890 | 0,422 | 0,699 | 8,0 | 8,4 h (105 s/época) |
| `rodada3-03` | `model: "tf_efficientnet_b4.ns_jft_in1k"` | 0,810 [0,789; 0,827] | 0,009 [-0,006; 0,024] **empatou** | 0,975 | 0,811 | 0,880 | 0,508 | 0,614 | 5,0 | 9,9 h (146 s/época) |
| `rodada3-04` | `crop: 0.6` | 0,761 [0,741; 0,780] | -0,040 [-0,054; -0,026] **perdeu** | 0,968 | 0,762 | 0,857 | 0,432 | 0,616 | 12,0 | 7,3 h (78 s/época) |

Seleção aninhada (a melhor configuração escolhida só com a validação interna): F2 0,809 [0,790; 0,827]; escolhas: `rodada3-03` × 3, `rodada3-01` × 1, `rodada3-02` × 1.

Base deste lote (`rodada3-00`) contra a base do lote anterior (`rodada2-00`): 0,038 [0,022; 0,053], **ganhou**.

## rodada4-20261119-0229

- Criado em 2026-11-19 02:29; commit `c756e834da3262b584ff2a73e405295a5103ed0d`.
- Tarefas: 60; faltando na análise: 0.
- Mudanças da base (`base_mudancas`): `augment: {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "color": 0.1}`; `aux_dr: 0.5`; `model: "tf_efficientnetv2_s.in21k_ft_in1k"`.
- Referência do veredito: `rodada4-00`.
- Receita comum: `model: "tf_efficientnetv2_s.in21k_ft_in1k"`, `pretrained: true`, `image_size: 512`, `batch_size: 16`, `epochs: 20`, `patience: 5`, `lr: 0.0003`, `weight_decay: 0.0001`, `warmup_epochs: 1`, `balance: "pos_weight"`, `augment: {"crop_scale": [0.85, 1.0], "hflip": true, "rotate_deg": 15, "color": 0.1}`, `aux_dr: 0.5`.

| configuração | mudança | F2 médio por base [IC95] | Δ contra a referência [IC95] | AUROC | AUPRC | sensibilidade | esp. em DR | DMRI precoce × DR (AUROC) | melhor época (mediana) | GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| `rodada4-00` | (base) | 0,804 [0,784; 0,822] | referência | 0,977 | 0,819 | 0,851 | 0,528 | 0,679 | 5,0 | 7,8 h (121 s/época) |
| `rodada4-01` | `model: "inception_v3.tv_in1k"` | 0,795 [0,775; 0,811] | -0,009 [-0,023; 0,004] **empatou** | 0,974 | 0,809 | 0,886 | 0,450 | 0,672 | 12,0 | 10,0 h (106 s/época) |
| `rodada4-02` | `model: "vit_base_patch16_384.augreg_in21k_ft_in1k"` | 0,465 [0,448; 0,479] | -0,340 [-0,359; -0,321] **perdeu** | 0,826 | 0,341 | 0,933 | 0,043 | 0,545 | 3,5 | 13,2 h (188 s/época) |

Seleção aninhada (a melhor configuração escolhida só com a validação interna): F2 0,797 [0,777; 0,814]; escolhas: `rodada4-00` × 4, `rodada4-01` × 1.

Base deste lote (`rodada4-00`) contra a base do lote anterior (`rodada3-00`): 0,003 [-0,010; 0,016], **empatou**.
