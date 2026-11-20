# Fluxo obrigatório de experimentos

**Leia antes de criar ou mudar qualquer experimento.** Vale para pessoas e agentes (Claude, Codex, etc.).
Se a tarefa pedida não couber nestas regras, **pare e pergunte aos donos do projeto** antes de escrever
código ou lançar jobs. Regras gerais do projeto: `CLAUDE.md`; motivos: `docs/decisoes.md`.

## 1. Branches

- `main`: código estável. A fase 1 (rodadas 1–4: comparação de arquiteturas importadas) entrou nela
  por PR da `experimentos-002` (19/11/2026).
- `experimentos-003`: **todo** o trabalho da fase 2 (rede permutada, drusas, explicabilidade, Optuna).
  Branches pessoais saem dela e voltam para ela (merge ou PR). Ao fim da fase, PR para a `main`.
- `experimentos-002`: encerrada (fase 1). Não acrescente commits.
- `resultados`: escrita só pelo pipeline. Nunca à mão. O que foi treinado, com o commit de cada lote, está
  no catálogo `docs/experimentos.md` (`python -m macdeg.catalogo` depois de `git fetch origin resultados`).
- Branch nova de fase só com decisão dos donos.

## 2. Rodadas: uma mudança por vez sobre a base vigente

1. Cada rodada tem uma **base** (`base: configs/train/baseline.yaml` + `base_mudancas`). Hoje a base vigente
   é a da rodada 4 (`configs/experimentos/008_rodada4.yaml`): a da rodada 3 com a EfficientNetV2-S
   (`rodada3-02`), por decisão dos donos (19/11/2026):
   ```yaml
   base: configs/train/baseline.yaml
   base_mudancas:
     augment: {crop_scale: [0.85, 1.0], hflip: true, rotate_deg: 15, color: 0.1}
     aux_dr: 0.5
     model: tf_efficientnetv2_s.in21k_ft_in1k
   ```
   O 004 do Gustavo (AlexNet, ConvNeXt) continua sobre a base da rodada 3 (ResNet-50), contra a `rodada3-00`.
2. **Cada variação muda UMA coisa** sobre essa base. Duas mudanças na mesma variação não dizem qual ajudou.
3. O veredito pareado (`ganhou` / `empatou` / `perdeu`) é sempre contra a base vigente. Para usar a base de
   outro lote, `referencia: resultados/<lote>/<configuração>` (sem retreinar a referência por conta própria).
   Se a base já tem treinos suficientes, `treinar_base: false` pula a 00 (economiza 20 tarefas); a
   `referencia` passa a ser obrigatória e precisa ter a mesma assinatura dos dados. O `--teste` treina a 00
   mesmo assim. A rodada 5 usa isso (a V2-S base já tem dois treinos: `rodada3-02` e `rodada4-00`).
4. O que **ganhou** entra na base da rodada seguinte, por decisão dos donos.
5. **Hiperparâmetros fixos** (receita da `baseline.yaml`: AdamW, lr 3e-4, weight decay 1e-4, 20 épocas,
   paciência 5, lote 16, 512 px) em todas as rodadas. O ajuste fino de hiperparâmetros é a **última**
   rodada, sobre a base final. Estudo de hiperparâmetros de uma arquitetura específica só como apêndice,
   com referência explícita (ver o exemplo da AlexNet abaixo), e com aprovação dos donos.
6. Só 512 px (1024 px perdeu na rodada 2 e custa 2–4× mais).

## 3. YAML de experimento: formato e numeração

- Arquivo novo em `configs/experimentos/NNN_nome.yaml`, com **número único e crescente** (veja os que já
  existem; o teste `test_experiment_files_have_unique_numbers` falha se repetir). Nunca edite um YAML já
  lançado: crie outro.
- Não crie arquivos novos em `configs/train/`: a base é sempre `baseline.yaml` + `base_mudancas`.
- Opção de treino nova = chave de configuração com padrão que mantém o comportamento antigo, implementada
  em `macdeg/` com teste.
- Referência com marcador (`<AAAAMMDD-HHMM>`) é recusada no lançamento: preencha com o lote real.

Modelo:
```yaml
nome: meu_experimento
base: configs/train/baseline.yaml
base_mudancas:            # copie a base vigente (seção 2)
  augment: {crop_scale: [0.85, 1.0], hflip: true, rotate_deg: 15, color: 0.1}
  aux_dr: 0.5
grade: {}
variacoes:                # cada item = UMA mudança
  - {model: alexnet}
referencia: resultados/rodada3-<AAAAMMDD-HHMM>/rodada3-00
```

## 4. Experimentos

Concluídos (resultados em `docs/experimentos.md`): `001_referencia`, `002_rodada2`, `003_rodada3` e
`008_rodada4` (V2-S vira a base; Inception empatou; ViT não aprendeu com a receita comum).

| YAML | quem | o que | referência |
|---|---|---|---|
| `004_gustavo_arquiteturas.yaml` | Gustavo | AlexNet e ConvNeXt-T na base da rodada 3 (em andamento) | `rodada3-00` |
| `005_alexnet_congelamento.yaml` | Gustavo | apêndice AlexNet: congelar 3 convoluções | `gustavo-01` |
| `006_alexnet_taxas.yaml` | Gustavo | apêndice AlexNet: lr separada da saída | (dentro do lote) |
| `007_alexnet_inicializacao.yaml` | Gustavo | apêndice AlexNet: Xavier na saída | (dentro do lote) |
| `009_rodada5.yaml` | donos | V2-S + drusas, + recorte da mácula em 1024 px, + híbrida do Gustavo sobre a V2-S, + augmentation ampliada (precisa do `drusas_dobras.csv`) | `rodada5-00` |

Ordem: 004 só depois que a rodada 3 estiver no GitHub; 005 depois do 004; 006 e 007 em sequência, cada
um só se o anterior não perdeu. Os resultados da Parte 1 (004) entram na comparação de arquiteturas; os
da Parte 2 (005–007) são um estudo à parte da AlexNet.

## 5. Antes de lançar

- `python -m pytest -q tests/` passa inteiro.
- Commit, push e `git pull` no cluster (o lote registra o commit).
- `sbatch -p debug --qos=complex --time=00:30:00 cluster/experimento.sh <yaml> --teste` antes do lote real.
- Nada de imagens, IDs ou dados de pacientes fora do computador local/cluster (`CLAUDE.md`, seção 1).
