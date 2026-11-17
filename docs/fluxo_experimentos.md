# Fluxo obrigatório de experimentos

**Leia antes de criar ou mudar qualquer experimento.** Vale para pessoas e agentes (Claude, Codex, etc.).
Se a tarefa pedida não couber nestas regras, **pare e pergunte aos donos do projeto** antes de escrever
código ou lançar jobs. Regras gerais do projeto: `CLAUDE.md`; motivos: `docs/decisoes.md`.

## 1. Branches

- `main`: só melhorias de desempenho (velocidade, infraestrutura). Nada de experimento.
- `experimentos-002`: **todo** o trabalho de experimentos (rodadas, novas opções de treino, YAMLs).
  Branches pessoais saem dela e voltam para ela (merge ou PR), nunca da `main`.
- `resultados`: escrita só pelo pipeline. Nunca à mão.
- Branch nova de fase (ex.: explicabilidade) só com decisão dos donos.

## 2. Rodadas: uma mudança por vez sobre a base vigente

1. Cada rodada tem uma **base** (`base: configs/train/baseline.yaml` + `base_mudancas`). Hoje a base vigente
   é a da rodada 3 (`configs/experimentos/003_rodada3.yaml`):
   ```yaml
   base: configs/train/baseline.yaml
   base_mudancas:
     augment: {crop_scale: [0.85, 1.0], hflip: true, rotate_deg: 15, color: 0.1}
     aux_dr: 0.5
   ```
2. **Cada variação muda UMA coisa** sobre essa base. Duas mudanças na mesma variação não dizem qual ajudou.
3. O veredito pareado (`ganhou` / `empatou` / `perdeu`) é sempre contra a base vigente. Para usar a base de
   outro lote, `referencia: resultados/<lote>/<configuração>` (sem retreinar a referência por conta própria).
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

## 4. Experimentos em andamento

| YAML | quem | o que | referência |
|---|---|---|---|
| `003_rodada3.yaml` | donos | DenseNet-121, EfficientNetV2-S, EfficientNet-B4, recorte da mácula | `rodada3-00` |
| `004_gustavo_arquiteturas.yaml` | Gustavo | AlexNet e ConvNeXt-T na base vigente | `rodada3-00` |
| `005_alexnet_congelamento.yaml` | Gustavo | apêndice AlexNet: congelar 3 convoluções | `gustavo-01` |
| `006_alexnet_taxas.yaml` | Gustavo | apêndice AlexNet: lr separada da saída | (dentro do lote) |
| `007_alexnet_inicializacao.yaml` | Gustavo | apêndice AlexNet: Xavier na saída | (dentro do lote) |

Ordem: 004 só depois que a rodada 3 estiver no GitHub; 005 depois do 004; 006 e 007 em sequência, cada
um só se o anterior não perdeu. Os resultados da Parte 1 (004) entram na comparação de arquiteturas; os
da Parte 2 (005–007) são um estudo à parte da AlexNet.

## 5. Antes de lançar

- `python -m pytest -q tests/` passa inteiro.
- Commit, push e `git pull` no cluster (o lote registra o commit).
- `sbatch -p debug --qos=complex --time=00:30:00 cluster/experimento.sh <yaml> --teste` antes do lote real.
- Nada de imagens, IDs ou dados de pacientes fora do computador local/cluster (`CLAUDE.md`, seção 1).
