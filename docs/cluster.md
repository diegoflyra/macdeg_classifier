# Treinar no Apuana (CIn/UFPE)

Fonte principal: o guia prático do Apuana no Notion (referência oficial: apuana.cin.ufpe.br/usage.html),
conferido com `sinfo` e `scontrol` em 11/11/2026.

## O cluster, em resumo

- **Acesso:** VPN do CIn + `ssh <login>@slurm-client1.cin.ufpe.br`. O nó de login é fraco (8 GB de RAM):
  nada pesado nele. Instalações vão para a partição `install`; trabalho interativo, para `install` ou `debug`.
- **Nós:** 1–5 com 2 × RTX 3090 (24 GB) cada (o nó 2 tem 3); 6–10 com 1 × A100 80 GB, dividida em até 7
  fatias MIG (~10 GB cada). Disco local NVMe de 1 TB em todos. Vários nós podem estar fora do ar.
- **Partições:**

  | Partição | QoS | Tempo máx. | Uso |
  |---|---|---|---|
  | `install` | simple | 30 min | instalar pacotes (interativa) |
  | `debug` | complex | 30 min | testes rápidos (interativa) |
  | `short-simple` | simple | 2 dias | **padrão dos nossos treinos** |
  | `short-complex` | complex | 2 dias | jobs maiores |
  | `long-simple` / `long-complex` | simple / complex | 7 dias | jobs longos |

- **Limites por QoS (por usuário):** `simple` = 2 GPUs, 16 CPUs, 64 GB; `complex` = 4 GPUs, 48 CPUs, 500 GB.
  Todo job precisa de `--qos` igual ao da partição. Sem preempção; o SLURM mata o job no tempo máximo.
- **Ambiente:** `venv` com o Python do sistema (Ubuntu 22.04: 3.10.12, igual em todos os nós). Os módulos
  (`module avail Python`) vão só até 3.10.8; o script usa um módulo apenas se ele for mais novo. Sem conda. `sacct`/`sacctmgr` indisponíveis para
  usuários (o banco de contabilidade está desligado). `$SCRATCH` não existe.

## Uma vez só

**No seu computador** (WSL, conectado à VPN):

```bash
cd /mnt/c/Users/lyrad/Documents/code/macdeg_classifier
.venv/bin/python -m macdeg.data.package        # gera ~/macdeg_data/dataset_1024.zip e imprime o sha256
ssh <login>@slurm-client1.cin.ufpe.br "mkdir -p ~/macdeg_data && chmod 700 ~/macdeg_data"
rsync -avP ~/macdeg_data/dataset_1024.zip <login>@slurm-client1.cin.ufpe.br:~/macdeg_data/
scp .env <login>@slurm-client1.cin.ufpe.br:~/macdeg_classifier/.env      # depois do clone abaixo
```

**No cluster, nó de login** (só tarefas leves):

```bash
git clone https://github.com/diegoflyra/macdeg_classifier.git ~/macdeg_classifier   # senha = token do GitHub
chmod 600 ~/macdeg_classifier/.env
sha256sum ~/macdeg_data/dataset_1024.zip                         # tem que bater com o do pacote
# Só os metadados (o lançador confere a assinatura com eles); as imagens ficam no zip
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2], ['manifest.csv','splits_nested_seed0.csv','extras_drusas.csv','summary.json'])" ~/macdeg_data/dataset_1024.zip ~/macdeg_data/dataset_1024
echo 'export MACDEG_RUNS=$HOME/macdeg_runs' >> ~/.bashrc
```

**No cluster, nó de instalação** (o ambiente):

```bash
salloc -p install --qos=simple -c 8 --mem=32G --time=00:30:00
cd ~/macdeg_classifier
bash cluster/setup_apuana.sh              # venv em ~/macdeg_classifier/.venv (Python do sistema) + PyTorch cu130
exit                                      # sai do nó de instalação
```

O script usa o Python do sistema, a menos que `module avail Python` tenha um mais novo (ou que você
passe um módulo como segundo argumento); nesse caso grava o nome em `.venv/MODULO_PYTHON` e os
jobs carregam o mesmo módulo. Instala o PyTorch e os pacotes de `cluster/requirements-cluster.txt` e
registra as versões em `.venv/freeze.txt`. Progresso em outra sessão: `tail -f ~/macdeg_classifier/.venv/install.log`.
O venv fica dentro do repositório, como no computador local (o `.gitignore` o exclui). Os jobs o procuram
em `$MACDEG_VENV`, `.venv` ou `venv` do repositório e, por último, em `~/envs/macdeg`.
Se passar de 30 minutos, peça outro `salloc` e rode de novo: o que já foi instalado é reaproveitado.

## Cada experimento

Um experimento = **um job com 2 GPUs** (o máximo do QoS `simple`). O `cluster/experimento.sh` confere o
ambiente, copia o conjunto para o disco do nó, roda as tarefas (`rodar.py`: um processo por GPU, cada um
pegando a próxima tarefa livre), a análise e o envio ao GitHub e ao W&B. No log, cada linha das tarefas
começa com `[gpu0]` ou `[gpu1]`. Com `--gres=gpu:1` antes do script, roda uma tarefa por vez.

1. No seu computador: crie ou edite `configs/experimentos/<n>_<nome>.yaml`, **commit e push**.
2. No cluster (nó de login, na raiz do repositório):
   ```bash
   cd ~/macdeg_classifier && git pull && mkdir -p logs
   sbatch -p debug --qos=complex --time=00:30:00 cluster/experimento.sh --teste    # teste rápido
   sbatch cluster/experimento.sh                                                    # a referência, valendo
   sbatch cluster/experimento.sh configs/experimentos/002_x.yaml                   # outro experimento
   ```
3. Acompanhar: `squeue -u $USER`; saída em `logs/macdeg_<job>.out` (`tail -f` mostra ao vivo) e erros em
   `logs/macdeg_<job>.err`; curvas no W&B.
4. No fim, `resultados/<lote>/` chega à branch `resultados` do GitHub, com os logs do job.

- **Se o job estourar o tempo** (2 dias na `short-simple`) ou cair: `sbatch cluster/experimento.sh --retomar
  ~/macdeg_runs/<lote>`. Tarefas prontas são puladas e a interrompida continua do `last.pt`.
- **Outra fila:** as opções do `sbatch` vêm antes do script, ex.: `sbatch -p long-simple cluster/experimento.sh`.
- **Sem cluster:** `python rodar.py --teste` roda o mesmo no computador.

## O que cada peça faz e por quê

- **Um job por experimento, 2 GPUs:** pede as GPUs uma vez e divide as tarefas entre elas (fila comum:
  quem termina pega a próxima). Menos espera na fila e nada de jobs encadeados. Cada tarefa = configuração
  × dobra externa × dobra interna. O resultado de uma tarefa não depende da GPU em que rodou.
- **Cache das imagens reduzidas (`macdeg/cache.py`):** na referência, a A100 passava ~2/3 do tempo esperando
  a CPU decodificar JPEGs de 1024 px e reduzi-los (117 s por época, contra ~35 s de GPU). Agora a primeira
  tarefa do job reduz todas as imagens uma vez (~1 min, 16 processos) para um `.npy` no disco do nó e
  os treinos leem dele por mmap. Os pixels são idênticos (testado), então os resultados continuam
  comparáveis com a referência. `cache: false` na configuração desliga. Com augmentation, os recortes
  partem da imagem já em `image_size` (prática comum; com `crop_scale` ≥ 0,85 a perda de detalhe é mínima).
- **Diagnóstico de velocidade:** cada época registra `train_seconds` e `data_wait_seconds` (tempo parado
  esperando dados; no W&B, `segundos_esperando_dados`). Se a espera for grande, o gargalo é CPU/disco.
- **Início do job:** imprime nó, GPU e PyTorch (para diagnosticar pelo log) e falha cedo, com mensagem no
  `.err`, se faltar venv ou CUDA.
- **Dados no disco do nó:** o zip é descompactado no `/tmp` do nó (NVMe) e apagado no fim, mesmo se o job
  falhar ou for cancelado.
- **Assinatura dos dados e commit** ficam no `lote.json`; o código do commit é copiado para `codigo/`.
- **Lote efetivo fixo (16):** em GPU com pouca memória (fatia MIG), o treino acumula gradientes; o
  resultado não depende da GPU.
- **Espaço:** lotes em `~/macdeg_runs` (cota de 100 GB). Treinos internos não guardam pesos; o treino final
  guarda ~0,5 GB por configuração.

## Publicar o modelo final de uma configuração

`sbatch cluster/experimento.sh --final ~/macdeg_runs/<lote analisado> <configuração>` treina as 5 dobras
externas dessa configuração e guarda o ensemble em `<lote final>/modelo/<configuração>/` (o cartão vai ao
GitHub). Publicar no Hugging Face é um passo separado: `python -m macdeg.hub publicar <lote final> <configuração>`
(ver `docs/huggingface.md`).
