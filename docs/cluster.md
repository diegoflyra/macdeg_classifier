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
bash cluster/setup_apuana.sh              # venv em ~/envs/macdeg (Python do sistema) + PyTorch cu130
exit                                      # sai do nó de instalação
```

O script usa o Python do sistema, a menos que `module avail Python` tenha um mais novo (ou que você
passe um módulo como segundo argumento); nesse caso grava o nome em `~/envs/macdeg/MODULO_PYTHON` e os
jobs carregam o mesmo módulo. Instala o PyTorch e os pacotes de `cluster/requirements-cluster.txt` e
registra as versões em `~/envs/macdeg/freeze.txt`. Progresso em outra sessão: `tail -f ~/envs/macdeg/install.log`.
Se passar de 30 minutos, peça outro `salloc` e rode de novo: o que já foi instalado é reaproveitado.

## Cada experimento

1. No seu computador: crie ou edite `configs/experimentos/<n>_<nome>.yaml`, **commit e push**.
2. No cluster (login): `cd ~/macdeg_classifier && git pull`, depois ative o ambiente:
   ```bash
   source ~/envs/macdeg/bin/activate
   ```
3. Teste rápido (vai para a `debug`; minutos; números sem valor):
   `python -m macdeg.cluster lancar configs/experimentos/001_referencia.yaml --teste`
4. Valendo: `python -m macdeg.cluster lancar configs/experimentos/001_referencia.yaml`
   - outra partição: `--particao short-complex` (QoS `complex`, até 4 tarefas simultâneas) ou `--particao long-simple`;
   - outros limites: `--tempo 06:00:00`, `--max-paralelo 2`.
5. Acompanhar: `watch -n 30 squeue -u $USER`; previsão de início: `squeue -u $USER --start`;
   logs em `~/macdeg_runs/<lote>/logs/` (`tail -f` mostra ao vivo); curvas no W&B.
6. Quando o array termina, o job de análise roda sozinho e envia `resultados/<lote>/` para a branch
   `resultados` do GitHub, incluindo `notebooks/` com as cópias executadas (só texto).

Jobs na fila podem ser mudados sem cancelar: `scontrol update JobId=<id> Partition=short-simple,long-simple
TimeLimit=06:00:00` (o QoS precisa valer para todas as partições da lista).

## O que cada peça faz e por quê

- **Job array** com até 2 tarefas simultâneas na `short-simple` (limite de GPUs do QoS `simple`; mais que
  isso só ficaria preso na fila). Cada tarefa = configuração × dobra externa × dobra interna.
- **Início de cada job:** imprime nó, GPU e versão do PyTorch (para diagnosticar falhas pelo log), carrega
  o `venv` (e o módulo Python, se houver), descompacta o zip numa pasta própria no `/tmp` do nó (NVMe) e **apaga a pasta no
  fim**, mesmo se o job falhar ou for cancelado.
- **Código fixo:** o lançador extrai o commit atual para `~/macdeg_runs/<lote>/codigo/`; um push no meio do
  lote não muda o código das tarefas que ainda vão começar.
- **Assinatura dos dados:** cada tarefa recusa rodar se o conjunto ou as divisões mudaram.
- **Lote efetivo fixo (16):** em GPU com pouca memória (fatia MIG), o treino acumula gradientes; o
  resultado não depende da GPU sorteada.
- **Retomada:** `--requeue` + `last.pt`: se um nó cair, a tarefa volta à fila e continua da última época.
- **W&B:** tenta online; sem internet no nó, grava offline e o job de análise sincroniza.
- **Espaço:** lotes em `~/macdeg_runs` (cota de 100 GB no `$HOME`). Treinos internos não guardam pesos;
  o treino final guarda ~0,5 GB por configuração. Lotes antigos podem ser apagados (o que importa está no
  GitHub e no Hugging Face).

## Publicar o modelo final de uma configuração

`python -m macdeg.cluster final ~/macdeg_runs/<lote analisado> <configuração>` treina as 5 dobras
externas dessa configuração e a análise envia o ensemble, privado, ao Hugging Face (ver `docs/huggingface.md`).
