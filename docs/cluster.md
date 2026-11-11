# Treinar no Apuana (CIn/UFPE)

O que sabemos do cluster (11/11/2026): SLURM; partições `debug` (30 min), `short-simple`/`short-complex`
(2 dias), `long-*` (7 dias), `install` (30 min); sem preempção; nós 1–5 com RTX 3090 (24 GB), nós 6–10
com A100 80 GB dividida em fatias MIG (~10 GB cada); driver 590 (CUDA 13.1); login com internet e sem
conda; `sacctmgr` indisponível para usuários (limites de QoS desconhecidos). Vários nós estavam fora do ar.

## Uma vez só

No seu computador (conectado à VPN do CIn):

```bash
# 1. Empacotar o conjunto (~4,5 GB) e anotar o sha256 que ele imprime
.venv/bin/python -m macdeg.data.package
# 2. Enviar ao seu $HOME no cluster
ssh <login>@slurm-client1.cin.ufpe.br "mkdir -p ~/macdeg_data && chmod 700 ~/macdeg_data"
rsync -P ~/macdeg_data/dataset_1024.zip <login>@slurm-client1.cin.ufpe.br:~/macdeg_data/
# 3. Enviar o .env (tokens) direto para o clone no cluster, sem passar pelo Git
scp .env <login>@slurm-client1.cin.ufpe.br:~/macdeg_classifier/.env
```

No nó de login do cluster (`ssh <login>@slurm-client1.cin.ufpe.br`):

```bash
git clone https://github.com/diegoflyra/macdeg_classifier.git ~/macdeg_classifier   # pede o token (repo privado)
cd ~/macdeg_classifier && chmod 600 .env
sha256sum ~/macdeg_data/dataset_1024.zip                         # tem que bater com o do passo 1
python -m zipfile -e ~/macdeg_data/dataset_1024.zip ~/macdeg_data/dataset_1024   # para o lançador conferir a assinatura
bash cluster/setup_apuana.sh                                      # Miniforge + ambiente "macdeg" (~15 min)
echo 'export MACDEG_RUNS=$SCRATCH/macdeg_runs' >> ~/.bashrc      # lotes no $SCRATCH (rápido, sem cota)
```

Os lotes ficam no `$SCRATCH` (arquivos sem uso há 90 dias são apagados lá); o que importa vai para o
GitHub e o Hugging Face. Pesos só são guardados no treino final.

## Cada experimento

1. No seu computador: crie ou edite `configs/experimentos/<n>_<nome>.yaml`, **commit e push**.
2. No cluster: `cd ~/macdeg_classifier && git pull && conda activate macdeg`.
3. Teste rápido (minutos, números sem valor):
   `python -m macdeg.cluster lancar configs/experimentos/001_referencia.yaml --teste`
4. Valendo: `python -m macdeg.cluster lancar configs/experimentos/001_referencia.yaml`
   (ou abra `notebooks/02_lancamento.ipynb`, que faz o mesmo com as verificações à vista).
5. Acompanhar: `squeue -u $USER`; logs em `~/macdeg_runs/<lote>/logs/`; curvas no W&B.
6. Quando o array termina, o job de análise roda sozinho e envia `resultados/<lote>/` para a branch
   `resultados` do GitHub, incluindo `notebooks/` com as cópias executadas (tarefas e análise, só texto:
   saídas de imagem são removidas antes do envio).

## O que cada peça faz e por quê

- **Job array**: cada tarefa é um treino independente (configuração × dobra externa × dobra interna).
  `max_paralelo` (padrão 5) limita quantas rodam ao mesmo tempo, para não estourar a cota do grupo.
- **Partição `short-simple` (QoS `simple`)**: alcança todos os nós e GPUs, até 2 dias, prioridade mais
  alta; o Apuana não tem preempção. Se um nó cair, `--requeue` recoloca a tarefa e o treino continua do `last.pt`.
- **Ambiente**: o login não tem conda; `cluster/setup_apuana.sh` instala o Miniforge no `$HOME`, que é
  compartilhado com os nós de GPU.
- **Código fixo**: o lançador extrai o commit atual para `~/macdeg_runs/<lote>/codigo/`. Um push seu no
  meio do lote não muda o código das tarefas que ainda vão começar.
- **Dados no disco do nó**: cada tarefa descompacta o zip em `$SLURM_TMPDIR` (NVMe local), muito mais
  rápido que ler 18 mil imagens pelo `$HOME` (rede).
- **Assinatura dos dados**: o lote guarda um hash do manifest e das divisões; uma tarefa recusa rodar
  se o conjunto mudou, porque os resultados não seriam comparáveis.
- **Lote efetivo fixo (16)**: em GPU com pouca memória, o treino divide o lote e acumula gradientes;
  o resultado não depende da GPU que a fila sortear.
- **W&B**: tenta online; sem internet no nó, grava offline e o job de análise sincroniza.

## Publicar o modelo final de uma configuração

`python -m macdeg.cluster final ~/macdeg_runs/<lote analisado> <configuração>` treina as 5 dobras
externas dessa configuração e a análise envia o ensemble, privado, ao Hugging Face (ver `docs/huggingface.md`).
