# Novo membro: ambiente completo (computador + Apuana)

Guia para quem entra no projeto deixar o ambiente igual ao dos donos. Só `git`, `venv`, `pip`, `ssh`,
`rsync` e `sbatch`. Leia antes `CLAUDE.md` e `docs/fluxo_experimentos.md`; agentes (Claude, Codex)
seguem os dois.

Troque `<login>` pelo seu usuário do CIn e `<sua-branch>` pela sua branch pessoal (ex.: `alexnet-finetuning`).

## 0. Pré-requisitos (antes de qualquer comando)

- **Acesso próprio aos dados.** HYAMD e BRSET vêm do PhysioNet (licença de dados credenciados: proíbe
  compartilhar acesso com quem não tem credenciamento próprio). Você precisa de: conta **credenciada** no
  PhysioNet (com o curso CITI) e os termos de uso (DUA) **assinados** para HYAMD e BRSET; cadastro no
  desafio **ADAM** (uso só em pesquisa). Sem isso, pare na seção 4: o resto do ambiente pode ser montado.
- **GitHub:** ser colaborador do repositório (os donos convidam) e um token pessoal (classic, escopo `repo`).
- **W&B:** convite para o time da UFPE no W&B e a sua chave de API.
- **Cluster:** conta no Apuana e a VPN do CIn.

## 1. No seu computador (editar código e rodar os testes)

Os dados não são necessários aqui: os testes usam imagens sintéticas.

```bash
git clone https://github.com/diegoflyra/macdeg_classifier.git
cd macdeg_classifier
git checkout <sua-branch>
git merge --ff-only origin/experimentos-003     # sua branch parte da base atual do projeto

python3 -m venv .venv                           # Python 3.10 ou mais novo
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1   # CPU; com GPU NVIDIA: --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests/            # precisa passar inteiro (alguns minutos)
```

## 2. No cluster: repositório (nó de login, só tarefas leves)

```bash
ssh <login>@slurm-client1.cin.ufpe.br           # com a VPN do CIn ligada
git clone https://github.com/diegoflyra/macdeg_classifier.git ~/macdeg_classifier   # senha = token do GitHub
cd ~/macdeg_classifier
git checkout <sua-branch>
git merge --ff-only origin/experimentos-003
mkdir -p logs ~/macdeg_data ~/macdeg_runs
chmod 700 ~/macdeg_data ~/macdeg_runs           # dados e resultados só para você
echo 'export MACDEG_RUNS=$HOME/macdeg_runs' >> ~/.bashrc && source ~/.bashrc
git config --global credential.helper store     # guarda o token para os próximos pull/push
```

## 3. No cluster: ambiente Python (nó de instalação, nunca no login)

```bash
salloc -p install --qos=simple -c 8 --mem=32G --time=00:30:00
cd ~/macdeg_classifier
bash cluster/setup_apuana.sh                    # cria .venv no repositório (Python do sistema) + PyTorch cu130
exit
```

Progresso em outra sessão: `tail -f ~/macdeg_classifier/.venv/install.log`. Se passar de 30 min, peça outro
`salloc` e rode de novo (o que já instalou é reaproveitado). As versões ficam em `.venv/freeze.txt`.

## 4. Segredos (`.env`), os SEUS

Nunca copie o `.env` de outra pessoa. No cluster (e, se quiser, no computador):

```bash
cd ~/macdeg_classifier
cat > .env <<'EOF'
GITHUB_TOKEN=cole_aqui_o_seu_token
WANDB_API_KEY=cole_aqui_a_sua_chave
HF_TOKEN=
EOF
chmod 600 .env
source .venv/bin/activate
python -c "from macdeg.ambiente import load_dotenv, which_secrets; load_dotenv(); print(which_secrets())"
```

`GITHUB_TOKEN` e `WANDB_API_KEY` têm que aparecer `True`. O `HF_TOKEN` fica vazio: o Hugging Face é dos donos.

## 5. Dados no cluster (só depois do item 0)

O conjunto é um zip único, `~/macdeg_data/dataset_1024.zip` (3,2 GB). Ele tem que ser **idêntico** ao dos
donos, senão os resultados não são comparáveis:

- sha256 do zip: `de151d9ec47e471dfc19664da0578ee69fa8e038a124f4121c7b636195c0709f`
- assinatura dos dados (rótulos + divisões): `72e45dfc1003894a`

**Caminho A, o mais rigoroso (você baixa das fontes e monta igual).** Com o seu acesso, baixe as bases nos
caminhos de `configs/sources.yaml` (`hyamd/`, `brazilian-ophthalmological/` na raiz do repositório; ODIR,
RFMiD, RFMiD 2.0 e ADAM em `~/macdeg_data/`) e, no seu computador:

```bash
.venv/bin/python -m macdeg.data.build      # 1024 px, cópias removidas, sorteio com semente 0
.venv/bin/python -m macdeg.data.splits     # validação cruzada aninhada, semente 0
.venv/bin/python -m macdeg.data.package    # gera ~/macdeg_data/dataset_1024.zip e imprime o sha256
rsync -avP ~/macdeg_data/dataset_1024.zip <login>@slurm-client1.cin.ufpe.br:~/macdeg_data/
```

**Caminho B, cópia dos donos (só com o item 0 confirmado por eles).** O dono libera a leitura só para o
seu usuário, você copia, e ele revoga:

```bash
# o DONO, no cluster:
setfacl -m u:<login>:x ~ ~/macdeg_data && setfacl -m u:<login>:r ~/macdeg_data/dataset_1024.zip
# VOCÊ, no cluster:
cp /home/CIN/<login_do_dono>/macdeg_data/dataset_1024.zip ~/macdeg_data/
# o DONO, logo depois:
setfacl -x u:<login> ~ ~/macdeg_data ~/macdeg_data/dataset_1024.zip
```

Se o `setfacl` não existir ou falhar no `$HOME`, não abra as permissões da pasta para todos: peça aos
administradores do Apuana uma pasta de grupo do projeto.

**Depois de qualquer caminho, no cluster:**

```bash
sha256sum ~/macdeg_data/dataset_1024.zip        # tem que bater com o sha256 acima
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2], ['manifest.csv','splits_nested_seed0.csv','extras_drusas.csv','summary.json'])" ~/macdeg_data/dataset_1024.zip ~/macdeg_data/dataset_1024
chmod -R go-rwx ~/macdeg_data
cd ~/macdeg_classifier && source .venv/bin/activate
python -c "from pathlib import Path; from macdeg.cluster import dataset_signature; print(dataset_signature(Path.home()/'macdeg_data/dataset_1024','splits_nested_seed0.csv'))"   # 72e45dfc1003894a
```

Só os metadados ficam descompactados no `$HOME`; as imagens saem do zip para o disco local do nó em cada job.

## 6. Primeiro teste no cluster

```bash
cd ~/macdeg_classifier && git pull
sbatch --gres=gpu:1 --time=00:30:00 cluster/experimento.sh configs/experimentos/004_gustavo_arquiteturas.yaml --teste
squeue -u $USER
tail -5 $(ls -t logs/*.out | head -1)           # quando terminar: "fim | status 0"
```

O `--teste` roda em minutos, com números sem valor, e não envia nada ao GitHub.

## 7. O dia a dia (igual ao dos donos)

1. **Computador:** atualizar a base (`git fetch && git merge origin/experimentos-003`), editar, criar o YAML
   seguindo `docs/fluxo_experimentos.md`, rodar `.venv/bin/python -m pytest -q tests/`, commit e push na
   `<sua-branch>`.
2. **Cluster:** `cd ~/macdeg_classifier && git pull`, `--teste` primeiro, depois o lote real:
   `sbatch --gres=gpu:1 cluster/experimento.sh configs/experimentos/<NNN>_<nome>.yaml`. Com a fila livre,
   um segundo job no mesmo lote: `sbatch --gres=gpu:1 cluster/experimento.sh --retomar $(cat ~/macdeg_runs/ultimo_lote.txt) --do-fim`.
3. **Acompanhar:** `squeue -u $USER`; `grep '"epoch"' $(ls -t logs/*.out | head -1) | tail -2`; curvas no W&B.
4. **Resultado:** no fim, `resultados/<lote>/` chega à branch `resultados` do GitHub (RESUMO.md, relatório,
   previsões por `sample_id`). Job que estourou o tempo: `--retomar <lote>`.
5. **Voltar o trabalho para o projeto:** PR da `<sua-branch>` para a `experimentos-003`, revisado pelos donos.

Nunca: imagens, IDs ou planilhas das bases fora do computador/cluster; commit na `main` ou na `resultados`;
editar um YAML já lançado; abrir permissões de `~/macdeg_data`.
