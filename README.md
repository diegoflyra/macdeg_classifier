# macdeg_classifier

Pré-processamento, augmentation e auditoria de perda de informação para classificar DMRI precoce em retinografias da base [HYAMD](https://physionet.org/).

O objetivo desta etapa é preparar e comparar variantes sem apagar, borrar nem criar drusas pequenas. Ainda não há treino de modelo.

## Dados restritos

A HYAMD é distribuída pela PhysioNet sob licença restrita, e **nada dela pertence a este repositório**.

- `hyamd/`, `cache/` e `outputs/` estão no `.gitignore`.
- Os nomes de arquivo contêm o identificador do paciente. Trate caches, CSVs e grades como dados restritos.
- Notebooks executados contêm imagens e IDs. Em cada clone novo, rode `sh scripts/setup_git_filters.sh` antes de qualquer commit, para remover as saídas. O filtro funciona com o Git do WSL e com o do Windows.

## Instalação

Testado com Python 3.14.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sh scripts/setup_git_filters.sh
```

Coloque a base em `hyamd/`, com as pastas `Images/` e `labels/`.

## Como rodar

```bash
bash scripts/run_overnight.sh
```

O script gera os caches, as grades de verificação e a auditoria, e roda os testes. Se for interrompido, rode de novo: o pré-processamento retoma de onde parou. Espere cerca de 12 GB em `cache/`; o espaço em disco é checado antes de começar. O log fica em `outputs/logs/`.

Depois, abra `notebooks/01_resultados.ipynb` para ver os resultados.

## Conjunto multibase e treino

Problema: **DMRI × sem DMRI**, em que "sem DMRI" = olhos normais (2/3) + outras doenças (1/3, com cotas
iguais por doença). Bases: HYAMD, BRSET, ODIR, RFMiD, RFMiD 2.0 e ADAM, descritas em `configs/sources.yaml`
e guardadas fora do projeto (`~/macdeg_data/`).

```bash
.venv/bin/python -m macdeg.data.build      # junta as bases, sorteia as outras doenças, remove cópias (1024 px)
.venv/bin/python -m macdeg.data.splits     # validação cruzada aninhada: 5 dobras externas x 4 internas, por paciente
.venv/bin/python -m macdeg.data.package    # zip para o cluster
```

Treino no cluster Apuana: ver [`docs/cluster.md`](docs/cluster.md). Cada experimento é um YAML em
`configs/experimentos/` (configuração base + grade de variações) e roda como job array; a análise
envia `resultados/<lote>/` para a branch `resultados`. No cluster: `sbatch cluster/experimento.sh`.

AlexNet para DMRI: [`docs/alexnet.md`](docs/alexnet.md) explica o fine-tuning com pesos ImageNet,
as três primeiras convoluções congeladas e as camadas profundas treináveis. O experimento é
`002_alexnet_finetuning.yaml`, com saída binária e a avaliação já definida no projeto.
A arquitetura está escrita camada por camada em `macdeg/models.py`; só os pesos pré-treinados
são carregados do checkpoint oficial, separadamente.
As camadas de `macdeg/layers.py` usam `nn.Parameter` e operações funcionais; o `forward`
da AlexNet está explícito por etapas, sem `nn.Sequential`.
As propostas de taxas separadas e inicialização da saída estão nos experimentos
`003_alexnet_taxas.yaml` e `004_alexnet_inicializacao.yaml`, com comparações controladas.

ConvNeXt para DMRI: [`docs/convnext.md`](docs/convnext.md) descreve a implementação manual
com pesos ImageNet, blocos depthwise, LayerNorm, GELU e fine-tuning dos estágios profundos.
O experimento inicial é `005_convnext_tiny_finetuning.yaml`; `006_convnext_congelamento.yaml`
compara o congelamento de dois estágios com todas as camadas treináveis.
Para o cluster, `007_convnext_arquiteturas.yaml` compara Tiny, Small e Base (60 treinos),
e `008_convnext_ajustes.yaml` compara seis receitas da Tiny (120 treinos), com pilotos
antes dos lotes completos. `009_convnext_large.yaml` compara Base e Large em um lote opcional
de 40 treinos. A receita `convnext_cluster.yaml` usa entradas de 512 px, microbatch fixo,
checkpointing de ativações e ajustes opcionais do otimizador.
`010_convnext_augmentation.yaml` mede o efeito da receita de augmentation na Tiny (40 treinos).
`011_convnext_arquiteturas_augmentation.yaml` compara Tiny, Small, Base e Large com a mesma
receita de augmentation durante o treino (80 treinos).
`012_convnext_aux_dr.yaml` compara a Tiny com e sem a cabeça auxiliar de retinopatia diabética.
`013_convnext_arquiteturas_aux_dr.yaml` compara as quatro arquiteturas com augmentation e cabeça auxiliar.

Rede híbrida para DMRI: [`docs/hibrida_dmri.md`](docs/hibrida_dmri.md) descreve a vista global,
o bloco residual multiescala e os recortes guiados por atenção. O experimento
`014_hibrida_dmri.yaml` compara dois pesos para a perda de localização.

Avaliação:
- métrica principal: **média do F2 dentro de cada base**, com o limiar sempre escolhido em outras dobras;
- também: AUROC, AUPRC, sensibilidade, especificidade em normais, em outras doenças e por doença;
- réguas: só a base de origem e só a idade (o modelo precisa superá-las);
- veredito pareado contra a referência (*ganhou / empatou / perdeu*, IC95 por paciente);
- distribuição das notas de normais, outras doenças, drusas sem diagnóstico e DMRI.

Proposta atualizada: [`docs/proposta.md`](docs/proposta.md). Decisões e motivos: [`docs/decisoes.md`](docs/decisoes.md).

Modelos finais (ensemble das 5 dobras) vão, privados, para o Hugging Face: ver [`docs/huggingface.md`](docs/huggingface.md).
Nada de imagens ou dados de paciente sai do computador ou do cluster: W&B só recebe números, o GitHub
recebe agregados e previsões com IDs sequenciais nossos.

## Marcar o disco óptico

```bash
.venv/bin/python -m macdeg.annotate
```

Abra `http://localhost:8765` e clique no centro do disco em cada imagem. O anel que segue o mouse tem o tamanho típico do disco. `N` indica que não dá para ver o disco, e `Z` volta uma imagem. Tudo é salvo a cada clique em `annotations/disc_clicks.csv` (fora do Git, pois contém IDs); ao abrir de novo, o programa continua de onde parou. Não apague esse arquivo: ele não pode ser regenerado.

A página não mostra classe, nome de arquivo nem o palpite do detector, e cerca de 3% das imagens reaparecem sem aviso para medir a repetibilidade. Para comparar os cliques com o detector, sem identificar imagens:

```bash
.venv/bin/python -m macdeg.annotation_report
```

## Testes

Para rodar só os testes (com imagens sintéticas, não precisam da base):

```bash
.venv/bin/python -m pytest tests
```

## Estrutura

```
rodar.py                   Roda um experimento inteiro (tarefas, análise, envio); ver docs/cluster.md
cluster/experimento.sh     O job do SLURM: sbatch cluster/experimento.sh [experimento]
configs/sources.yaml       Registro das bases e regra do sorteio das outras doenças
configs/train/             Configuração de treino da referência
configs/experimentos/      Experimentos (configuração base + grade de variações)
configs/variants.yaml      Variantes do pré-processamento por ROI (suspenso)
macdeg/
  data/                    Leitores das bases, sorteio, cópias, divisões aninhadas, empacotamento
  train.py                 Treino de uma dobra (ResNet e afins)
  metrics.py               F2 por base, limiar em outras dobras, IC por paciente, veredito pareado
  cluster.py               Lança experimentos no SLURM (Apuana) ou localmente
  analysis.py              Junta as dobras, réguas, ranking, resumo; envia ao GitHub e ao W&B
  hub.py                   Publica e carrega modelos no Hugging Face (privado)
  ambiente.py              Segredos (.env) e envio à branch resultados
  metadata.py, localization.py, preprocess.py, augment.py, audit.py, grids.py
                           Pré-processamento por ROI da HYAMD (suspenso)
  annotate.py, annotation_report.py
                           Marcação manual do disco óptico
  exploration/             Inspeção inicial da HYAMD (Passo 0)
notebooks/
  00_inspecao.ipynb        Inspeção da HYAMD
  01_resultados.ipynb      Pré-processamento por ROI: sinalizações, grades e auditoria
cluster/setup_apuana.sh    Ambiente no Apuana (venv + PyTorch), rodar no nó install
cluster/requirements-cluster.txt  Pacotes do cluster (sem versão fixa; registradas no freeze.txt)
scripts/run_overnight.sh   Pipeline do pré-processamento por ROI
scripts/nb_clean.*         Filtro do Git que remove as saídas dos notebooks
tests/                     Testes automáticos
docs/                      Proposta, decisões, guias do cluster e do Hugging Face, relatório do Passo 0
```

## Usar uma variante

Um ID tem o formato `<ROI>_<resolução>_<receita>`, por exemplo `R2.0_S512_A2-G1`.

```python
import numpy as np
from macdeg.augment import Augmenter, get_variant, load_patch

variant = get_variant("R2.0_S512_A2-G1")
patch = load_patch("cache", variant, file_name)

train = Augmenter(variant, train=True)
image = train(patch, np.random.default_rng(0))

# Várias imagens do mesmo olho recebem os mesmos parâmetros.
images = train([patch_a, patch_b], np.random.default_rng(0))

# Validação e teste só normalizam.
evaluation = Augmenter(variant, train=False)
```

As escolhas de método e os desvios da especificação estão em [`docs/decisoes.md`](docs/decisoes.md).
