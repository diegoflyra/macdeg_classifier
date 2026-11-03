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

Para rodar só os testes (com imagens sintéticas, não precisam da base):

```bash
.venv/bin/python -m pytest tests
```

## Estrutura

```
configs/variants.yaml      Registro de variantes (ROI, resolução, geometria, fotometria, qualidade)
macdeg/
  metadata.py              Liga labels.csv aos arquivos, por olho
  localization.py          Campo de visão, disco e fóvea
  preprocess.py            Gera os caches e o CSV de metadados
  augment.py               Carrega qualquer variante por ID e aplica as transformações
  audit.py                 Mede quanta estrutura pequena sobrevive a cada variante
  grids.py                 Grades de verificação visual
  exploration/             Inspeção inicial da base (Passo 0); o pipeline não depende dela
notebooks/
  00_inspecao.ipynb        Inspeção da base
  01_resultados.ipynb      Sinalizações, grades e auditoria
scripts/run_overnight.sh   Executa o pipeline inteiro
scripts/nb_clean.*         Filtro do Git que remove as saídas dos notebooks
tests/                     Testes automáticos
docs/                      Relatório do Passo 0 e registro de decisões
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
