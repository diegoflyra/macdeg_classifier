# Fine-tuning da AlexNet para DMRI

> **Estrutura dos experimentos (17/11/2026):** siga `docs/fluxo_experimentos.md`. A AlexNet entra em duas
> partes. **Parte 1** (`004_gustavo_arquiteturas.yaml`): AlexNet com a receita comum, sobre a base vigente
> da rodada 3 (augmentation + cabeça auxiliar de DR), comparada com a `rodada3-00`. **Parte 2** (apêndice):
> `005_alexnet_congelamento.yaml` (referência `gustavo-01`), `006_alexnet_taxas.yaml` e
> `007_alexnet_inicializacao.yaml`, uma mudança por vez. Os antigos `002`–`004_alexnet_*.yaml` e
> `configs/train/alexnet_finetuning*.yaml` foram renumerados e convertidos para `base_mudancas`. Onde o
> texto abaixo fala em "sem augmentation" ou nesses arquivos, vale a estrutura nova.

Vamos usar a **AlexNet pré-treinada na ImageNet**, mantendo as convoluções
superficiais congeladas e adaptando as camadas profundas às retinografias.
O alvo é DMRI (1) × sem DMRI (0), com os dados e a avaliação já definidos no
projeto. A implementação usada é a do torchvision, compatível com seus pesos
`AlexNet_Weights.IMAGENET1K_V1`.

Nesta primeira etapa, a entrada fica sem tratamentos de realce e sem augmentation
(`augment: null`). CLAHE, equalização, remoção de ruído e alterações aleatórias
de cor ou geometria ficam para experimentos posteriores. A preparação numérica
continua sendo conversão para RGB, tensor e normalização ImageNet, com entrada
de 512 px.

Isso não significa que o conjunto atual seja composto por arquivos originais
intactos. A construção do dataset já aplica `fov_square`: enquadra a região da
retina, completa o quadrado com bordas pretas, redimensiona e salva um JPEG
derivado. Esse preparo geométrico deve ser descrito no método. Uma avaliação com
o enquadramento original completo exigiria uma versão separada do conjunto;
esta etapa não reconstrói os dados nem altera suas divisões.

## O que fica congelado e o que aprende

| Camada | Módulo do torchvision | Inicialização | Durante o treino |
|---|---|---|---|
| Conv1, 64 filtros de 11 × 11 | `features[0]` | ImageNet | congelada |
| Conv2, 192 filtros de 5 × 5 | `features[3]` | ImageNet | congelada |
| Conv3, 384 filtros de 3 × 3 | `features[6]` | ImageNet | congelada |
| Conv4, 256 filtros de 3 × 3 | `features[8]` | ImageNet | treinável |
| Conv5, 256 filtros de 3 × 3 | `features[10]` | ImageNet | treinável |
| Linear, 4096 neurônios | `classifier[1]` | ImageNet | treinável |
| Linear, 4096 neurônios | `classifier[4]` | ImageNet | treinável |
| Linear, um logit para DMRI | `classifier[6]` | aleatória, nova saída | treinável |

A hipótese é reaproveitar as características mais gerais aprendidas nas
camadas iniciais e deixar as mais profundas se adaptar à tarefa. Isso não
significa que as primeiras camadas sejam ideais para toda retinografia nem
que as últimas vão aprender drusas automaticamente. O resultado precisa ser
medido nas dobras de avaliação.

As duas camadas densas ocultas também conservam os pesos ImageNet. Só a saída
de 1000 classes é substituída por uma camada com um logit. O treino usa
`BCEWithLogitsLoss`, e a inferência aplica sigmoide para produzir um escore entre
0 e 1. Esse escore não é, por si só, uma probabilidade clinicamente calibrada.

## Como o congelamento funciona

`freeze_first_convs: 3` conta as três primeiras convoluções, não os três primeiros
módulos do Sequential. ReLU e pooling também aparecem nessa sequência, mas não
têm pesos. Os parâmetros congelados usam `requires_grad=False` e ficam fora do
otimizador. O otimizador recebe só parâmetros com `requires_grad=True`.

O congelamento fica fixo durante todo o treino. Não há descongelamento por
época nesta versão. A AlexNet do torchvision não tem BatchNorm, e não há dropout
no extrator convolucional; chamar `model.train()` não muda os parâmetros das
convoluções congeladas. O dropout do classificador fica ativo durante o treino
e desligado na avaliação.

O padrão do construtor continua sendo `freeze_first_convs=0`, preservando as
configurações antigas. O experimento novo define explicitamente `3`. Valores de
0 a 5 permitem comparar graus de congelamento em novos experimentos. Essa opção
é específica da AlexNet; outros modelos recusam uma contagem maior que zero.

## Receita consolidada

```yaml
model: alexnet
pretrained: true
freeze_first_convs: 3
head_init: xavier
image_size: 512
lr: 1.0e-5
head_lr: 1.0e-4
```

A proposta corresponde à configuração `xavier` de `configs/experimentos/007_alexnet_inicializacao.yaml`. Só a nova
saída recebe Xavier uniforme com ganho 1 e viés zero. Os pesos pré-treinados das
demais camadas são mantidos. As convoluções profundas e as densas pré-treinadas
usam taxa de 1e-5; só a última camada usa 1e-4. São valores iniciais para avaliar,
sem desempenho medido. O nome do arquivo identifica a proposta, não uma melhora
já demonstrada.

A receita mantém AdamW, weight decay, normalização ImageNet, balanceamento pela
perda e nenhuma augmentation. O warmup e o decaimento cosseno já existentes
aplicam o mesmo fator aos dois grupos, preservando a razão de dez entre as taxas.
O histórico registra `lr` e `head_lr`. As configurações antigas continuam com a
inicialização padrão e uma taxa única quando essas opções não são informadas.

O treino inicial só da saída, seguido de descongelamento, ainda é uma proposta
separada. Não está implementado. O warmup atual aumenta as taxas gradualmente;
não muda quais camadas estão congeladas.

Mantemos a resolução de 512 px do projeto. O pooling adaptativo original do
torchvision leva os mapas convolucionais a 6 × 6 antes do classificador.
Preservar mais detalhes na entrada é uma hipótese; a primeira convolução tem
stride 4 e ainda pode perder estruturas pequenas. Comparar resoluções exige
uma avaliação própria.

O YAML `configs/experimentos/005_alexnet_congelamento.yaml` roda uma configuração,
com cinco dobras externas e quatro internas por dobra. A época é escolhida pela
AUROC interna, e os limiares são escolhidos em outras dobras. O F2 médio por base
continua sendo a métrica principal. O lote não inclui um comparador e não permite
dizer que o congelamento melhorou o resultado sem uma comparação posterior
pareada; por isso a referência dele é a `gustavo-01` (AlexNet sem congelamento). As métricas de cada tarefa também registram as contagens de parâmetros
treináveis e congelados.

Os experimentos novos isolam cada mudança dentro do próprio lote:

| Experimento | Controle | Variação | Mantido igual |
|---|---|---|---|
| `006_alexnet_taxas.yaml` | taxa única de 1e-5 | saída a 1e-4 | inicialização padrão, camadas pré-treinadas a 1e-5 |
| `007_alexnet_inicializacao.yaml` | inicialização padrão da saída | Xavier + viés zero | taxas separadas de 1e-5 e 1e-4 |

Cada lote tem duas configurações e 40 tarefas completas. A segunda configuração
do lote 007 corresponde à proposta consolidada.
Comparar seus resultados com os lotes novos muda mais de um fator; essa comparação
não isola a contribuição de cada ajuste.

## Próximas hipóteses para avaliar

- Augmentation leve, preservando a região macular e estruturas pequenas. Hoje,
  ativar `augment` também ativa `RandomResizedCrop`; é preciso separar o controle
  do recorte antes de avaliar aumentos que devam preservar o campo inteiro.
- Aquecimento só da saída antes do fine-tuning parcial, comparado com o treino
  direto. É uma estratégia adicional, diferente do warmup da taxa de aprendizado.
- Se houver overfitting, testar o congelamento das densas pré-treinadas ou um
  classificador menor. Mesmo congelando três convoluções, cerca de 56 milhões
  de parâmetros continuam treináveis; as densas concentram a maior parte deles.
- Usar mapas de atribuição como auditoria e avaliar uma base deixada de fora.
  Um mapa destacado não prova que a rede reconheceu uma lesão. Dados, imagens e
  visualizações de pacientes continuam só no ambiente autorizado do projeto.

## Fundamentação

- [D2L, AlexNet](https://pt.d2l.ai/chapter_convolutional-modern/alexnet.html)
  explica convoluções, ReLU, pooling e dropout. O exemplo do livro usa uma
  disposição de canais diferente da versão do torchvision; usamos a versão
  compatível com os pesos ImageNet, sem reconstruir a arquitetura do livro.
- [D2L, Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html)
  apresenta a transferência de pesos e a adaptação da saída para outra tarefa.
- [Krizhevsky et al. (2012), ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
  é a referência original da AlexNet.
- [Burlina et al. (2017), Automated Grading of Age-Related Macular Degeneration From Color Fundus Images Using Deep Convolutional Neural Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC5710387/)
  usaram AlexNet em retinografias AREDS para classificar DMRI encaminhável e
  avaliaram separação por paciente. O alvo difere do nosso DMRI × sem DMRI.
- [Peng et al. (2019), DeepSeeNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6435402/)
  avaliaram fine-tuning de Inception-v3 para sinais de DMRI. Isso ajuda a motivar
  transferência de aprendizado, mas não valida especificamente congelar as três
  primeiras convoluções da AlexNet. Essa escolha é nossa hipótese inicial.
- [Torchvision, código da AlexNet](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html)
  documenta a arquitetura e os pesos usados.
- [PyTorch, Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
  explica congelar parâmetros com `requires_grad=False` e otimizar só os
  parâmetros treináveis.

## Execução

No Apuana, depois de versionar e sincronizar o código conforme `docs/cluster.md`:

```bash
# Teste rápido da execução; números sem valor científico
sbatch -p debug --qos=complex --time=00:30:00 cluster/experimento.sh configs/experimentos/004_gustavo_arquiteturas.yaml --teste

# Parte 1 (depois que a rodada 3 estiver no GitHub e a referência do YAML preenchida)
sbatch cluster/experimento.sh configs/experimentos/004_gustavo_arquiteturas.yaml

# Parte 2, em sequência (cada uma só se a anterior não perdeu; ver docs/fluxo_experimentos.md)
sbatch cluster/experimento.sh configs/experimentos/005_alexnet_congelamento.yaml
sbatch cluster/experimento.sh configs/experimentos/006_alexnet_taxas.yaml
sbatch cluster/experimento.sh configs/experimentos/007_alexnet_inicializacao.yaml
```

A primeira execução precisa ter acesso ao download dos pesos ImageNet, ou os
pesos precisam estar previamente disponíveis no cache do PyTorch do cluster.
Sem os pesos, o construtor falha; ele não segue silenciosamente do zero.

Testes usam tensores e imagens sintéticos. Quando simulam a seleção de pesos
ImageNet, substituem só o download, sem pretender medir transferência de
aprendizado. Os testes verificam também que uma atualização muda pesos profundos
e preserva exatamente os pesos congelados.

```bash
python -m pytest -q tests/test_models.py
python -m pytest -q tests/
```
