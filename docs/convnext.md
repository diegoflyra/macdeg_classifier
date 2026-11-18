# ConvNeXt para DMRI

As ConvNeXt Tiny, Small, Base e Large foram adicionadas como classificadores para o mesmo problema
binário de DMRI. A arquitetura e o forward estão escritos em `macdeg/convnext.py`,
usando nossas camadas com parâmetros explícitos de `macdeg/layers.py`.
Os pesos ImageNet-1K são carregados separadamente dos checkpoints oficiais
de cada variante, equivalentes aos respectivos `IMAGENET1K_V1` do torchvision.
O treino não instancia o modelo pronto do torchvision.

Tiny é a variante inicial. Small e Base ampliam a comparação principal no cluster.
Large tem um lote opcional com Base como controle. Ainda não há desempenho medido em DMRI.

## Arquitetura

| Variante | Blocos nos quatro estágios | Canais nos quatro estágios | Parâmetros com saída binária |
|---|---|---|---|
| Tiny | 3, 3, 9, 3 | 96, 192, 384, 768 | 27.820.897 |
| Small | 3, 3, 27, 3 | 96, 192, 384, 768 | 49.455.457 |
| Base | 3, 3, 27, 3 | 128, 256, 512, 1024 | 87.567.489 |
| Large | 3, 3, 27, 3 | 192, 384, 768, 1536 | 196.231.873 |

A tabela abaixo detalha a Tiny. As quatro variantes usam as mesmas reduções espaciais.

| Etapa | Blocos | Canais | Saída espacial para entrada de 512 px |
|---|---|---|---|
| Stem, convolução 4 × 4 com stride 4 e LayerNorm | — | 96 | 128 × 128 |
| Estágio 1 | 3 | 96 | 128 × 128 |
| LayerNorm e redução 2 × 2, stride 2 | — | 192 | 64 × 64 |
| Estágio 2 | 3 | 192 | 64 × 64 |
| LayerNorm e redução 2 × 2, stride 2 | — | 384 | 32 × 32 |
| Estágio 3 | 9 | 384 | 32 × 32 |
| LayerNorm e redução 2 × 2, stride 2 | — | 768 | 16 × 16 |
| Estágio 4 | 3 | 768 | 16 × 16 |
| Average pooling global | — | 768 | 1 × 1 |
| LayerNorm, flatten e saída linear | — | 1 logit | — |

Cada bloco tem convolução **depthwise 7 × 7**, que aplica um filtro a cada canal
separadamente. Depois, os canais vão para a última dimensão do tensor, recebem
LayerNorm e duas transformações lineares por posição espacial. A primeira expande
os canais em quatro vezes, aplica GELU e a segunda volta ao tamanho original.
O tensor retorna para NCHW, recebe LayerScale e stochastic depth e é somado à
entrada do bloco por uma conexão residual.

A LayerNorm usa epsilon de 1e-6. Sua escala e viés são `nn.Parameter`, assim
como os filtros, pesos lineares e o LayerScale. O LayerScale começa em 1e-6
quando não há checkpoint; no fine-tuning, seus valores vêm dos pesos ImageNet.
A Tiny tem 18 blocos, com stochastic depth crescendo linearmente de 0 até 0,1.
Small, Base e Large têm 36 blocos. Seus construtores usam os padrões oficiais de 0,4,
0,5 e 0,5, respectivamente. `stochastic_depth_prob` permite sobrescrever o máximo.
Nos novos lotes do cluster, o valor é **0,1 entre as variantes de arquitetura**, para manter
a mesma política enquanto se compara a arquitetura. Essa opção não altera os pesos
nem as chaves dos checkpoints. Na avaliação, todos os ramos são usados, sem sorteio.

A expansão linear dentro de cada bloco age sobre os canais de cada posição,
não sobre a imagem inteira achatada. A agregação espacial global só acontece
no fim. Isso difere das densas de 4096 neurônios da AlexNet.

## Pesos pré-treinados e saída binária

Os nomes e formatos dos parâmetros acompanham a variante oficial, incluindo
chaves como `features.1.0.block.0.weight` e `features.1.0.layer_scale`.
O carregamento usa `torch.hub.load_state_dict_from_url`, com hash no download,
`weights_only=True` e correspondência estrita de todas as chaves do state_dict.
Uma falha de carregamento interrompe a execução; não há troca silenciosa para
pesos aleatórios.

Primeiro são carregadas as 1000 saídas originais da ImageNet. Depois, só a última
linear é substituída por uma saída. Ela recebe 768 canais na Tiny e na Small,
1.024 na Base e 1.536 na Large.
A nova saída usa Xavier com viés zero
na configuração proposta; o restante conserva os pesos pré-treinados.
A perda continua sendo `BCEWithLogitsLoss`, e a sigmoide é aplicada na previsão.
O escore produzido não é, por si só, uma probabilidade clínica calibrada.

Os testes incluem transferência estrita de state_dict entre a referência do
torchvision e o nosso modelo, comparação numérica de saídas, gradientes nas
camadas funcionais, preservação dos parâmetros congelados e recarga de pesos
para inferência no mesmo fluxo usado pelo Hugging Face.

## Fine-tuning inicial

```yaml
model: convnext_tiny
pretrained: true
freeze_first_stages: 2
head_init: xavier
lr: 1.0e-5
head_lr: 1.0e-4
image_size: 512
augment: null
```

`freeze_first_stages: 2` congela o stem, o estágio 1, a primeira redução e o
estágio 2 (`features[0:4]`). A redução que entra no estágio 3 fica treinável,
assim como os estágios 3 e 4 e todo o classificador, incluindo sua LayerNorm.
O congelamento usa `requires_grad=False`; só parâmetros treináveis entram no
AdamW. Os módulos congelados ficam em avaliação mesmo após `model.train()`,
desligando o stochastic depth nesse trecho fixo do extrator.

Os demais blocos continuam com stochastic depth durante o treino, que é uma
regularização interna da rede. O congelamento dos dois primeiros estágios e
as taxas de aprendizado são hipóteses iniciais, a serem avaliadas nas dobras.
Valores de `freeze_first_stages` entre 0 e 4 permitem comparar outros cortes.
O padrão do construtor é 0, e os modelos anteriores conservam seus padrões.

O número total de parâmetros com a saída binária é 27.820.897. Com o corte em
dois estágios, 1.235.040 ficam congelados e 26.585.857 ficam treináveis. São
contagens da arquitetura, não resultados de desempenho.

## Dados e avaliação

A entrada mantém a preparação existente do projeto: enquadramento da retina,
preenchimento do quadrado, redimensionamento, RGB, tensor e normalização ImageNet.
O conjunto derivado não corresponde aos arquivos originais intactos.
Os lotes 005 a 009 usam `augment: null`. Os lotes 010 a 013 incluem a receita leve
da rodada multibase durante o treino.
As estatísticas de normalização RGB coincidem com as usadas pelos pesos oficiais.
O checkpoint foi preparado para a receita de ImageNet em 224 px; usar 512 px
mantém o protocolo do projeto e precisa ser avaliado para esta tarefa.

São mantidos os dados, rótulos, divisões por paciente, validação aninhada 5 × 4,
escolha de época pela AUROC interna e limiares escolhidos em outras dobras.
O F2 médio por base continua sendo a métrica principal.
O experimento `005_convnext_tiny_finetuning.yaml` tem uma configuração e 20
treinos completos. Ele não demonstra, sozinho, superioridade contra outra rede.

O experimento `006_convnext_congelamento.yaml` compara a mesma ConvNeXt com todas
as camadas treináveis e com os dois primeiros estágios congelados. Só o corte
de congelamento muda entre as duas configurações; são 40 tarefas completas.
O controle é a primeira configuração, e o veredito continua sendo pareado.

Comparar as receitas finais da AlexNet e da ConvNeXt inclui diferenças de
arquitetura e de política de congelamento. Isso compara as receitas completas;
não isola a contribuição de cada componente.

## Referências

- [Liu et al. (2022), A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).
  Fundamenta a arquitetura ConvNeXt, seus blocos residuais e as modernizações
  feitas em uma CNN. O estudo não valida nosso protocolo de DMRI.
- [Torchvision, implementação de ConvNeXt](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/convnext.html).
  Define as variantes, nomes de parâmetros e checkpoints usados para compatibilidade.
  Os pesos desse checkpoint seguem a receita de treino do torchvision, que difere
  da receita original do artigo. Nossa implementação não reproduz o pré-treino.
- [D2L, Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html).
  Sustenta o reaproveitamento dos pesos e a taxa maior na saída nova.
- [PyTorch, Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html).
  Fundamenta a troca de memória por recálculo no backward e a preservação do sorteio.

## Receita para o cluster

`configs/train/convnext_cluster.yaml` é uma receita candidata. Otimiza o uso de memória
e organiza os ajustes de treino; melhorar a classificação continua sendo uma hipótese
que precisa passar pela comparação pareada do projeto. Os YAMLs 005 e 006 permanecem
com suas receitas originais.

| Opção | Valor no lote novo | Função |
|---|---|---|
| `pretrained`, `head_init` | `true`, `xavier` | Reaproveita ImageNet e inicializa só a saída nova |
| `freeze_first_stages` | 2 | Preserva os estágios iniciais, ajusta os profundos |
| `lr`, `head_lr` | 1e-5, 1e-4 | Taxa menor nos pesos reaproveitados e maior na saída |
| `weight_decay` | 1e-4 | Regularização AdamW dos pesos selecionados |
| `exclude_norm_bias_from_weight_decay` | `true` | Não penaliza vieses, escalas da normalização e LayerScale |
| `max_grad_norm` | 1,0 | Limita a norma global dos gradientes após desfazer a escala do AMP |
| `gradient_checkpointing` | `true` | Guarda menos ativações e recalcula blocos treináveis no backward |
| `batch_size`, `micro_batch` | 16, 2 | Acumula oito microbatches por atualização |
| `num_workers`, `torch_num_threads` | 4, 2 por treino | Ponto de partida para as 16 CPUs do job com duas GPUs |
| `image_size`, `augment` | 512, `null` | Define a resolução e o controle dos experimentos de augmentation |

O checkpointing usa `use_reentrant=False` e preserva o RNG para repetir a mesma
máscara de stochastic depth no recálculo. Não é aplicado aos estágios congelados,
à inferência ou a operações sem gradiente. Pode aumentar o tempo de cada passo:
o ganho pretendido é caber na memória, não uma aceleração já medida.
AMP FP16, formato channels-last, cache de imagens por mmap, warmup, cosine decay
e early stopping já existentes continuam ativos. O pico de memória é reiniciado
por tarefa para não herdar o pico do modelo treinado antes na mesma GPU.
O treino transfere as perdas juntas ao fim de cada época, e a inferência transfere
as probabilidades juntas ao fim de cada conjunto. Isso evita sincronizar CPU e GPU
a cada microbatch só para ler esses valores.
Os checkpoints são desserializados na CPU antes de carregar os parâmetros e o
estado do otimizador nos dispositivos, evitando uma segunda cópia completa na VRAM.

O microbatch é fixo entre modelos e GPUs, e precisa dividir o lote efetivo.
Não há estimativa de memória da ResNet sendo usada para escolher o microbatch
destes lotes. O piloto precisa confirmar que duas imagens cabem na GPU recebida;
se não couberem, crie uma receita nova com `micro_batch: 1`. Igualar o lote efetivo
não implica igualdade bit a bit entre hardwares ou diferentes microbatches.

As opções novas são desligadas por padrão nos experimentos antigos. A candidata
inclui vários ajustes em comum; os lotes abaixo isolam diferenças **dentro dessa
receita**. Eles não medem o efeito isolado de cada ajuste em relação ao experimento 005.

## Lotes de comparação

| Experimento | Configurações | Piloto | Teste rápido | Validação 5 × 4 |
|---|---|---|---|---|
| `007_convnext_arquiteturas.yaml` | Tiny, Small, Base | 3 tarefas | 24 tarefas | 60 tarefas |
| `008_convnext_ajustes.yaml` | Tiny de controle + 5 variações | 6 tarefas | 48 tarefas | 120 tarefas |
| `009_convnext_large.yaml` | Base e Large | 2 tarefas | 16 tarefas | 40 tarefas |
| `010_convnext_augmentation.yaml` | Tiny com e sem augmentation | 2 tarefas | 16 tarefas | 40 tarefas |
| `011_convnext_arquiteturas_augmentation.yaml` | Tiny, Small, Base e Large com augmentation | 4 tarefas | 32 tarefas | 80 tarefas |
| `012_convnext_aux_dr.yaml` | Tiny com augmentation, com e sem cabeça auxiliar de DR | 2 tarefas | 16 tarefas | 40 tarefas |
| `013_convnext_arquiteturas_aux_dr.yaml` | Tiny, Small, Base e Large com augmentation e cabeça de DR | 4 tarefas | 32 tarefas | 80 tarefas |

No 007, só `model` muda: `convnext_arquiteturas-00` é a Tiny de controle, `-01`
é a Small e `-02` é a Base. Todas usam stochastic depth máximo de 0,1, corte em
dois estágios e a mesma política de otimização. Os checkpoints ImageNet também
mudam com a arquitetura; não é um estudo dos tamanhos treinados do zero.

O 008 mantém a Tiny previamente definida, sem escolher a arquitetura pelo teste
externo do 007. As variações são independentes, sem produto cartesiano:

| Configuração | Única mudança em relação a `convnext_ajustes-00` |
|---|---|
| `-00` | Controle da receita candidata |
| `-01` | Congela 1 estágio |
| `-02` | Congela 3 estágios |
| `-03` | Taxa dos pesos reaproveitados passa para 3e-5; saída mantém 1e-4 |
| `-04` | Weight decay passa para 0,01 |
| `-05` | Stochastic depth máximo passa para 0,2 |

O 009 repete a Base com a mesma receita e muda só a arquitetura na Large.
A Large tem 196,2 milhões de parâmetros com a saída binária. Meça seu custo em
512 px com `--piloto` antes de agendar o lote completo.

Os lotes 007 e 008 somam **180 treinos**, incluindo uma repetição do controle Tiny.
O 009 acrescenta 40 tarefas, incluindo a Base de controle. Com duas GPUs, rodam no máximo duas tarefas por vez,
uma por GPU; não se trata de dividir um modelo entre as duas GPUs. Não submeta os
dois lotes simultaneamente no QoS que limita o usuário a duas GPUs.

### Augmentation da rodada multibase

A variação `rodada2-01` de `configs/experimentos/002_rodada2.yaml`, na branch
`experimentos-002`, usa recorte aleatório com 85% a 100% da área e redimensionamento
para 512 px, flip horizontal com probabilidade de 50%, rotação de −15° a +15°
com interpolação bilinear e ajustes de brilho e contraste de até 10%. O cache
guarda a imagem redimensionada. As transformações aleatórias são sorteadas
na leitura das imagens de treino. Validação, teste e avaliação de drusas usam
o redimensionamento determinístico e a normalização ImageNet.

No 010, a Tiny sem augmentation é o controle e a segunda configuração muda
só `augment`. No 011, a Tiny com augmentation é o controle e cada configuração
muda só `model`. As duas configurações da Tiny do 010 usam as mesmas dobras e
a mesma receita de treino. A época é escolhida pela AUROC de validação interna.
O F2 médio por base e o veredito pareado são calculados com as previsões das dobras
externas. A escolha de configuração no resultado aninhado usa a validação interna.

### Cabeça auxiliar de retinopatia diabética

A variação `rodada2-04` da branch `experimentos-002` acrescenta um segundo logit à
mesma camada de saída. O primeiro logit continua supervisionado pelo rótulo de DMRI.
Para o segundo, um olho sem DMRI recebe alvo 1 se `disease` contém a categoria
`retinopatia_diabetica`, e 0 nos demais olhos sem DMRI. Olhos com DMRI ficam fora da
perda auxiliar por terem estado de DR desconhecido no protocolo. Drusas sem diagnóstico
continuam no conjunto de avaliação extra.

A perda total é `BCE de DMRI + 0,5 × BCE de DR`. O `pos_weight` do pipeline vale
para DMRI. A BCE auxiliar é a média entre os alvos conhecidos do microbatch e vale
zero quando todos são desconhecidos. Seleção de época, limiares, métricas e previsão
do ensemble usam só o logit de DMRI. O cartão dos novos modelos registra o número
de saídas para reconstruir a camada na recarga dos pesos.

O 012 mede só o efeito da cabeça auxiliar com a mesma augmentation nas duas configurações
da Tiny. O 013 fixa `aux_dr: 0.5` e a mesma augmentation para comparar as quatro arquiteturas.

Use o piloto de cada lote para medir custo. `analise/RESUMO.md` reúne por configuração
o tempo médio por época, o tempo da tarefa, a GPU usada, o pico de memória alocada,
o pico reservado pelo PyTorch e a VRAM total. Cada `metrics.json` também registra
os parâmetros e `history.csv` registra os passos do otimizador por época.
Para estimar o teto de custo do lote completo, some,
para cada configuração, `20 dobras × 20 épocas × segundos_por_época` e divida por
3600 para GPU-horas. Dividir por duas dá uma aproximação otimista do tempo de parede;
fila, cache, download, inferência, tarefas desiguais e retomadas aumentam o tempo.
Parada antecipada pode reduzi-lo. Não há duração garantida para o limite de 48 horas.

Os lotes têm controles e análises próprios. Comparações exploratórias entre
vencedores de lotes diferentes não substituem a seleção dentro da validação aninhada.

## Execução

No Apuana, depois de versionar e sincronizar o código conforme `docs/cluster.md`:

```bash
# Teste rápido do encadeamento; números sem valor científico
sbatch cluster/experimento.sh configs/experimentos/005_convnext_tiny_finetuning.yaml --teste

# Piloto para medir tempo e memória em uma dobra
sbatch cluster/experimento.sh configs/experimentos/005_convnext_tiny_finetuning.yaml --piloto

# Experimento completo
sbatch cluster/experimento.sh configs/experimentos/005_convnext_tiny_finetuning.yaml

# Comparação controlada do congelamento
sbatch cluster/experimento.sh configs/experimentos/006_convnext_congelamento.yaml
```

Para preparar os checkpoints ImageNet antes do primeiro job, rode no nó de instalação
com o ambiente do projeto ativo. O cache fica disponível para os jobs que usam o
mesmo `$HOME` e `TORCH_HOME`.

```bash
python - <<'PY'
import torch
from macdeg.convnext import CONVNEXT_IMAGENET_URLS

for name, url in CONVNEXT_IMAGENET_URLS.items():
    print(f'Preparando {name}', flush=True)
    torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True, weights_only=True)
PY
```

Para os novos lotes, com o código commitado e sincronizado no Apuana:

```bash
mkdir -p logs

# Verificação rápida das três arquiteturas, sem valor científico
sbatch cluster/experimento.sh configs/experimentos/007_convnext_arquiteturas.yaml --teste

# Uma tarefa real por arquitetura para medir memória e tempo
sbatch cluster/experimento.sh configs/experimentos/007_convnext_arquiteturas.yaml --piloto

# Depois de concluir o piloto e conferir o custo: 60 tarefas
sbatch cluster/experimento.sh configs/experimentos/007_convnext_arquiteturas.yaml

# Depois de terminar o 007: piloto das seis receitas da Tiny
sbatch cluster/experimento.sh configs/experimentos/008_convnext_ajustes.yaml --piloto

# Depois de concluir esse piloto: 120 tarefas
sbatch cluster/experimento.sh configs/experimentos/008_convnext_ajustes.yaml

# Lote opcional de Base e Large: piloto de 2 tarefas, seguido de 40 tarefas no lote completo
sbatch cluster/experimento.sh configs/experimentos/009_convnext_large.yaml --piloto
sbatch cluster/experimento.sh configs/experimentos/009_convnext_large.yaml

# Mede custo de treinamento da Tiny com e sem augmentation
sbatch cluster/experimento.sh configs/experimentos/010_convnext_augmentation.yaml --piloto
# Validação aninhada com e sem a receita de augmentation
sbatch cluster/experimento.sh configs/experimentos/010_convnext_augmentation.yaml

# Quatro arquiteturas com a mesma receita de augmentation
sbatch cluster/experimento.sh configs/experimentos/011_convnext_arquiteturas_augmentation.yaml --piloto
sbatch cluster/experimento.sh configs/experimentos/011_convnext_arquiteturas_augmentation.yaml

# Cabeça auxiliar de DR com augmentation fixo na Tiny
sbatch cluster/experimento.sh configs/experimentos/012_convnext_aux_dr.yaml --piloto
sbatch cluster/experimento.sh configs/experimentos/012_convnext_aux_dr.yaml

# Quatro arquiteturas com augmentation e cabeça de DR
sbatch cluster/experimento.sh configs/experimentos/013_convnext_arquiteturas_aux_dr.yaml --piloto
sbatch cluster/experimento.sh configs/experimentos/013_convnext_arquiteturas_aux_dr.yaml

# Se o tempo do job acabar, retome o caminho exato impresso no log
sbatch cluster/experimento.sh --retomar ~/macdeg_runs/NOME_EXATO_DO_LOTE
```

Execute uma etapa por vez. A retomada pula tarefas concluídas e usa `last.pt` nas
interrompidas. O estado do otimizador, scheduler e scaler é recuperado; o fluxo
atual não restaura todos os estados de RNG e não promete uma trajetória idêntica
à execução contínua. Não apague o lote interrompido. Mudanças de código ou receita
pedem um lote novo. O registro `codigo/` vem do commit, portanto alterações ainda
sem commit não ficam documentadas corretamente no snapshot.

```bash
python -m pytest -q tests/test_convnext.py tests/test_layers.py
python -m pytest -q tests/
```
