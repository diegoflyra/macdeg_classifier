# Rede híbrida para DMRI

A [proposta em duas resoluções](arquitetura_dmri.pdf) entra no pipeline com `model: hibrida_dmri`.
A entrada é uma retinografia de 1024 × 1024. A ResNet-18 recebe uma cópia de 512 × 512.
Depois da `layer2`, um ramo aplica o bloco multiescala residual com alfa treinável iniciado em 0,1.
O outro passa pelas `layer3` e `layer4` da ResNet. Os dois mapas chegam à atenção espacial.

O softmax espacial pondera o resumo multiescala. O mesmo mapa de escores escolhe três janelas
sem sobreposição. A rede copia diretamente os pixels de 192 × 192 da entrada. Uma CNN compartilhada
extrai um vetor por recorte. A cabeça concatena a média desses vetores, a média ponderada do mapa
multiescala e a média de `layer4`. Uma densa de 128 unidades com ReLU precede o logit de DMRI.

Os pesos ImageNet inicializam a ResNet-18. O stem (`conv1` e `bn1`), `layer1` e `layer2` ficam congelados,
incluindo as estatísticas de BatchNorm. `lr` vale para `layer3` e `layer4`. `head_lr` vale para o bloco
multiescala, a atenção, o extrator de recortes e o classificador.

O lote `014` compara `aux_localization` de 0 e 0,25. Com peso positivo, a média dos três maiores
escores antes do softmax recebe uma perda BCE adicional com o rótulo de DMRI do olho.
A seleção discreta dos recortes não propaga gradiente para suas coordenadas. A média ponderada
também fornece gradiente ao mapa nos dois casos. O lote usa a validação aninhada 5 × 4 e o F2 médio
por base do pipeline. A configuração `hibrida_dmri-00`, com peso zero, é o controle do lote.

## Execução no Apuana

No nó de instalação, com o ambiente do projeto ativo, prepare os pesos da ResNet-18 uma vez.

```bash
python -c 'from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)'
```

Depois de sincronizar o commit no Apuana, rode a partir da raiz do repositório.

```bash
mkdir -p logs
sbatch -p debug --qos=complex --time=00:30:00 cluster/experimento.sh configs/experimentos/014_hibrida_dmri.yaml --teste
```

O teste reduz a entrada para 128 px e os recortes para 24 px. O piloto mede a execução com as
resoluções originais em uma dobra por configuração.

```bash
sbatch cluster/experimento.sh configs/experimentos/014_hibrida_dmri.yaml --piloto
```

Após o piloto, o lote completo roda 40 tarefas. Para dividi-lo entre duas GPUs,
use o [fluxo de dois jobs](cluster.md#cada-experimento) no mesmo lote.

```bash
sbatch cluster/experimento.sh configs/experimentos/014_hibrida_dmri.yaml
```

## Variante com ResNet-50

O lote `015` usa o backbone ResNet-50 com pesos `IMAGENET1K_V2`, os mesmos da ResNet-50
de referência do projeto. O setup do Apuana já prepara esses pesos. Duas convoluções
`1 × 1` projetam a saída de `layer2` de 512 para 128 canais no ramo multiescala e a de
`layer4` de 2048 para 512 canais. Essas projeções treinam com `head_lr`. O ramo profundo da
ResNet recebe a saída da `layer2` com os 512 canais originais.

A receita herda a vista global de 512 px, os três recortes de 192 px e o peso 0,25 da perda
auxiliar. O piloto roda uma tarefa para medir tempo e memória. O lote completo roda 20.

```bash
sbatch cluster/experimento.sh configs/experimentos/015_hibrida_resnet50.yaml --piloto
```
