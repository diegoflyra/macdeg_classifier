# Decisões do projeto

Só agregados, sem IDs. Contexto e justificativas completas: `docs/proposta.md`.

## Conjunto multibase e treino (11/11/2026)

### Dados
- **Bases:** HYAMD, BRSET, ODIR-5K, RFMiD, RFMiD 2.0 e ADAM (`configs/sources.yaml`). AREDS indisponível.
  Kaggle: só a ODIR original; cópias derivadas, OCT e bases já aumentadas descartadas.
- **Tarefa:** DMRI × sem DMRI. "Sem DMRI" = normais + outras doenças (no máximo 1/3, `other_to_normal: 0.5`).
- **Outras doenças** (`macdeg/data/select.py`): cotas iguais por categoria, divididas por igual entre as bases;
  imagem com várias doenças conta para a mais rara; HYAMD e ADAM (sem normais) recebem ao menos tantos
  "sem DMRI" quanto DMRI. Sorteio com semente 0.
- **Fora:** qualidade insuficiente; drusas sem diagnóstico de DMRI (as bases não dão o tamanho; só entram
  na análise da distribuição das notas).
- **Cópias:** exatas pelo SHA1; quase cópias por dhash (≤ 24 bits, após recortar até a retina) confirmadas
  por correlação de detalhes ≥ 0,95. Entre bases fica a da base listada primeiro; rótulos conflitantes saem.
  Dentro da base, cópias ficam no mesmo grupo. Um grupo de 106 imagens da ODIR liga 60 pacientes por falsos
  positivos (fotos parecidas, correlação mediana 0,965): mantido, pois é conservador (fica numa dobra só).
- **Imagens:** retina inteira num quadrado (bordas pretas completam), 1024 px, JPEG 95. Na BRSET o círculo é
  25% mais largo que a altura: um quadrado central perderia as bordas.
- **IDs:** o que sai do cluster usa `sample_id` sequencial; o hash do arquivo original fica só no manifest.
- **Composição:** 18.039 imagens (1.515 DMRI, 11.017 normais, 5.507 outras doenças).

### Avaliação
- **Validação cruzada aninhada** 5 × 4, por paciente/grupo, estratificada por base e tipo de olho.
- **Métrica principal:** média do F2 por base (RFMiD 2.0 conta junto com a RFMiD: só 10 DMRI).
- **Limiar:** sempre escolhido em outras dobras (maximiza a média do F2 por base).
- **Época:** pela AUROC de validação (o F2 oscila demais entre épocas).
- **IC95:** bootstrap por grupo. **Veredito pareado** contra a referência: ganhou / empatou / perdeu.
- **Réguas:** só a base de origem (0,81 na composição inicial) e só a idade (0,83; 0,92 na BRSET).
- **Modelo final:** ensemble das 5 dobras externas; limiar escolhido nas notas fora da dobra.

### Treino e execução
- **Referência:** ResNet-50 ImageNet, 512 px, sem aumento de dados, pos_weight, AdamW + cosseno, AMP.
- **Lote efetivo fixo (16)** em qualquer GPU (acumulação de gradientes; desde a rodada 3, micro-lote calculado para a menor GPU, para o BatchNorm também não depender da GPU).
- **Apuana:** partição `short-simple` (QoS `simple`: até 2 GPUs por usuário), sem preempção; GPUs RTX 3090
  ou fatias de A100 (MIG, ~10 GB). Ambiente `venv` com o Python do sistema (3.10.12; os módulos vão só até
  3.10.8), instalado no nó `install`; pacotes sem versão fixa no cluster, registradas em `freeze.txt`. Dados descompactados no `/tmp`
  do nó e apagados no fim de cada job. Código fixado no commit do lançamento; assinatura dos dados
  conferida em cada tarefa. Pesos só guardados no treino final.
- **Saídas:** W&B só com números (o projeto fica no time da UFPE: a conta não tem entidade pessoal);
  GitHub, branch `resultados`, com agregados e previsões por `sample_id`; Hugging Face, repositório
  privado `macdeg_classifier` (tipo model), só modelos finais.

### Licenças (resumo)
- PhysioNet (HYAMD, BRSET): não compartilhar acesso aos dados; uso só em pesquisa; código de publicações
  aberto. Não tratam de pesos de modelos.
- RFMiD: CC BY 4.0. ODIR: sem licença declarada.
- ADAM: cadastro feito; pesquisa não comercial; publicação científica (inclusive relatório técnico e
  apresentação) exige aprovação por e-mail dos organizadores; resultados (modelos) pertencem ao participante.

## Rodada 2 e modelo final (15/11/2026)

Resultado da referência (`referencia-20261112-1618`): F2 médio por base 0,770 [0,749; 0,789], AUROC 0,967.
Pontos fracos: especificidade 0,43 nos olhos com retinopatia diabética (HYAMD: AUROC 0,77), ODIR com F2
0,65, melhor época cedo (2-6, sobreajuste) e limiar variando entre as dobras (0,18 a 0,63).

- **Olho com DMRI e outra doença (inclusive DR) é positivo.** Já era o comportamento dos leitores das
  bases; registrado como decisão (consistente com a literatura: as lesões de uma não anulam a outra).
- **Uma mudança por variação** (`configs/experimentos/002_rodada2.yaml`): A augmentation, B olhos com DR
  sorteados 3x mais (`oversample`), C CLAHE (`preprocess: clahe`), D cabeça auxiliar de DR (`aux_dr`),
  E 1024 px. As que ganharem no veredito pareado se combinam na rodada 3. Arquitetura (EfficientNet,
  RETFound, FLAIR) fica para depois.
- **A referência é retreinada dentro da rodada** (`rodada2-00`), com o mesmo código das variações. Ela
  também serve para conferir que a otimização de velocidade não mudou o resultado da referência antiga.
- **Alvo da cabeça de DR:** 1 nos "sem DMRI" com DR, 0 nos outros "sem DMRI", desconhecido (fora da
  perda) nos olhos com DMRI, porque as bases não dizem se eles também têm DR.
- **Discordâncias pareadas** (`discordancias.csv` e o RESUMO): DMRI detectada só pela referência,
  só pela variação ou por nenhuma, e falsos positivos corrigidos ou criados, por base, estágio e doença.
  Responde "que diagnósticos se perdem em 512 px?" melhor que o F2 sozinho.
- **FLAIR:** pré-treinado com textos de diagnóstico de bases que usamos como teste (ODIR, RFMiD,
  BRSET). Se for usado, a avaliação nessas bases fica otimista. Decisão adiada para a rodada de arquitetura.
- **Modelo final:** o `--final` exporta o ensemble para `<lote>/modelo/<configuração>/` (safetensors
  float16 + cartão com a origem e as métricas; o cartão vai ao GitHub). O Hugging Face é manual
  (`python -m macdeg.hub publicar`).
- Leitura crítica do levantamento bibliográfico feito com o Gemini (fora do repositório): vários números
  não se confirmaram nas fontes (ex.: um artigo de OCT citado como de retinografia); só entram no texto
  referências conferidas.

## Rodada 3 (17/11/2026)

Resultado da rodada 2 (`rodada2-20261115-1245`, contra a `rodada2-00`): nenhuma variação ganhou.
Augmentation +0,013 [−0,000; +0,028] e cabeça de DR +0,012 [−0,001; +0,025] empataram no limite; sorteio
de DR e CLAHE trocaram DMRI precoce por especificidade em DR; 1024 px perdeu (−0,036 [−0,053; −0,021]).
A `rodada2-00` empatou com a referência antiga (−0,007), mas 5,6% das previsões mudaram entre os dois
treinos idênticos (cuDNN não determinístico): diferenças de ~0,01 estão dentro dessa variação.

- **AUROC por estágio** (acrescentada ao relatório; não substitui a métrica principal): só o HYAMD estadia,
  e os "sem DMRI" dele são todos olhos com DR. DMRI precoce × DR: 0,49 a 0,63 em todas as configurações
  (perto do acaso); intermediária/tardia × DR: ~0,83. O modelo detecta DMRI estabelecida, mas não separa
  drusas iniciais de DR.
- **Só 512 px daqui em diante.** 1024 px perdeu e custa 2–4× mais GPU.
- **Fluxo por rodadas:** a base nova incorpora o que chegou mais perto de ganhar (augmentation + cabeça de
  DR, `base_mudancas`); cada variação muda uma coisa sobre ela. Hiperparâmetros fixos (receita da ResNet)
  até o fim; o ajuste (lr × weight decay) fica para a última rodada. Limitação: com a mesma receita, uma
  arquitetura pode perder por receita, não por arquitetura.
- **Arquiteturas:** DenseNet-121, EfficientNetV2-S e EfficientNet-B4 (pré-treinos em 224, 300–384 e 380 px).
  Cada rede usa a normalização do seu pré-treino. AlexNet (histórica) e ConvNeXt ficam com o Gustavo (`004_gustavo_arquiteturas.yaml`), com a
  mesma base e referência = `rodada3-00`.
- **Recorte central da mácula (`crop: 0.6`):** quadrado central com 60% do lado da imagem de 1024 px,
  reduzido a 512 px (1,67× mais pixels na mácula). Estudo prévio (só agregados): na fóvea marcada à mão no
  ADAM (915 imagens), supor a fóvea no centro erra 0,39 DD na mediana (94% a menos de 1 DD); o detector
  clássico (`localization.py`, adaptado sem o lado do olho) erra mais, 0,49 DD (80%). Pelo detector, todas
  as bases são centradas na mácula: 0,2% das fóveas fora de um recorte de 50% (nenhuma com DMRI). Por isso
  o recorte é fixo, sem detector nem rede de localização. 60% em vez de 50%: a grade do AREDS (raio
  ~1,7 DD) cabe inteira em quase todas as imagens.
- **Micro-lote calculado para a menor GPU (24 GB) e a arquitetura:** a acumulação de gradientes iguala a
  atualização, mas não o BatchNorm (médias dentro do micro-lote). Agora o micro-lote é o mesmo em qualquer
  GPU. Em 512 px, a ResNet-50 continua com 16 (como em todas as tarefas de 512 px da rodada 2).

## Rodadas 4 e 5 (19/11/2026)

Resultado da rodada 3 (`rodada3-20261117-1754`, contra a `rodada3-00`): a base nova (augmentation + cabeça
de DR) ganhou da `rodada2-00` (+0,038 [+0,023; +0,053], comparação pareada entre lotes, mesma assinatura).
Arquiteturas empataram com a ResNet-50: EfficientNetV2-S +0,008 [−0,005; +0,021] (maiores AUROC 0,977 e
AUPRC 0,816; melhor DMRI precoce × DR, 0,699), EfficientNet-B4 +0,009 [−0,006; +0,024] (melhor
especificidade em DR, pior DMRI precoce; instabilidade de NaN numa tarefa), DenseNet-121 −0,008. O recorte
da mácula perdeu (−0,040 [−0,054; −0,026]; ODIR caiu de 0,670 para 0,595). 75 DMRI não são detectadas por
nenhuma configuração (36 no ODIR).

- **Base da rodada 4 = EfficientNetV2-S** (decisão dos donos), embora só tenha empatado: à frente em quase
  todas as métricas, mais barata e estável que a B4. O 004 do Gustavo continua sobre a ResNet-50.
- **Rodada 4 (`008_rodada4.yaml`):** fecha as arquiteturas importadas com Inception-v3 e ViT-B/16.
  O ViT tem entrada fixa: `build_model` passa `image_size` e o timm reamostra a posição pré-treinada
  (384 → 512 px). Limitação: a receita comum (lr 3e-4) é alta para ViT; derrota = "nesta receita".
  Memória medida em 512 px (múltiplos da ResNet-50): V2-S 1,9; Inception-v3 0,75; ViT-B/16 1,6.
- **Todas as redes recebem 512 px.** A vantagem das EfficientNets não vem da resolução de entrada; hipóteses:
  escala do pré-treino (380 px contra 224 px da ResNet, e a ResNet em 1024 px perdeu) e dados do pré-treino
  (ImageNet-21k, JFT contra ImageNet-1k). A ConvNeXt (22k, 384 px) ajuda a separar.
- **Idade: nunca como entrada da rede.** Exploratório (decidido depois de ver os dados, não vale como
  resultado): nota da V2-S + idade numa regressão logística ajustada nas outras dobras deu +0,013
  [+0,008; +0,020], mas a sensibilidade abaixo de 55 anos caiu (0,68 → 0,64) e os falsos positivos acima
  de 75 subiram (15% → 18%): a idade empurra para "velho = DMRI". Se entrar no modelo final, será combinada
  depois da rede, declarada antes, e sempre com o desempenho por faixa etária.
- **Rodada 5 (`009_rodada5.yaml`): saída auxiliar de drusas (`aux_drusas`).** Muda a regra "drusas sem
  diagnóstico ficam fora": agora ficam fora da tarefa principal e da avaliação, mas entram no treino da
  saída de drusas. O que não muda: manifest, divisões, assinatura (`72e45dfc1003894a`) e métrica binária.
  - Alvo: 1 = drusas sem diagnóstico (2832: BRSET 2442, RFMiD 230, ODIR 160); 0 = "sem DMRI" de BRSET,
    ODIR e RFMiD (nelas, todo olho com drusas sem DMRI foi separado, então os outros não têm drusas);
    desconhecido (fora da perda) nos olhos com DMRI e nos "sem DMRI" de HYAMD e ADAM. As drusas ficam fora
    da perda de DMRI.
  - Dobras por paciente (`python -m macdeg.data.drusas` → `drusas_dobras.csv`, gerado onde estão as bases
    originais e copiado para o conjunto): 489 drusas são de pacientes do conjunto principal (BRSET 444,
    ODIR 45) e herdam o grupo e as dobras deles; nenhum grupo atravessa dobras. Pacientes só com drusas
    ganham grupo próprio sorteado (semente 0). RFMiD sem paciente: grupo por imagem, como no principal
    (cópias entre RFMiD e RFMiD 2.0 não foram conferidas para as drusas).
  - Treino: drusas das dobras de treino; teste: só as da dobra externa, pontuadas pelas duas saídas;
    validação interna: sem drusas (a época continua pela AUROC de DMRI).
  - Leitura: a idade só entra na análise (`drusas_por_idade`: notas das drusas por faixa etária). Se a
    variação ganhar, o ganho vem de imagens a mais e de uma tarefa a mais, juntas; o texto diz assim.
  - Não interpreta o que as drusas significam para a idade (Ferris et al., 2013): não há rótulo para isso.

## Fase 2: rodada 5 (20/11/2026)

Resultado da rodada 4 (`rodada4-20261119-0229`): Inception-v3 empatou (−0,010 [−0,023; +0,004]); o ViT-B/16
não aprendeu com a receita comum (perda de treino parada em ~1,3; F2 0,465): limitação da receita, não
conclusão sobre a arquitetura. A V2-S retreinada empatou com a da rodada 3 (−0,005).

- **Ruído entre treinos idênticos:** as duas V2-S (mesma configuração e semente; só o não determinismo da
  GPU muda) diferem até +0,040 no F2 do BRSET e −0,029 no RFMiD, com IC95 sem cruzar o zero. O IC do
  veredito só reamostra pacientes, não a variação do treino. Logo, "a rede X é melhor no aspecto Y" a
  partir de um treino só não se sustenta; a média dos dois treinos ganhou de um só (+0,013 [+0,005; +0,022]).
  Conclusão principal no texto: com dois treinos de cada lado.
- **Arquitetura a partir do problema, não das diferenças entre redes:** a 512 px (~25 µm/px), uma drusa
  média (63–125 µm, DMRI precoce pelo Ferris et al., 2013) tem 2,5–5 px; a 1024 px, 5–10 px. Na V2-S, o
  estágio 2 (64 × 64, 8 px por posição) é o último em que ela tem endereço próprio.
- **Rede híbrida do Gustavo** (branch `rede-hibrida`, inspirada em GMIC, attention MIL e Inception-ResNet)
  portada para a V2-S (`hibrida_v2s`, `macdeg/hybrid.py`): ramo multiescala do estágio 2, atenção espacial,
  3 recortes de 192 px em 1024 px e perda de localização (0,25). Mudança em relação ao original: os
  recortes passam por uma cópia pré-treinada dos estágios 0–3 da V2-S (no original, uma CNN de 3 camadas
  começava do zero com ~900 DMRI). Usa a receita da base (augmentation, cabeça de DR, lr 3e-4).
- **Recorte da mácula em resolução nativa** (`v2s_macula`): vista global 512 px + centro de 448 px da imagem
  em 1024 px (~6 mm, a grade do AREDS), cada um com a sua V2-S. Diferente do `crop: 0.6` da rodada 3, que
  substituía a imagem e a reduzia.
- **Explicabilidade em todas as tarefas** (`macdeg/explain.py`, `explain: true` por padrão): Grad-CAM da
  saída de DMRI (e de drusas), atenção da híbrida e mapa do ramo da mácula, nas imagens do teste externo.
  No GitHub só números (massa na mácula, fora da retina, área com metade do mapa, por grupo e por erro);
  pranchas com imagens só no cluster/local (`python -m macdeg.explain pranchas`).
- Entrada de 1024 px sem cache (os pixels são os mesmos; o cache teria ~65 GB). Memória medida: 0,51 GB
  por imagem (híbrida) e 0,72 (mácula), contra 0,40 da V2-S em 512 px: o lote inteiro (16) cabe em 24 GB,
  então o BatchNorm vê os mesmos 16 da base.

## Fase 2: rodada 6, modelo final (21/11/2026)

Prazo: entrega em 23/11. Vereditos **parciais** da rodada 5 (`rodada5-20261120-0225`, análise com 31
tarefas faltando; a híbrida, 03, ainda sem tarefas), contra a `rodada4-00`: nada ganhou.

| variação | dobras externas | Δ F2 médio por base [IC95] | veredito |
|---|---|---|---|
| 01 saída de drusas | 5/5 | +0,001 [−0,013; +0,015] | empatou |
| 02 mácula em 1024 px | 4/5 | +0,001 [−0,014; +0,015] | empatou |
| 04 augmentation ampliada | 3/5 | +0,010 [−0,007; +0,026] | empatou |

- **Especificidade em retinopatia diabética oscila ±0,1 entre treinos idênticos:** as duas V2-S iguais
  (rodada3-02 × rodada4-00) deram 0,422 × 0,528 nas mesmas 671 imagens. Por isso o +0,07 das drusas e o
  −0,05 da augmentation ampliada nessa métrica não se distinguem do ruído (um treino de cada lado).
- **Drusas entram no modelo final por decisão dos donos (21/11), por cobertura, não por melhora:** olhos com
  drusas sem diagnóstico existem na população rastreada, e a saída de drusas não alterou a métrica
  principal. Comportamento nas drusas da dobra externa (sem rótulo de DMRI, logo fora do F2): marcadas como
  DMRI 16,9% (V2-S) × 22,3% (com a saída), crescendo com a idade (15% até 55 anos; 39% acima de 75), sem
  a idade entrar na rede: coerente com a definição de Ferris (drusas maiores e mais numerosas com a idade).
- **Rodada 6** (`010_rodada6.yaml`): híbrida + drusas, 20 tarefas, comparada com a `rodada4-00`. A
  `rodada5-03` (híbrida sem drusas) mede o efeito das drusas na híbrida e é o modelo de reserva.
  Junta duas mudanças de propósito (modelo final, não "uma por vez").
- **Sem Optuna nesta entrega:** com a validação aninhada, uma busca custaria centenas de GPU-h; ajustar numa
  divisão só e avaliar nas 5 dobras vazaria o teste. Mudanças de receita moveram o F2 em no máximo ~0,013
  (rodada 2), dentro da variação entre treinos. Os hiperparâmetros próprios da híbrida (alfa 0,1, peso de
  localização 0,25, 3 recortes de 192 px, lr única) nunca foram ajustados: se ela perder, não se separa
  "a ideia" de "o ajuste".
- **Previsão em fp32 quando o fp16 estoura** (`predict`): uma tarefa da `rodada5-02` caiu com 2 de 3609
  notas não finitas. Só o lote com NaN é refeito; as notas finitas não mudam (resultados comparáveis).
- **Execução dividida entre duas contas** (`--tarefas`): sem ACL no cluster, cada conta rodou 10 tarefas num
  lote próprio (mesmo YAML, commit `4e80f14` e assinatura); as pastas das tarefas foram juntadas no lote
  `rodada6-20261121-2030` antes da análise. ~12 h com 4 GPUs. A `rodada5-03` não rodou (GPUs para a 6).

### Resultado da rodada 6 (22/11/2026): modelo final

`rodada6-01`: F2 médio por base **0,810 [0,791; 0,829]**, AUROC 0,977, sensibilidade 0,855, especificidade
0,992 em normais e 0,902 em outras doenças. Réguas: origem 0,792 e idade 0,796 de AUROC (a rede: 0,977).

| comparação pareada (mesmas 18 039 imagens) | Δ F2 médio por base [IC95] | veredito |
|---|---|---|
| contra a referência inicial (ResNet-50, `referencia-00`) | +0,040 [+0,025; +0,057] | **ganhou** |
| contra a V2-S (`rodada4-00`, referência do YAML) | +0,006 [−0,007; +0,020] | empatou |
| contra a V2-S + drusas (`rodada5-01`) | +0,005 [−0,009; +0,020] | empatou |

- **Escolhida como modelo final:** maior F2 entre as configurações com as 5 dobras, empate estatístico com
  a V2-S, e duas vantagens fora do F2: cobertura das drusas e explicabilidade (abaixo). O ganho de todo o
  projeto contra a referência inicial é significativo; nenhuma rodada isolada depois da 3 ganhou.
- **F2 por base** (V2-S → híbrida): ADAM 0,879 → 0,874; BRSET 0,795 → 0,835; HYAMD 0,822 → 0,821;
  ODIR 0,699 → 0,700; RFMiD 0,826 → 0,822.
- **DMRI precoce continua o ponto fraco:** AUROC 0,585 (108 olhos, HYAMD); contra a V2-S, a híbrida perdeu 15
  e ganhou 3 desses olhos. A intermediária/tardia: AUROC 0,859.
- **Retinopatia diabética:** especificidade 0,550 (V2-S 0,528), dentro do ruído de ±0,1.
- **Drusas da dobra externa:** 19,3% marcadas como DMRI (V2-S + drusas: 22,3%), crescendo com a idade
  (12,7% até 55 anos → 31,9% acima de 75), sem a idade entrar na rede.
- **Explicabilidade** (massa do mapa dentro da mácula, raio de 3 mm do AREDS; um mapa uniforme dá 0,166):
  - Grad-CAM da saída de DMRI, olhos com DMRI: **0,536** na híbrida × 0,350 na V2-S + drusas, e mais
    concentrado (metade da massa em 9% da imagem × 16%). Em normais, 0,216. A decisão olha para a mácula.
  - **A atenção que escolhe os recortes da híbrida não está na mácula:** 0,094 nos olhos com DMRI (abaixo do
    uniforme), e nos normais 32% da massa cai fora da retina. A perda de localização só treina os maiores
    escores nos olhos com DMRI; nos sem DMRI a atenção não tem alvo. Não apresentar esse mapa como
    "onde está a lesão". Se os recortes ajudam ou não, só uma ablação diria (não feita).

## Pré-processamento por ROI e auditoria (HYAMD, suspenso em 11/11/2026)

Desenvolvido antes da mudança para o conjunto multibase. Fica disponível para voltar como experimento
comparado com a referência.

Só agregados, sem IDs.

### Localização
- **Disco:** busca só no lado nasal, com abertura morfológica (remove manchas menores que ~0,45 DD, como drusas) e densidade de vasos. Motivo: na inspeção visual, o detector anterior circulava drusas.
- **Fóvea:** disco + vetor médio (−0,61; +0,09) R. A mancha escura fica só como diagnóstico (`fovea_disagree`).
- **Disco incerto:** usa a posição típica do formato (`fovea_method=width_prior`) e sinaliza. Taxa na base inteira: 11,5% (8,3% controle, 16,7% precoce, 18,2% intermediária/tardia), acima do limite de 10%.
- Sem verdade de referência: a validação depende das grades.

### Desvio da especificação: DD
A especificação pede DD por imagem. Uso 0,247 R em todas, supondo 2,5 DD entre disco e fóvea, com raio de 945 px nas imagens de campo suspeito. A medida por imagem foi válida em 19% da base; misturá-la faria a escala do recorte depender da qualidade do olho. Ela fica em `dd_measured_px`. Mediana das medidas válidas: 0,268 R.

### Variantes
- **R2.5** foi acrescentada. **R3.0** só aceita G0 e G1, com `margin_black`.
- A rotação de cada ROI é decidida pelos dados: exige margem em ≥ 90% das imagens (`flag_summary.json`). Numa variante com rotação, imagem sem margem usa D4 em vez de criar preto.
- **A1** "melhor entre G1 e G2": as duas opções ficam registradas (`A1-G1`, `A1-G2`), e o mesmo vale para A2 e A3.
- **A3:** Q1 a Q4 como OneOf com p=1, depois do recorte. **Q2:** desvio de 0,005 a 0,02 na escala [0, 1].
- **F1:** ganho multiplicativo e contraste em torno da média do campo. Pixels pretos não entram nem mudam.
- A janela pode se deslocar até 0,5 × o meio lado do ROI para caber no campo (`shift_px`).
- **Excluídas:** 3 imagens com lado `D`/`E` (`excluded.csv`).

### Auditoria
- Acréscimos à especificação: `spurious_rate` (estruturas novas, pelo princípio de não criar drusas), `retention_low_contrast` e `frac_in_frame`.
- Tolerância: máx(1,5 px; 0,3 × menor diâmetro na escala final).
- Limiar: percentil 98 das respostas positivas, calibrado em R2.0 nativo.
- Controles: comparação pareada com A0, IC de 95%. Perda esperada: Q1 e Q3 em retenção e energia; Q2 em estruturas novas; Q4 em retenção e contraste.
- Q4 passou a exigir também retenção depois de um teste sintético, antes da auditoria na base inteira.

### Como ler os resultados
- Retenção de G2 acima de A0 não significa ganho: G2 avalia uma região diferente, porque os cantos do ROI saem do quadro (`roi_area_lost` ≈ 9%).
- Ruído pode inflar a retenção por coincidência com detecções falsas; leia `spurious_rate` junto.
