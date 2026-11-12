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
- **Lote efetivo fixo (16)** em qualquer GPU (acumulação de gradientes): o resultado não depende da GPU.
- **Apuana:** partição `short-simple` (QoS `simple`: até 2 GPUs por usuário), sem preempção; GPUs RTX 3090
  ou fatias de A100 (MIG, ~10 GB). Ambiente `module load Python` + `venv` (guia oficial), instalado no nó
  `install`; pacotes sem versão fixa no cluster, registradas em `freeze.txt`. Dados descompactados no `/tmp`
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
