# CLAUDE.md: regras do projeto macdeg_classifier

Classificador de DMRI (degeneração macular relacionada à idade) em retinografias de várias bases.
Este arquivo resume as decisões que sustentam o pipeline. **Quem mexer no código (pessoa ou agente)
segue estas regras.** Se uma tarefa exigir quebrar uma delas, pare e pergunte aos donos do projeto
antes de mudar. Detalhes e motivos: `docs/decisoes.md` e `docs/proposta.md`. Responda em português.

## 1. Dados: o que nunca pode acontecer

- **Nenhuma imagem, nenhum dado de paciente sai do computador local ou do cluster Apuana.** As bases
  são de acesso restrito (PhysioNet) ou têm acordo de uso (ADAM). Nada de enviar imagens, CSVs das
  bases ou trechos deles a serviços externos, APIs, issues, PRs ou mensagens.
- **Não abrir imagens de pacientes** com ferramentas de leitura/visualização do agente, nem notebooks
  executados que possam conter imagens. Trabalhe com agregados (contagens, médias, quantis).
- **Não imprimir identificadores** (image_id, patient_id, nomes de arquivo das bases, `uid`). Em logs e
  saídas, só agregados. O que sai do cluster usa apenas `sample_id` (`img_00001`, `dru_00001`).
- **Nunca alterar os originais** das bases. Saídas derivadas vão para diretórios separados (`~/macdeg_data/`).
- **Segredos:** `.env` (GITHUB_TOKEN, WANDB_API_KEY, HF_TOKEN). Nunca ler, imprimir, copiar para notebook,
  commit ou mensagem. O código só usa `macdeg.ambiente.secret()` e `which_secrets()` (True/False).
- O `.gitignore` bloqueia imagens, planilhas, `manifest.csv`, divisões, pesos e execuções. **Não afrouxe
  essas regras.** Figuras próprias de documentação só em `docs/figuras/`.

## 2. Definição do problema (não mude sem decisão dos donos)

- Tarefa binária: **DMRI (1) × sem DMRI (0)**. "Sem DMRI" = olhos normais + outras doenças (no máximo 1/3,
  cotas iguais por doença; HYAMD e ADAM com ao menos tantos "sem DMRI" quanto DMRI). Regras em
  `configs/sources.yaml` (`selection`) e `macdeg/data/select.py`.
- Ficam fora: qualidade insuficiente e drusas sem diagnóstico de DMRI. Estas últimas só são pontuadas
  (`extras_drusas.csv`) para a análise da distribuição das notas.
- Bases: HYAMD, BRSET, ODIR, RFMiD, RFMiD 2.0, ADAM. O ADAM precisa continuar **desligável** (`sources`
  na configuração). Publicação científica com ele exige aprovação dos organizadores.
- Cópias: exatas por SHA1; quase cópias por dhash + correlação de detalhes ≥ 0,95 (`macdeg/data/build.py`).
  Não mexa nos limiares sem recalibrar com pares conhecidos.
- **Refazer o conjunto ou as divisões muda a assinatura dos dados** (`macdeg.cluster.dataset_signature`).
  Resultados de assinaturas diferentes não são comparáveis. Mudança de dados = experimento novo, decidido
  pelos donos, nunca efeito colateral de outra tarefa.

## 3. Avaliação: invariantes

- **Validação cruzada aninhada 5 × 4**, por paciente/grupo (`splits_nested_seed0.csv`). A dobra externa k
  **nunca** participa de nenhuma escolha (época, limiar, configuração, hiperparâmetro) feita para ela.
- **Métrica principal: média do F2 dentro de cada base** (`metrics.f2_mean`; RFMiD 2.0 conta com a RFMiD).
- **Limiar sempre escolhido em outras dobras** (`metrics.crossfit`, `best_threshold`). Nunca nas mesmas
  imagens em que se mede.
- **Época pela AUROC de validação**; o F2 compara experimentos.
- IC95 por bootstrap **por grupo** (paciente). Comparação contra a referência sempre **pareada nas mesmas
  imagens** (`metrics.paired_verdict`): ganhou / empatou / perdeu. Só "ganhou" permite dizer "melhorou".
- Todo resultado vem com as **réguas**: só a base de origem e só a idade. O modelo precisa superá-las.
- Também registrados: F2 geral, AUROC, AUPRC, sensibilidade, especificidade em normais, em outras
  doenças e por doença, tudo também por base.
- Modelo final = **ensemble das 5 dobras externas**, limiar escolhido nas notas fora da dobra.
- Não troque a definição de uma métrica existente. Se precisar de outra, **acrescente** uma nova.

## 4. Como fazer um experimento

- **Mude uma coisa por vez** em relação à referência. Crie um YAML novo em `configs/experimentos/`
  (`base`, `grade`, `variacoes`, `referencia: resultados/<lote>/<configuração>`). Veja `exemplo_grade.yaml`.
- **Não edite** `configs/train/baseline.yaml` nem um YAML de experimento já lançado: crie outro. A
  referência atual é `001_referencia.yaml` (ResNet-50 ImageNet, 512 px, sem aumento de dados).
- Opções novas de treino entram como **chave de configuração com padrão que mantém o comportamento
  atual** (ex.: `augment: null` = sem aumento). Assim, experimentos antigos continuam reproduzíveis.
- Grid search = `grade` no YAML. Cada combinação × 5 externas × 4 internas vira uma tarefa do job array.

## 5. Estrutura do código (contratos que a análise e os testes esperam)

- **Lógica no pacote `macdeg/`, com testes. Notebooks são finos**: configuram, chamam o pacote, mostram
  texto. Não coloque lógica de treino ou métrica em notebook.
- Notebooks: `02_lancamento` (login do cluster), `03_tarefa` (executado pelo SLURM via papermill, com
  célula `parameters`: LOTE, TAREFA, DADOS, WANDB_MODE), `04_analise` (LOTE, DADOS, ENVIAR). Mantenha os
  nomes e os parâmetros: `macdeg/cluster.py` os chama pelo nome.
- Contratos de arquivos (mudar exige atualizar `analysis.py`, `cluster.py` e os testes juntos):
  - `manifest.csv`: uid, sample_id, source, label, label_detail, disease, disease_group, patient_id,
    eye, age, sex, camera, group_id, width, height, image, path.
  - `splits_nested_seed0.csv`: uid, outer, inner_0..inner_4 (-1 = teste daquela dobra externa).
  - Previsões (`val/test/extra_predictions.csv`): sample_id, source, group_id, label, kind,
    disease_group, prob. **Nunca** uid, path ou nome de arquivo.
  - Lote: `lote.json`, `tarefas/<configuração>/outer<k>_inner<j>|final/`, `notebooks/`, `logs/`,
    `analise/`. `macdeg.train.run(config, outer, inner, seed, out, log, on_model)`.
- Lote efetivo fixo (`batch_size`), com acumulação de gradientes conforme a GPU (`micro_batch`): o
  resultado não pode depender da GPU sorteada.
- Pesos: `last.pt` é apagado ao fim da tarefa; `best.pt` só fica no treino final (cota do `$HOME`).

## 6. Cluster, serviços e Git

- Execução pelo `python -m macdeg.cluster lancar <experimento> [--teste]`. O lançador fixa o **commit**
  (código não commitado não roda) e a assinatura dos dados. Apuana: partição `short-simple`, QoS
  `simple` (no máximo 2 GPUs por usuário), ambiente `venv` dentro do repositório (`.venv` ou `venv`; Python do sistema, 3.10)
  criado no nó `install` (`cluster/setup_apuana.sh`). Nada pesado no nó de login. Detalhes em `docs/cluster.md`.
- **W&B:** só números (curvas, métricas, histogramas de pesos). Nunca imagens nem tabelas por imagem.
- **GitHub:** a branch órfã `resultados` é escrita **só pelo pipeline** (`macdeg.ambiente.push_results`).
  Não faça commits manuais nela. Trabalho de código em branches próprias, com PR para `main`.
- **Hugging Face:** repositório **privado** `macdeg_classifier` (tipo model), só modelos finais de poucas
  configurações (`macdeg/hub.py`, `docs/huggingface.md`). Torná-lo público é decisão dos donos.
- Notebooks entram na `main` sem saídas (filtro `scripts/nb_clean`). Executados só na branch `resultados`,
  sem imagens.
- Não faça commit, push ou PR a menos que a pessoa responsável peça.

## 7. Antes de entregar qualquer mudança

- `.venv/bin/python -m pytest -q tests/` precisa passar inteiro. O teste de ponta a ponta é
  `tests/test_experimento.py` (lançamento → tarefas → análise com dados sintéticos).
- Funcionalidade nova = teste novo. Mudança de contrato = testes atualizados no mesmo commit.
- O `/tmp` local é pequeno (tmpfs de ~5 GB): testes que salvam pesos devem apagá-los.
- Para conferir no cluster sem gastar: `--teste` (8 tarefas minúsculas; números sem valor).

## 8. Quem decide o quê

- **Decisões de pesquisa** (dados, rótulos, métricas, o que entra ou sai, publicação, licenças): dos
  donos do projeto. Pergunte.
- **Decisões técnicas de execução** (orquestração, scripts, desempenho): pode decidir, mas explique o que
  a escolha significa e por que é a melhor.
- Relate resultados com honestidade: teste que falhou é relatado com a saída; nada é dado como
  verificado sem ter sido rodado.
