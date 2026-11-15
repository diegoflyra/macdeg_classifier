# Hugging Face: como os modelos são publicados

O Hugging Face (HF) é um "GitHub de modelos": cada **repositório de modelo** guarda arquivos de pesos,
uma página (README) e um histórico de versões. Repositórios podem ser **privados** (só você e quem você
convidar veem) ou públicos. Tudo aqui é privado.

## Passo a passo (uma vez)

1. Crie a conta em huggingface.co (pode usar o mesmo e-mail do GitHub).
2. Gere um token: foto do perfil → **Settings → Access Tokens → Create new token**.
   - Tipo **Fine-grained**; em *Repositories*, marque **Write access to contents/settings of all repos
     under your personal namespace** (o pipeline cria o repositório `macdeg_classifier` sozinho).
   - Copie o token (começa com `hf_`). Ele aparece uma vez só.
3. No `.env` da raiz do projeto (e no do clone no cluster), acrescente a linha `HF_TOKEN=hf_...`.
   Nunca escreva o token em notebook, commit ou mensagem.

Pronto: não é preciso criar o repositório à mão.

## Do treino final ao Hugging Face (em dois passos)

**1. Treino final (no cluster):**
`sbatch cluster/experimento.sh --final ~/macdeg_runs/<lote analisado> <configuração>` treina a
configuração nas 5 dobras externas. No fim, a análise **exporta** o ensemble, sem enviar nada:

    ~/macdeg_runs/<lote>-final-<data>/modelo/<configuração>/
      dobra0..4.safetensors   os 5 modelos, em float16 (~47 MB cada)
      config.json             cartão: lote de origem, commit, assinatura dos dados, arquitetura, tamanho
                              da imagem, normalização, limiar e métricas da validação cruzada aninhada

O cartão (sem os pesos) vai também para a branch `resultados` do GitHub, em
`resultados/<lote final>/modelo_<configuração>.json`: é o registro de qual modelo foi o melhor e de
onde ele veio. Os pesos ficam só no cluster (cota de 100 GB do `$HOME`).

Os pesos estão em `safetensors`, não em `joblib`/`pickle`. É o formato padrão para redes do PyTorch,
não executa código ao abrir e é o que o Hugging Face espera.

**2. Publicar, quando quiser (no cluster, com `HF_TOKEN` no `.env`):**
`python -m macdeg.hub publicar ~/macdeg_runs/<lote final> <configuração>`

1. cria (se ainda não existir) o repositório **privado** `<seu-usuário>/macdeg_classifier`;
2. envia a pasta `<configuração>/` (os 5 modelos e o `config.json`);
3. atualiza o `README.md` e o `ranking.json` do repositório: a tabela das configurações publicadas, **do
   pior para o melhor** pelo F2 médio por base.

O plano gratuito tem espaço privado limitado (~235 MB por configuração): publique só o que valer a pena.
Nada de imagens, previsões por imagem ou dados de paciente vai para o HF.

## Usar um modelo publicado

```python
import cv2
from macdeg.hub import load_ensemble
modelo = load_ensemble("<seu-usuário>/macdeg_classifier", "referencia-00")   # do HF (baixa só aquela pasta)
modelo = load_ensemble("~/macdeg_runs/<lote final>", "referencia-00")          # ou da pasta local, no cluster
prob = modelo.predict(cv2.imread("retinografia.jpg"))
print("DMRI" if prob >= modelo.threshold else "sem DMRI", round(prob, 3))
```

## Compartilhar ou abrir ao público depois

- Dar acesso a alguém: na página do repositório, **Settings → Collaborators** (ou crie uma organização).
- Tornar público: **Settings → Change repository visibility**. Antes, confira os termos das bases
  (`docs/` e a conversa sobre o ADAM): a decisão é sua e do professor.
- Apagar uma configuração: na aba **Files**, apague a pasta, ou rode
  `HfApi().delete_folder(path_in_repo="<configuração>", repo_id="<usuário>/macdeg_classifier")`.
