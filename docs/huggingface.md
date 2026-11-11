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

## O que acontece quando um modelo é publicado

`python -m macdeg.cluster final <lote analisado> <configuração>` treina a configuração nas 5 dobras
externas e, no fim, a análise:

1. cria (se ainda não existir) o repositório **privado** `<seu-usuário>/macdeg_classifier`;
2. envia a pasta `<configuração>/` com os 5 modelos (`dobra0..4.safetensors`, em float16, ~47 MB cada)
   e um `config.json` com arquitetura, tamanho da imagem, normalização, **limiar**, commit, assinatura
   dos dados e as métricas da validação cruzada aninhada;
3. atualiza o `README.md` e o `ranking.json` do repositório: a tabela das configurações publicadas, **do
   pior para o melhor** pelo F2 médio por base.

Só os modelos finais de poucas configurações vão para o HF (~235 MB cada): os 20 modelos internos de
cada configuração ficam no cluster. O plano gratuito tem espaço privado limitado, então publique só o
que valer a pena.

Nada de imagens, previsões por imagem ou dados de paciente vai para o HF.

## Usar um modelo publicado

```python
import cv2
from macdeg.hub import load_ensemble
modelo = load_ensemble("<seu-usuário>/macdeg_classifier", "referencia-00")   # baixa só aquela pasta
prob = modelo.predict(cv2.imread("retinografia.jpg"))
print("DMRI" if prob >= modelo.threshold else "sem DMRI", round(prob, 3))
```

## Compartilhar ou abrir ao público depois

- Dar acesso a alguém: na página do repositório, **Settings → Collaborators** (ou crie uma organização).
- Tornar público: **Settings → Change repository visibility**. Antes, confira os termos das bases
  (`docs/` e a conversa sobre o ADAM): a decisão é sua e do professor.
- Apagar uma configuração: na aba **Files**, apague a pasta, ou rode
  `HfApi().delete_folder(path_in_repo="<configuração>", repo_id="<usuário>/macdeg_classifier")`.
