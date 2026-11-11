# Passo 0: relatório de inspeção da HYAMD

> **Situação em 11/11/2026:** continua válido como descrição da HYAMD. Consequências para o projeto atual:
> a ausência de datas confirma que a progressão não pode ser estudada com ela; o controle (retinopatia
> diabética) entra hoje como "outra doença" no "sem DMRI"; a quase-duplicata entre pacientes passou a ser
> tratada pela correlação de detalhes vasculares (`macdeg/data/build.py`, ver `docs/decisoes.md`).

Só agregados, sem IDs. Unidade R = raio do campo de visão (mediana de 946 px). Reprodução: `notebooks/00_inspecao.ipynb`.

## Base
- 1560 imagens, 325 pacientes, 536 olhos pelo nome do arquivo (534 pela coluna `side`). SHA256 confere em 1560/1560; nenhum par de arquivos idênticos.
- Formato: 1278 imagens de 1934×2576 e 282 de 1934×1960 (o README cita só 1960×1934). 126 pacientes têm os dois formatos.
- Imagens por paciente: mediana 4, máximo 30. Por olho: mediana 2, máximo 18.

| Classe | Imagens | Olhos | Pacientes* | Idade mediana (por imagem) |
|---|---:|---:|---:|---:|
| 0 Controle (DR) | 1028 | 343 | 190 | 68 |
| 1 DMRI precoce | 108 | 21 | 20 | 73,5 |
| 2 Intermediária/tardia | 424 | 172 | 127 | 78 |

*12 pacientes têm rótulos diferentes entre os dois olhos e contam em duas classes.

## Metadados
- Colunas: `image_id, patient_id, sex, age, side, AMD`. Sem nulos. Há identificador de paciente e rótulo.
- **Sem data**: os PNGs também não têm metadados embutidos, e o sufixo do nome não tem sentido temporal comprovado. A idade é constante por paciente. Não dá para ordenar visitas nem distinguir visita de repetição na mesma sessão.
- O rótulo é constante por olho: não há progressão registrada.
- `image_id` difere do nome do arquivo em 1370 linhas; a junção é feita por olho e validada em `macdeg/metadata.py`.
- 3 imagens têm lado `D`/`E` no nome, e a coluna diz `L`. Nas três, o disco está na posição de olho direito.

## Centralização
| | 2576 px (n=1278) | 1960 px (n=282) |
|---|---:|---:|
| Disco, distância ao centro (mediana) | 0,63 R | 0,43 R |
| Vetor disco→mácula na média alinhada | (−0,61; +0,09) R | (−0,63; +0,09) R |
| Fóvea implícita, em relação ao centro | (+0,01; +0,05) R | (−0,21; +0,06) R |
| Fóvea por imagem a menos de 0,2 R do centro | 59% | 36% |

As linhas de fóvea usam as imagens no 75% superior do score do disco. Não há verdade de referência; falta a conferência visual.
- Lado da imagem (pelo disco) concorda com a coluna em 92,8% das imagens; 96,7% (38 discordâncias) no 75% superior do score. Não dá para separar erro do detector de erro de rótulo.
- O vetor é estável entre as classes: −0,60 a −0,61 R em x, +0,07 a +0,10 R em y.

## Campo de visão
- Raio: p1 939, mediana 946, p99 1029 px. Bandas pretas laterais nas imagens largas.
- Topo tocando a borda da imagem: 435 imagens; base: 125. Ajuste de círculo suspeito: 63 com RMS > 20 px e 45 com raio fora de 920–1010 px.

## Encaixe previsto dos ROI
Estimativa, sem medir o disco: supõe disco→fóvea = 2,5 DD, logo 1 DD ≈ 0,25 R ≈ 234 px.

| Meio lado | Meio lado (px) | Raio p/ rotação ou cantos (px) | Cabe no campo (R≈946)? |
|---:|---:|---:|---|
| 1,5 DD | 351 | 496 | Sim |
| 2,0 DD | 468 | 662 | Sim, incl. rotação |
| 3,0 DD | 702 | 993 | **Não**: nem os cantos sem rotação |

## Não conclusivo (na época)
- Quase-duplicatas entre pacientes: a comparação de miniaturas não separou olhos iguais de pacientes diferentes. Só estava descartada a duplicata byte a byte. Resolvido depois com a correlação de detalhes vasculares (≥ 0,95), que separa cópias de olhos diferentes.
