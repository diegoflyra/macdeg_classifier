# **Identificação da Degeneração Macular Relacionada à Idade em Retinografias via Inteligência Artificial Explicável**

**Autores**

Diego Ferreira de Andrade Lyra

Gustavo Silva Nogueira

*Versão revisada em 11/11/2026. Principais mudanças em relação à primeira versão: a base AREDS/AREDS2 não
pôde ser obtida, e nenhuma das bases disponíveis tem acompanhamento longitudinal; por isso a predição da
progressão passou a trabalho futuro. O trabalho concentra-se na identificação da DMRI em bases
heterogêneas, com um protocolo de avaliação desenhado para separar o que o modelo aprende sobre a doença
do que ele aprende sobre a origem das imagens (seção 3.5).*

## **1\. Objetivos**

### **1.1. Objetivo geral**

Desenvolver e avaliar modelos de inteligência artificial (IA) para identificar a degeneração macular relacionada à idade (DMRI) em fotografias coloridas de fundo de olho de bases heterogêneas, distinguindo-a tanto de olhos saudáveis quanto de outras doenças retinianas. O trabalho investiga diferentes arquiteturas e técnicas de IA explicável (XAI) para revelar os padrões retinianos usados pelos modelos e mede, de forma controlada, quanto do desempenho se deve à doença e quanto se deve a atalhos ligados à origem das imagens.

### **1.2. Objetivos específicos**

Desenvolver e comparar modelos de IA (redes neurais convolucionais e abordagens híbridas) para a classificação DMRI × sem DMRI, partindo de uma referência (ResNet-50) e alterando um fator por vez (arquitetura, resolução, aumento de dados, pré-processamento), com comparações pareadas que distingam melhora real de variação ao acaso.

Aplicar técnicas de XAI para mapear as regiões retinianas determinantes nas decisões e verificar sua correspondência com achados anatômicos da literatura (drusas, alterações pigmentares, atrofia, neovascularização).

Avaliar a generalização dos modelos dentro de cada base e entre bases, medindo explicitamente o quanto a base de origem e a idade do paciente, sozinhas, já explicam o rótulo.

Analisar como os modelos pontuam olhos com drusas sem diagnóstico de DMRI, para avaliar se uma categoria intermediária ("atenção") entre saudável e doente se justifica.

> * Discutir a viabilidade técnica e prática do uso dos modelos em cenários de triagem e rastreamento em larga escala.

## **2\. Justificativa**

A DMRI é uma das principais causas de perda visual em idosos. O diagnóstico precoce é vital para evitar o comprometimento severo da visão. Fotografias de fundo de olho oferecem uma alternativa acessível e escalável em relação a exames mais complexos, sendo ideais para triagem populacional.

Estudos anteriores (como o DeepSeeNet e pesquisas de predição de risco) comprovam a viabilidade de redes profundas na análise de DMRI. Contudo, persiste a necessidade de compreender as características visuais que sustentam as decisões dos modelos. A utilização de XAI (ex.: Grad-CAM, Integrated Gradients) permite explorar as regiões focadas pelas redes neurais, levantando hipóteses sobre padrões visuais que podem aprimorar o desempenho, a robustez e a confiabilidade dos classificadores.

Há também um problema de avaliação. Ao reunir bases diferentes, câmeras, populações e critérios de rotulagem passam a se correlacionar com o rótulo. Na nossa composição inicial, só saber de qual base veio a imagem já dava AUROC de 0,81, e só a idade dava 0,83. Um modelo pode assim parecer bom sem ter aprendido nada sobre a doença. Medir esses atalhos e avaliar o desempenho dentro de cada base é condição para que os resultados signifiquem algo.

## **3\. Metodologia**

### **3.1. Delineamento do estudo**

Estudo computacional quantitativo com uma tarefa principal: classificação binária de retinografias em **DMRI** ou **sem DMRI**, em que "sem DMRI" inclui olhos saudáveis e olhos com outras doenças retinianas, como na triagem real. A predição da progressão temporal, prevista na primeira versão, depende de dados longitudinais que não estão disponíveis (seção 3.9).

### **3.2. Bases de dados**

| Base | Origem | Uso | Licença / condição |
| :---- | :---- | :---- | :---- |
| HYAMD | PhysioNet | DMRI (precoce e intermediária/tardia); controles com retinopatia diabética | PhysioNet Restricted Health Data License 1.5.0 |
| BRSET | PhysioNet (Brasil) | DMRI, normais e outras doenças; só imagens de qualidade adequada | PhysioNet Credentialed Health Data License 1.5.0 |
| ODIR-5K | Desafio da Universidade de Pequim | DMRI, normais e outras doenças, por olho | sem licença declarada; uso de pesquisa |
| RFMiD e RFMiD 2.0 | Índia | DMRI, normais e outras doenças | CC BY 4.0 |
| ADAM (iChallenge-AMD) | Sun Yat-sen / Baidu | DMRI e "não DMRI" sem especificar a doença | acordo do desafio: pesquisa não comercial; publicação científica exige aprovação dos organizadores |

A AREDS/AREDS2, prevista originalmente, não pôde ser obtida. Bases do Kaggle derivadas da ODIR foram descartadas em favor da ODIR original; bases de OCT ou com imagens já aumentadas foram excluídas.

**Composição.** Após remover cópias, o conjunto tem **18.039 imagens**: 1.515 com DMRI, 11.017 normais e 5.507 com outras doenças.
- As outras doenças somam no máximo um terço do "sem DMRI" e têm cotas iguais por categoria, divididas entre as bases: retinopatia diabética, glaucoma, catarata, miopia, edema macular, membrana epirretiniana, retinopatia hipertensiva, oclusão vascular, cicatriz, nevo, hemorragia, descolamento, outras e não especificada (ADAM).
- Uma imagem com mais de uma doença conta para a categoria mais rara.
- Nas bases sem olhos normais (HYAMD e ADAM), o número de "sem DMRI" é ao menos igual ao de DMRI, para que reconhecer a base não revele o rótulo.

**Ficam fora:**
- imagens de qualidade insuficiente;
- olhos com drusas sem diagnóstico de DMRI (~2.800). As bases não informam o tamanho das drusas: as pequenas (< 63 µm) são envelhecimento normal e as médias (63–125 µm) já seriam DMRI precoce. Essas imagens são usadas só na análise da seção 3.6.

### **3.3. Pré-processamento e organização dos dados**

**Cópias.** São removidas as cópias exatas (hash do arquivo) e as quase cópias, que podem ter sido redimensionadas, recomprimidas ou recortadas.
- Quase cópias têm dois critérios: candidatas por hash perceptual após recortar até o círculo da retina, confirmadas pela correlação dos detalhes vasculares (≥ 0,95).
- Entre bases, fica a cópia de uma só base. Com rótulos conflitantes, as duas saem.
- Dentro de uma base, as cópias permanecem agrupadas.

**Imagens.**
- A retina inteira é recortada, completada com preto até formar um quadrado e guardada em 1024 px.
- O treino reduz para a resolução de cada experimento (512 px na referência), com normalização ImageNet.
- Um quadrado central cortaria as bordas da retina em bases como a BRSET.

**Divisão: validação cruzada aninhada.**
- São 5 dobras externas e, dentro de cada uma, 4 internas.
- A divisão é sempre por paciente (ou por grupo de cópias, quando a base não identifica o paciente) e estratificada por base e por tipo de olho.
- A dobra externa nunca participa de nenhuma escolha (época, limiar ou configuração) feita para ela. Assim, cada imagem do conjunto recebe uma previsão de teste, e o resultado não depende da sorte de um único conjunto de teste.

O pré-processamento por regiões de interesse (mácula e disco óptico), desenvolvido antes com a HYAMD, fica suspenso. Ele volta como experimento comparado com a referência.

### **3.4. Desenvolvimento e comparação dos modelos**

**Referência:** ResNet-50 pré-treinada na ImageNet, 512 px, sem aumento de dados.
- Perda BCE com peso das DMRI proporcional ao desbalanço.
- AdamW com aquecimento e decaimento em cosseno, precisão mista.
- Lote efetivo de 16, fixo em qualquer GPU por acumulação de gradientes.
- Parada antecipada pela AUROC de validação.

**Comparações.** Cada experimento altera **um** fator em relação à referência: arquitetura (LeNet, AlexNet, DenseNet, ConvNeXt, abordagens híbridas com SVM), resolução (1024 px), aumento de dados, pré-processamento, composição do "sem DMRI".

**Busca.** A busca de hiperparâmetros é feita por grade, com o modelo final de cada configuração definido como o **ensemble** dos 5 modelos das dobras externas.

**Execução.** Os treinos rodam em paralelo no cluster Apuana (CIn/UFPE), como job arrays do SLURM.
- Cada lote fixa o commit do código e a assinatura do conjunto e das divisões.
- Os resultados agregados ficam no GitHub e as curvas de treino no Weights & Biases.
- Os modelos finais ficam num repositório privado do Hugging Face.

### **3.5. Avaliação do desempenho**

**Métrica principal: média do F2-score dentro de cada base.** O F2 pesa a sensibilidade quatro vezes mais que a precisão, porque, na triagem, perder uma DMRI custa mais que um alarme falso. A média por base impede que reconhecer a origem das imagens infle o resultado.

**Limiar.** O limiar de decisão de cada dobra é sempre escolhido nas outras dobras, nunca nas imagens em que é medido.

**Métricas registradas junto:** F2 geral, AUROC, AUPRC, sensibilidade, especificidade em olhos normais, especificidade em outras doenças (geral e por doença), e as mesmas métricas por base.

**Intervalos e comparações.**
- Os intervalos de confiança de 95% vêm de reamostragem por paciente.
- Cada configuração é comparada com a referência **nas mesmas imagens**: o intervalo da diferença de F2 dá o veredito ("ganhou", "empatou" ou "perdeu"). Só "ganhou" permite dizer que houve melhora.

**Réguas de atalho.** Todo resultado é apresentado ao lado de duas réguas, que o modelo precisa superar:
- **origem:** AUROC usando só a fração de DMRI da base da imagem;
- **idade:** AUROC usando só a idade do paciente.

**Resultado aninhado.** Quando uma grade de configurações é comparada, a escolha da melhor é feita só com a validação interna. O resultado nas dobras externas estima o desempenho do procedimento completo de "buscar e escolher".

### **3.6. Inteligência artificial explicável e investigação de padrões visuais**

Aplicação de Grad-CAM, Grad-CAM++ e Integrated Gradients para identificar as regiões anatômicas influentes e comparar os padrões aprendidos por diferentes arquiteturas, verificando se o modelo olha para a mácula e para as lesões ou para artefatos de aquisição.

Análise da distribuição das notas do modelo para olhos normais, outras doenças, drusas sem diagnóstico de DMRI e DMRI. Se as drusas sem diagnóstico ficarem consistentemente entre os normais e as DMRI, isso fundamenta uma categoria intermediária de atenção (saudável / atenção / DMRI).

### **3.7. Testes da contribuição de características adicionais**

Realização de testes de ablação e incorporação explícita de características visuais mapeadas via XAI (por exemplo, recortes da região macular, resolução maior para drusas pequenas) para avaliar melhorias na precisão e estabilidade das respostas, sempre pelo veredito pareado contra a referência.

### **3.8. Avaliação da generalização**

Desempenho medido dentro de cada base (o F2 por base é parte da métrica principal) e comparação entre bases. Como experimento complementar, treino com todas as bases menos uma e teste na base deixada de fora, para verificar a sustentação do desempenho sob câmeras e populações não vistas no treino.

### **3.9. Análise da progressão da DMRI (trabalho futuro)**

Nenhuma base disponível tem acompanhamento longitudinal. A AREDS não pôde ser obtida, e a HYAMD não tem datas nem mudanças de rótulo por olho. A predição da progressão fica como trabalho futuro, condicionada ao acesso a dados longitudinais. Se houver acesso, a análise será estritamente prognóstica, usando apenas dados anteriores ao desfecho para evitar vazamento temporal.

### **3.10. Considerações éticas e limitações**

**Dados.** O uso é exclusivo de dados anonimizados, nos termos de cada licença.
- Nenhuma imagem ou dado de paciente sai do computador local ou do cluster da universidade.
- Os serviços externos recebem só números agregados (W&B), resultados com identificadores sequenciais próprios que não permitem localizar a imagem original (GitHub) ou pesos de modelos em repositório privado (Hugging Face).

**ADAM.** O acordo do desafio exige aprovação dos organizadores para publicação científica (inclusive relatórios técnicos e apresentações) que use seus dados. O ADAM é uma opção desligável do pipeline: qualquer resultado pode ser reproduzido sem ele.

**Limitações:**
- desbalanço de classes (~8% de DMRI);
- rótulos de critérios diferentes entre bases;
- ausência do tamanho das drusas;
- artefatos de imagem;
- a necessidade de validação clínica antes da aplicação prática.

## **4\. Cronograma**

| Etapa | Descrição das Atividades | Situação (11/11/2026) |
| :---: | :---- | :---- |
| **Etapa 1** | Revisão bibliográfica sobre DMRI, inteligência artificial aplicada à análise de imagens oftalmológicas e técnicas de inteligência artificial explicável. | em andamento |
| **Etapa 2** | Levantamento, seleção e avaliação das bases de dados disponíveis. | concluída (6 bases; AREDS indisponível) |
| **Etapa 3** | Definição da tarefa de classificação e do protocolo de avaliação. | concluída (DMRI × sem DMRI; validação cruzada aninhada; F2 médio por base) |
| **Etapa 4** | Pré-processamento das imagens e organização dos conjuntos de treinamento, validação e teste. | concluída (conjunto multibase, cópias removidas, divisões aninhadas) |
| **Etapa 5** | Implementação e treinamento dos modelos de inteligência artificial selecionados. | pipeline pronto; treino da referência no cluster Apuana |
| **Etapa 6** | Comparação do desempenho dos modelos e seleção dos mais promissores. | a fazer |
| **Etapa 7** | Aplicação de técnicas de inteligência artificial explicável para investigar os padrões visuais utilizados nas previsões. | a fazer |
| **Etapa 8** | Realização de experimentos para avaliar se características adicionais podem melhorar o desempenho e a generalização dos modelos. | a fazer |
| **Etapa 9** | Avaliação dos resultados, análise das limitações e da viabilidade de uma categoria de atenção; progressão, se houver dados longitudinais. | a fazer |
| **Etapa 10** | Redação do trabalho, discussão dos resultados e preparação da apresentação final. | a fazer |

## **5\. Referências**

> 1. PENG, Y. et al. DeepSeeNet: A Deep Learning Model for Automated Classification of Patient-Based Age-Related Macular Degeneration Severity from Color Fundus Photographs. Ophthalmology, v. 126, n. 4, p. 565–575, 2019\. DOI: 10.1016/j.ophtha.2018.11.015.  
> 2. AGE-RELATED EYE DISEASE STUDY RESEARCH GROUP. The Age-Related Eye Disease Study (AREDS): design implications. Controlled Clinical Trials, v. 20, n. 6, p. 573–600, 1999\.  
> 3. PREDICTING risk of late age-related macular degeneration using deep learning. npj Digital Medicine, 2020\. DOI: 10.1038/s41746-020-00317-z.  
> 4. NAKAYAMA, L. F. et al. Brazilian Multilabel Ophthalmological Dataset (BRSET): a large dataset of ophthalmological images and diagnostic labels. PLOS Digital Health, v. 3, n. 7, e0000454, 2024\. DOI: 10.1371/journal.pdig.0000454.  
> 5. MEISEL, A. et al. HYAMD: a dataset related to age-related macular degeneration. PhysioNet, 2025\. DOI: 10.13026/ydf1-z238. Disponível em: https\://physionet.org/.  
> 6. WONG, W. L. et al. Global prevalence of age-related macular degeneration and disease burden projection for 2020 and 2040: a systematic review and meta-analysis. The Lancet Global Health, v. 2, n. 2, p. e106–e116, 2014\. DOI: 10.1016/S2214-109X(13)70145-1.  
> 7. PREDICTING Age-related Macular Degeneration Progression with Longitudinal Fundus Images Using Deep Learning. 2023\. Disponível em: https\://pmc.ncbi.nlm.nih.gov/articles/PMC9842432/.  
> 8. I-SCREEN: estudo relacionado à implementação de uma infraestrutura de rastreamento e acompanhamento da DMRI baseada em OCT. Eye, 2026\. DOI: 10.1038/s41433-026-04487-0. Disponível em: https\://www\.nature.com/articles/s41433-026-04487-0.  
> 9. FERRIS, F. L. et al. Clinical classification of age-related macular degeneration. Ophthalmology, v. 120, n. 4, p. 844–851, 2013\. DOI: 10.1016/j.ophtha.2012.10.036.  
> 10. FANG, H. et al. ADAM Challenge: Detecting Age-Related Macular Degeneration From Fundus Images. IEEE Transactions on Medical Imaging, v. 41, n. 10, p. 2828–2847, 2022\. DOI: 10.1109/TMI.2022.3172773.  
> 11. PACHADE, S. et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research. Data, v. 6, n. 2, p. 14, 2021\. DOI: 10.3390/data6020014.  
> 12. PANCHAL, S. et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD) 2.0: A Dataset of Frequently and Rarely Identified Diseases. Data, v. 8, n. 2, p. 29, 2023\. DOI: 10.3390/data8020029.  
> 13. LI, N. et al. A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-disease Detection. In: Benchmarking, Measuring, and Optimizing (Bench 2020). Lecture Notes in Computer Science, v. 12614. Springer, 2021\. (ODIR-5K)  
