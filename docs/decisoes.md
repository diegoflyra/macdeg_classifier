# Decisões do pré-processamento e da auditoria

Só agregados, sem IDs.

## Localização
- **Disco:** busca só no lado nasal, com abertura morfológica (remove manchas menores que ~0,45 DD, como drusas) e densidade de vasos. Motivo: na inspeção visual, o detector anterior circulava drusas.
- **Fóvea:** disco + vetor médio (−0,61; +0,09) R. A mancha escura fica só como diagnóstico (`fovea_disagree`).
- **Disco incerto:** usa a posição típica do formato (`fovea_method=width_prior`) e sinaliza. Taxa na base inteira: 11,5% (8,3% controle, 16,7% precoce, 18,2% intermediária/tardia), acima do limite de 10%.
- Sem verdade de referência: a validação depende das grades.

## Desvio da especificação: DD
A especificação pede DD por imagem. Uso 0,247 R em todas, supondo 2,5 DD entre disco e fóvea, com raio de 945 px nas imagens de campo suspeito. A medida por imagem foi válida em 19% da base; misturá-la faria a escala do recorte depender da qualidade do olho. Ela fica em `dd_measured_px`. Mediana das medidas válidas: 0,268 R.

## Variantes
- **R2.5** foi acrescentada. **R3.0** só aceita G0 e G1, com `margin_black`.
- A rotação de cada ROI é decidida pelos dados: exige margem em ≥ 90% das imagens (`flag_summary.json`). Numa variante com rotação, imagem sem margem usa D4 em vez de criar preto.
- **A1** "melhor entre G1 e G2": as duas opções ficam registradas (`A1-G1`, `A1-G2`), e o mesmo vale para A2 e A3.
- **A3:** Q1 a Q4 como OneOf com p=1, depois do recorte. **Q2:** desvio de 0,005 a 0,02 na escala [0, 1].
- **F1:** ganho multiplicativo e contraste em torno da média do campo. Pixels pretos não entram nem mudam.
- A janela pode se deslocar até 0,5 × o meio lado do ROI para caber no campo (`shift_px`).
- **Excluídas:** 3 imagens com lado `D`/`E` (`excluded.csv`).

## Auditoria
- Acréscimos à especificação: `spurious_rate` (estruturas novas, pelo princípio de não criar drusas), `retention_low_contrast` e `frac_in_frame`.
- Tolerância: máx(1,5 px; 0,3 × menor diâmetro na escala final).
- Limiar: percentil 98 das respostas positivas, calibrado em R2.0 nativo.
- Controles: comparação pareada com A0, IC de 95%. Perda esperada: Q1 e Q3 em retenção e energia; Q2 em estruturas novas; Q4 em retenção e contraste.
- Q4 passou a exigir também retenção depois de um teste sintético, antes da auditoria na base inteira.

## Como ler os resultados
- Retenção de G2 acima de A0 não significa ganho: G2 avalia uma região diferente, porque os cantos do ROI saem do quadro (`roi_area_lost` ≈ 9%).
- Ruído pode inflar a retenção por coincidência com detecções falsas; leia `spurious_rate` junto.
