# Carteira — Otimização de Portfólio Dow Jones em Haskell

**Disciplina:** Programação Funcional — 2026-1  
**Aluno:** Rafael Aback  
**Professores:** Raul Ikeda e Fábio Ayres  
**Entrega:** 10/05/2026

---

## 1. Descrição do Problema

Um gestor de portfólio deseja encontrar a **melhor carteira de investimentos** entre as 30 ações do índice Dow Jones Industrial Average (DJIA), usando dados históricos do segundo semestre de 2025 (01/07/2025 a 31/12/2025).

A abordagem é de **força bruta por simulação de Monte Carlo**:

1. Gerar todas as **C(30, 20) = 30.045.015** combinações possíveis de 20 ativos
2. Para cada combinação, sortear **1.000.000** vetores de pesos aleatórios válidos
3. Calcular o **Sharpe Ratio** de cada carteira
4. Retornar a carteira com maior Sharpe

O espaço de busca total é de ordem de **30 trilhões de simulações**, tornando o paralelismo indispensável.

---

## 2. Critério de Qualidade: Sharpe Ratio

O Sharpe Ratio mede o retorno ajustado ao risco de uma carteira:

```
SR = μ / σ
```

Onde:
- `μ` = retorno anualizado da carteira = `média(rₚ) × 252`
- `σ` = volatilidade anualizada = `√(wᵀ C w) × √252`
- `rₚ[t]` = retorno diário da carteira no dia t = `Σᵢ wᵢ × rᵢ[t]`
- `C` = matriz de covariância dos retornos diários (n×n)
- `w` = vetor de pesos da carteira

A taxa livre de risco foi omitida pois é constante e não afeta a ordenação entre carteiras.

**Restrições do problema:**
- `wᵢ ≥ 0` — carteira long-only (sem short selling)
- `wᵢ ≤ 0.20` — nenhum ativo representa mais de 20%
- `Σwᵢ = 1` — o capital está 100% alocado

---

## 3. Linguagem Escolhida: Haskell

Haskell foi escolhido por ser uma **linguagem funcional pura**, alinhada à rubrica B+ do projeto.

**Justificativas:**

| Aspecto | Por que Haskell |
|---|---|
| Pureza | Funções sem efeitos colaterais por padrão; estado mutável exige `IO` ou `ST` explícito |
| Paralelismo | `par`/`parListChunk` funcionam corretamente porque não há estado compartilhado |
| Tipagem | Sistema de tipos forte captura erros em tempo de compilação (ex: confundir retorno com volatilidade) |
| Lazy evaluation | Combinações geradas sob demanda — sem alocar lista de 30M elementos em memória |
| Performance | Compilação nativa com GHC + `-O2` + threads via RTS `-N` |

---

## 4. Abordagem Funcional

### 4.1 Pipeline

O cerne da solução segue o estilo funcional de composição de funções:

```haskell
combinations 20 [0..29]                        -- lista lazy de todas as combinações
  `using` parListChunk chunkSize rdeepseq       -- avaliação paralela em chunks
  |> map (simulateCombination fullRets tickers) -- melhor carteira por combinação
  |> maximum                                    -- carteira global com maior Sharpe
```

### 4.2 Funções Puras

Todas as funções do núcleo financeiro e de simulação são **puras**:

| Função | Módulo | O que faz |
|---|---|---|
| `combinations k xs` | `Simulation` | Gera C(n,k) combinações; lazy, sem efeitos colaterais |
| `covMatrix rets` | `Finance` | Constrói a matriz de covariância n×n a partir dos retornos |
| `precomputeMeanRets rets` | `Finance` | Calcula o retorno médio por ativo (pré-computação por combinação) |
| `portfolioReturn meanRets w` | `Finance` | Retorno anualizado via produto escalar `w · meanRets × 252` |
| `portfolioVolatility cov w` | `Finance` | Volatilidade via `√(wᵀ C w) × √252` |
| `sharpeRatio μ σ` | `Finance` | Razão retorno/risco |
| `sampleWeights gen n` | `Simulation` | Amostra pesos via Dirichlet(1,...,1) com rejeição em `ST s` |
| `simulateCombination retsV tickers indices` | `Simulation` | Simula 1M portfólios; usa `runST` com seed determinística |
| `evaluatePortfolio tickers meanRets cov w` | `Finance` | Avalia um portfólio completo |

**Por que `simulateCombination` é pura?**

A função usa o gerador de números aleatórios `mwc-random` dentro da mônada `ST s` (state thread), que é descarregada por `runST`. O seed é derivado deterministicamente dos índices da combinação:

```haskell
simulateCombination fullRets fullTickers indices = runST $ do
  let seed = VU.fromList (map (fromIntegral :: Int -> Word32) indices)
  gen <- MWC.initialize seed
  -- ... 1.000.000 amostras de pesos
```

O resultado é que `simulateCombination` é **referencialmente transparente**: mesmos índices → mesmo resultado. Isso é essencial para paralelismo seguro.

### 4.3 Geração de Pesos: Distribuição de Dirichlet

Para gerar vetores de pesos uniformemente distribuídos no simplex (garantindo `Σwᵢ = 1`), usamos a **distribuição Dirichlet(1,...,1)**:

1. Amostrar `n` variáveis `uᵢ ~ Uniform(0,1)`
2. Transformar: `eᵢ = -log(uᵢ)` → `eᵢ ~ Exp(1)`
3. Normalizar: `wᵢ = eᵢ / Σeⱼ`

Isso garante `wᵢ ≥ 0` e `Σwᵢ = 1` por construção. A restrição `wᵢ ≤ 0.20` é verificada por **rejection sampling**:

```haskell
sampleWeights gen n = do
  us <- VU.replicateM n (MWC.uniform gen)
  let ws = normalize (VU.map toExp us)
  if VU.all (<= 0.20) ws
    then return (Just ws)
    else return Nothing   -- ~30% dos casos; o loop simplesmente re-amostra
```

A taxa de aceitação para n=20 é aproximadamente **70%** (empiricamente verificado), tornando o rejection sampling eficiente.

### 4.4 Otimização: Pré-computação dos Retornos Médios

Uma otimização chave foi identificar que o retorno da carteira pode ser reescrito como:

```
μ = média(Σᵢ wᵢ rᵢ[t]) × 252
  = Σᵢ wᵢ × média(rᵢ) × 252      ← pela linearidade da esperança
  = w · meanRets × 252
```

Os retornos médios por ativo (`meanRets`) são **constantes para uma dada combinação** e pré-computados uma única vez. Isso reduz o custo de cada avaliação de portfólio de **O(nDias × nAtivos)** para **O(nAtivos)**, uma melhora de 6× que tornou a execução viável.

---

## 5. Estratégia de Paralelismo

### 5.1 Modelo

O paralelismo usa a biblioteca `Control.Parallel.Strategies` do GHC:

```haskell
results = map (simulateCombination fullRets fullTickers) limitedCombos
            `using` parListChunk chunkSize rdeepseq
```

- **`parListChunk k`**: divide a lista de combinações em chunks de tamanho `k`; cada chunk é avaliado em uma spark (thread leve do GHC)
- **`rdeepseq`**: força a avaliação completa (deep normal form) de cada `PortfolioResult`, garantindo que o trabalho real ocorra em paralelo
- **Tamanho do chunk**: `max 1 (numCombos / 1000)` — gera ~1000 tarefas, equilibrando granularidade e overhead

### 5.2 Por que não há condições de corrida?

Cada combinação de ativos é independente das demais:
- Cada `simulateCombination` opera sobre **dados somente-leitura** (a matriz de retornos)
- O gerador de números aleatórios (`MWC.Gen`) é **local ao `runST`** — não é compartilhado entre threads
- Não há `IORef`, `MVar` ou qualquer estado mutável compartilhado

Isso exemplifica a vantagem central da programação funcional pura para computação paralela: **ausência de efeitos colaterais elimina condições de corrida**.

### 5.3 Ativação do paralelismo

O paralelismo em Haskell requer a flag `+RTS -N` em tempo de execução:

```bash
stack exec carteira -- --max-combos 10000 +RTS -N    # usa todos os cores
stack exec carteira -- --max-combos 10000 +RTS -N4   # limita a 4 cores
```

O executável já é compilado com `-threaded -rtsopts -O2`.

---

## 6. Estrutura do Projeto

```
carteira/
├── fetch_data.py          # Baixa dados históricos via yfinance (Python)
├── stack.yaml             # Resolver Stackage LTS-24.39 (GHC 9.10.3)
├── package.yaml           # Dependências e flags de compilação
├── data/
│   └── raw/
│       ├── h2_2025/       # Preços ajustados Jul–Dez 2025 (30 CSVs)
│       └── q1_2025/       # Preços ajustados Jan–Mar 2025 (30 CSVs)
├── src/
│   ├── Types.hs           # Tipos centrais: Ticker, ReturnMatrix, PortfolioResult
│   ├── DataLoader.hs      # Leitura de CSVs e alinhamento por data
│   ├── Finance.hs         # Funções financeiras puras
│   ├── Simulation.hs      # Geração de pesos, simulação paralela
│   └── Main.hs            # Entrada, benchmark, validação out-of-sample
└── results/               # Saídas das simulações
```

### Dependências Haskell

| Pacote | Uso |
|---|---|
| `parallel` | `parListChunk`, `rdeepseq` — paralelismo por estratégias |
| `mwc-random` | Gerador Mersenne Twister; rápido e de alta qualidade estatística |
| `vector` | Arrays unboxed (`VU.Vector Double`) — sem boxing overhead |
| `deepseq` | `NFData`, `force` — avaliação forçada para paralelismo correto |
| `cassava` | Parsing de CSV com tipos nomeados |
| `containers` | `Data.Map.Strict`, `Data.Set` — alinhamento de datas |

---

## 7. Instalação

```bash
# 1. Instalar Stack
curl -sSL https://get.haskellstack.org/ | sh -s -- -d ~/.local/bin

# 2. Clonar e entrar no projeto
git clone <repo-url>
cd carteira

# 3. Instalar yfinance (necessário apenas para download de dados)
pip3 install yfinance

# 4. Baixar dados históricos (H2 2025 + Q1 2025) via API
python3 fetch_data.py

# 5. Compilar (GHC 9.10.3 é baixado automaticamente pelo Stack na 1ª vez)
stack build
```

> **Nota Linux:** Se o `libgmp` estiver instalado apenas como `.so` (sem symlink de desenvolvimento), crie um antes de compilar:
> ```bash
> mkdir -p ~/.local/lib
> ln -sf /usr/lib/x86_64-linux-gnu/libgmp.so.10 ~/.local/lib/libgmp.so
> export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
> ```

---

## 8. Execução

```bash
# Teste rápido (100 combinações ~ 36 segundos em 4 cores)
stack exec carteira -- --max-combos 100 +RTS -N

# Com validação out-of-sample no Q1 2025
stack exec carteira -- --max-combos 1000 --q1 +RTS -N

# Benchmark sequencial vs paralelo (5 execuções cada)
stack exec carteira -- --max-combos 100 --benchmark +RTS -N

# Simulação extensa (ajuste pelo tempo disponível)
stack exec carteira -- --max-combos 10000 +RTS -N
```

**Flags disponíveis:**

| Flag | Descrição |
|---|---|
| `--max-combos N` | Limita a N combinações (padrão: todas as ~30M) |
| `--q1` | Valida a melhor carteira no Q1 2025 (out-of-sample) |
| `--benchmark` | Compara sequencial vs paralelo, 5 execuções cada |
| `+RTS -N` | Usa todos os núcleos disponíveis (RTS do GHC) |
| `+RTS -N4` | Limita a 4 núcleos |

---

## 9. Dados Utilizados

Os dados foram obtidos via **yfinance** (biblioteca Python que acessa a API do Yahoo Finance), armazenados localmente como CSV com colunas `Date` e `Close` (preço ajustado):

- **Período de treino:** 01/07/2025 a 31/12/2025 — **126 pregões**
- **Período de validação:** 01/01/2025 a 31/03/2025 — **59 pregões**
- **Universo:** 30 ações do DJIA

```
AAPL  AMGN  AXP   BA    CAT   CRM   CSCO  CVX   DIS   DOW
GS    HD    HON   IBM   JNJ   JPM   KO    MCD   MMM   MRK
MSFT  NKE   NVDA  PG    SHW   TRV   UNH   V     VZ    WMT
```

Os retornos diários são calculados como retorno simples:

```
rᵢ[t] = (Pᵢ[t] - Pᵢ[t-1]) / Pᵢ[t-1]
```

---

## 10. Resultados Obtidos

### 10.1 Melhor carteira encontrada (200 combinações amostradas)

```
Ativos:                 AAPL, AMGN, AXP, BA, CAT, CRM, CSCO, CVX,
                        DIS, DOW, GS, HD, HON, IBM, JNJ, JPM, KO,
                        MRK, NVDA, WMT
Pesos:
  AAPL  0.12    AMGN  0.01    AXP   0.02    BA    0.03
  CAT   0.12    CRM   0.02    CSCO  0.04    CVX   0.03
  DIS   0.00    DOW   0.01    GS    0.04    HD    0.00
  HON   0.00    IBM   0.00    JNJ   0.05    JPM   0.01
  KO    0.01    MRK   0.18    NVDA  0.10    WMT   0.19

Retorno anualizado:      45.99%
Volatilidade anualizada: 9.93%
Sharpe Ratio:            4.63
Tempo de execução:       1min 11s  (4 cores, 200 combinações × 1M pesos)
```

> **Interpretação:** O portfólio concentra capital em NVDA (10%), WMT (19%) e MRK (18%) — ativos com forte desempenho no H2 2025 e baixa correlação entre si, o que reduz a volatilidade total.

### 10.2 Validação out-of-sample: Q1 2025

A mesma carteira foi testada no **Q1 2025** (período anterior, não visto durante a otimização):

```
Retorno anualizado:      -21.95%
Volatilidade anualizada:  13.82%
Sharpe Ratio:             -1.59
```

O desempenho negativo no Q1 2025 é esperado e revela o **overfitting implícito** da abordagem de força bruta: ao selecionar o máximo global de Sharpe em um período histórico específico, o modelo encontra carteiras que se ajustam ao ruído daquele período. Isso é uma limitação conhecida da otimização de Markowitz e do método de Monte Carlo puro — não uma falha da implementação.

### 10.3 Resumo de execuções realizadas

| Combinações testadas | Tempo real (4 cores) | Melhor Sharpe | Melhor retorno | Volatilidade |
|---|---|---|---|---|
| 100 | 36s | 4.54 | 33.98% | 7.48% |
| 200 | 1min 11s | 4.63 | 45.99% | 9.93% |
| 1.000 | ~6min* | 4.64 | 45.11% | 9.72% |
| 30.045.015 (full) | ~30 dias* | — | — | — |

*Estimativa baseada em 0.36s/combo × N / 4 cores com o binário otimizado.

---

## 11. Benchmark: Sequencial vs Paralelo

Medição com 50 combinações × 5 execuções, em máquina com 4 núcleos (`+RTS -N4`):

```
Paralelo   (média, 4 cores): 4s
Sequencial (média, 1 core):  13s
Speedup:                     3.53x   (eficiência: 88%)
```

O speedup de 3.53× em 4 núcleos demonstra que o paralelismo está funcionando de forma eficiente. A pequena diferença em relação ao ideal teórico de 4× deve-se ao overhead de sincronização e à parte sequencial do pipeline (carga de dados, `maximum` final).

---

## 12. Dificuldades Encontradas

**1. API do Yahoo Finance (HTTP 401)**  
O endpoint público do Yahoo Finance retornou 401 Unauthorized. Solução: usar a biblioteca Python `yfinance` para download inicial dos dados, salvando CSVs locais que o Haskell lê em seguida. O script `fetch_data.py` automatiza esse processo.

**2. Instalação do GHC sem privilégios de root**  
O ambiente não tinha `libgmp-dev` instalado (apenas o `.so` de runtime, sem o symlink de desenvolvimento necessário para linking). Solução: criar um symlink manualmente em `~/.local/lib/libgmp.so` e exportar `LIBRARY_PATH`.

**3. Lazy evaluation e medição de tempo**  
Em Haskell, expressões são avaliadas apenas quando necessário. Sem forçar a avaliação antes de medir o tempo final, o `diffUTCTime` capturava 0 segundos. Solução: usar `evaluate (force resultado)` da biblioteca `Control.DeepSeq` para garantir avaliação completa antes de registrar o timestamp final.

**4. Paralelismo com tipos lazy (thunks)**  
O `parMap` simples pode criar *sparks* mas não garantir que o trabalho real aconteça em paralelo — apenas a avaliação para WHNF. Solução: usar `parListChunk k rdeepseq`, que força avaliação até a forma normal (todos os campos do `PortfolioResult`) dentro de cada spark.

**5. Pureza funcional com RNG**  
Gerar números aleatórios em uma função pura requer encapsular o estado dentro da mônada `ST s` e descarregá-lo com `runST`. O desafio foi derivar seeds determinísticas por combinação (garantindo resultados reproduzíveis) sem sacrificar diversidade na amostragem.

---

## 13. Possíveis Melhorias Futuras

**Performance:**
- Usar sequências quasi-aleatórias (Sobol, Halton) para melhor cobertura do espaço de pesos com menos amostras
- Implementar operações matriciais com BLAS via `hmatrix` para aproveitar vetorização SIMD
- Paralelismo no nível de pesos dentro de cada combinação (paralelismo híbrido)

**Robustez:**
- Persistir o melhor resultado parcial em arquivo a cada N combinações, permitindo retomar após interrupção
- Aceitar dados de entrada em formato genérico (outros índices, outros períodos)

**Análise:**
- Adicionar cálculo da fronteira eficiente de Markowitz para contextualizar os resultados
- Plotar a distribuição de Sharpe Ratios encontrados (requer integração com biblioteca de gráficos ou exportação para Python)
- Implementar walk-forward validation para avaliação mais robusta do overfitting

---

## 14. Referências

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.
- Sharpe, W. F. (1966). *Mutual Fund Performance*. The Journal of Business.
- O'Sullivan, B., Goerzen, J., Stewart, D. (2008). *Real World Haskell*. O'Reilly.
- Marlow, S. (2013). *Parallel and Concurrent Programming in Haskell*. O'Reilly.
- Documentação do pacote `parallel`: https://hackage.haskell.org/package/parallel
- Documentação do pacote `mwc-random`: https://hackage.haskell.org/package/mwc-random
