# EDA Suzano - Análise Exploratória de Dados

Projeto de EDA reprodutível relacionando câmbio (PTAX), juros (SELIC), clima, preço da celulose e preço de SUZB3.

## Objetivo

Construir um pipeline automatizado que:
- Coleta dados de mercado (SUZB3, PTAX, SELIC) via APIs
- Realiza feature engineering (retornos log, defasagens climáticas)
- Constrói índice sintético via regressão OLS
- Identifica sinais de trading baseados em z-scores
- Executa modelagem VECM para cointegração

## Estrutura do Projeto

```
/eda-suzano
├── data/
│   ├── raw/          # CSVs de fallback (pulp, climate, credit)
│   ├── interim/      # Dados processados em parquet
│   └── out/          # Resultados, sumários, gráficos
├── notebooks/
│   └── EDA.ipynb     # Análise exploratória detalhada
├── src/eda/
│   ├── config.py     # Configurações e paths
│   ├── loaders.py    # Carregadores de dados
│   ├── features.py   # Engenharia de features
│   ├── synthetic.py  # Índice sintético e z-scores
│   ├── models.py     # VECM e Johansen
│   ├── plots.py      # Visualizações
│   └── cli.py        # Interface CLI
├── pyproject.toml
├── Makefile
└── README.md
```

## Fontes de Dados

### Dados Reais (APIs)
- **SUZB3**: Yahoo Finance (yfinance) - Ações Suzano B3
- **PTAX**: Banco Central do Brasil SGS API (série 1) - Câmbio USD/BRL
- **SELIC**: Banco Central do Brasil SGS API (série 432) - Taxa de juros

### Dados Placeholder (CSV)
- **Pulp USD**: Preços da celulose BHKP/NBSK em USD/ton (`data/raw/pulp_usd.csv`)
- **Climate**: Precipitação (mm) e NDVI (`data/raw/climate.csv`)
- **Credit**: Índice de crédito mensal (`data/raw/credit.csv`)

## Setup

### 1. Criar ambiente virtual e instalar dependências

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
make setup
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
make setup
```

### 2. Executar pipeline completo

```bash
make all
```

Ou executar etapas individualmente:

```bash
make ingest      # Coleta e merge de dados
make synthetic   # Índice sintético e z-scores
make vecm        # Modelagem VECM
make report      # Gera gráficos
```

### 3. Explorar notebook

```bash
make notebook
```

Abre Jupyter Lab com o notebook `notebooks/EDA.ipynb`.

## Interface CLI

```bash
# Ingerir dados
python -m eda.cli ingest

# Calcular índice sintético
python -m eda.cli synthetic

# Modelagem VECM
python -m eda.cli vecm

# Gerar relatório com gráficos
python -m eda.cli report

# Executar tudo
python -m eda.cli all
```

## Outputs Gerados

Após executar `make all`, os seguintes arquivos são gerados:

- `data/interim/merged.parquet` - Dados mergeados e preprocessados
- `data/out/ols_summary.txt` - Sumário da regressão OLS
- `data/out/vecm_summary.txt` - Sumário do modelo VECM
- `data/out/synthetic.parquet` - Índice sintético e z-scores
- `data/out/signals.parquet` - Sinais de trading (|z| > 2)
- `data/out/plots/*.png` - Visualizações (níveis, retornos, correlações, etc.)

## Metodologia

### Feature Engineering
- **Frequência**: Dias úteis (B) com forward fill para dados mensais
- **Retornos**: Log-retornos `r_t = log(P_t) - log(P_{t-1})`
- **Pulp BRL**: `pulp_brl = pulp_usd * ptax`
- **Defasagens climáticas**: 15, 30 e 60 dias para precip_mm e NDVI

### Índice Sintético
1. Regressão OLS: `suzb_r ~ const + ptax_r + selic_r + pulp_brl_r + credit_r + climate_lags`
2. Índice sintético: `synthetic_index = X @ betas`
3. Spread: `spread = suzb_r - synthetic_index`
4. Z-score rolling (janela 60): `z = (spread - mean_60) / std_60`
5. Sinais: `|z| > 2`

### VECM
- Sistema bivariado: `[pulp_brl, suzb]`
- Teste de cointegração de Johansen
- VECM com `k_ar_diff=2`, `coint_rank=1`

## Próximos Passos

- [ ] Substituir CSV placeholders por APIs reais:
  - FOEX para preços de celulose
  - INMET para dados climáticos
  - BCB para crédito (séries SGS relevantes)
- [ ] Implementar backtesting dos sinais
- [ ] Adicionar testes unitários
- [ ] Otimizar seleção de ordem do VECM via AIC/BIC
- [ ] Winsorization de outliers

## Licença

MIT License

## Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

