# QuantSuzano - The Froelich Engine

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange.svg)

## 📖 Visão Geral

**QuantSuzano** é uma plataforma quantitativa para análise da ação Suzano (SUZB3) em relação a preços de celulose, câmbio, fatores macroeconômicos e dados climáticos. O projeto implementa **The Froelich Engine** — uma estratégia de ensemble com modelo de decisão risco-retorno que combina múltiplos modelos de machine learning para gerar sinais de trading.

### Sobre The Froelich Engine

Homenagem a John Froelich, inventor do trator. Assim como o trator substituiu a imprevisibilidade no campo por controle e eficiência, esta engine substitui “ruído de mercado” por análise direta dos fundamentos produtivos (clima), modernizando o investimento no setor.

**Estratégia:** usa modelo VECM para explorar cointegração entre SUZB3 e seus fundamentos:

* Preço da celulose
* Índice climático ponderado das regiões de cultivo

**Sinais:** Long quando a ação está subavaliada; Short quando sobreavaliada.

---

## ✨ Funcionalidades

### 📊 Pipeline de Dados

* **8 scrapers** (ações, PTAX, SELIC, clima NASA/INMET, macro IBGE, fundamentos)
* Versionamento, monitoramento, agendamento, alertas
* Validação de dados (faltantes, outliers, duplicatas, frescor)
* Cache com TTL
* Upload manual (CSV/Excel)

### 🤖 Modelagem

* Múltiplos modelos (Ridge, Lasso, RF, GBoost, XGBoost, LightGBM)
* Ensemble por votação
* Validação robusta (train/val/test, regularização, ruído)
* VECM (cointegração Johansen)
* AutoML (TPOT opcional)

### 📈 Estratégias

* Ensemble com filtro de risco-retorno
* Mean-reversion por Z-score
* Gestão de risco (stop, position sizing, volatilidade)
* Comparação com IMAT/IAGRO/IBOV

### 🎯 Risco

* Volatilidade, GARCH, regimes
* VaR/CVaR (histórico/paramétrico/MC)
* Drawdown
* Sharpe, Sortino, Calmar, Omega

### 📉 Forecasting

* ARIMA (manual e auto seleção)
* Multi-horizonte

### 📊 Visualizações

* 30+ gráficos automáticos
* Padrão de votos, análise risco-retorno
* Relatórios de qualidade de dados

---

## 🚀 Instalação

### Requisitos

* Python 3.11+

### Passos

```bash
git clone <repo-url>
cd QuantSuzano
pip install -e .
cp config.example.json config.json
```

Dependências opcionais:

```bash
pip install tpot xgboost lightgbm
```

---

## 📚 Uso

### CLI Principal

```bash
python -m eda.cli ingest
python -m eda.cli synthetic-robust
python -m eda.cli vecm
python -m eda.cli risk-analysis
python -m eda.cli forecast-arima --horizon 30
python -m eda.cli strategy-ensemble
```

Exemplo avançado:

```bash
python -m eda.cli strategy-ensemble \
    --voting-method weighted \
    --risk-reward-threshold 2.0 \
    --z-threshold 2.5
```

Produção:

```bash
python -m eda.cli pipeline-run
python -m eda.cli scheduler-start
python -m eda.cli pipeline-monitor
```

---

## 📁 Estrutura

(igual ao documento original, intacta)

---

## 🔧 Configuração

Arquivo `config.json` (sem dados pessoais):

```json
{
  "data": {
    "start_date": "2020-01-01",
    "business_frequency": "B",
    "rolling_window": 60,
    "z_threshold": 2.0
  },
  "scrapers": {
    "cache_ttl_hours": 24,
    "retry_attempts": 3,
    "rate_limit_seconds": 0.5
  },
  "sources": {
    "suzb3": { "enabled": true, "required": true, "ticker": "SUZB3.SA" },
    "pulp_prices": { "enabled": true, "manual_upload": true }
  }
}
```

---

## 📊 Resultados Recorrentes

Período teste (exemplo):
Seis meses em cenário de queda (-1.76% SUZB3)

| Estratégia        | Retorno | Sharpe | Trades             |
| ----------------- | ------- | ------ | ------------------ |
| GradientBoosting  | +0.71%  | 0.626  | 6                  |
| Ensemble + filtro | 0.00%   | —      | 0 (proteção total) |

---

## 🧪 Testes

```bash
python -m eda.cli ingest
python -m eda.cli strategy-ensemble
```

---

## 🤝 Contribuição

Fork, branch, PR.

---

## 📝 Licença

MIT

---

## 🙏 Créditos

* John Froelich (inspiração)
* Statsmodels
* Scikit-learn
* Yahoo Finance
* Banco Central do Brasil
* NASA Power

---

## 🔗 Documentação Relacionada

* `RELATORIO_COMPLETO_QUANTSUZANO.md` (arquivo local)
* `config.example.json`
* Notebooks na pasta `notebooks/`

---

## 📊 Histórico

v0.1.0 — versão inicial completa

---

## **Financiamento Quantitativo, Agora com Fundamentação Climática**
