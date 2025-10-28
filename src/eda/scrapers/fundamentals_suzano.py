"""
Scraper for Suzano company fundamentals.

Data sources:
- Yahoo Finance: Financial statements, key metrics
- B3/CVM: Brazilian regulatory filings
- Manual upload: Production data, guidance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from .base import BaseScraper


class SuzanoFundamentalsScraper(BaseScraper):
    """Scraper for Suzano fundamental data."""
    
    TICKER = "SUZB3.SA"
    
    def __init__(self, **kwargs):
        """Initialize Suzano fundamentals scraper."""
        super().__init__(
            name="suzano_fundamentals",
            cache_ttl_hours=24,  # Daily update sufficient
            **kwargs
        )
    
    def _fetch_data(
        self,
        start_date: str,
        end_date: str,
        data_type: str = "all",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch fundamental data.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        data_type : str
            Type: 'income_stmt', 'balance_sheet', 'cash_flow', 'all'
        
        Returns
        -------
        pd.DataFrame
            Fundamental data
        """
        print(f"[FUNDAMENTALS] Fetching {data_type} for {self.TICKER}")
        
        # Get ticker object
        ticker = yf.Ticker(self.TICKER)
        
        if data_type == "income_stmt" or data_type == "all":
            return self._fetch_income_statement(ticker)
        elif data_type == "balance_sheet":
            return self._fetch_balance_sheet(ticker)
        elif data_type == "cash_flow":
            return self._fetch_cash_flow(ticker)
        elif data_type == "info":
            return self._fetch_info(ticker)
        elif data_type == "financials":
            return self._fetch_all_financials(ticker)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _fetch_income_statement(self, ticker) -> pd.DataFrame:
        """Fetch income statement (quarterly and annual)."""
        try:
            # Quarterly
            income_q = ticker.quarterly_income_stmt
            if income_q is not None and not income_q.empty:
                income_q = income_q.T  # Transpose so dates are rows
                income_q.index.name = 'date'
                income_q['period'] = 'Q'
            
            # Annual
            income_a = ticker.income_stmt
            if income_a is not None and not income_a.empty:
                income_a = income_a.T
                income_a.index.name = 'date'
                income_a['period'] = 'A'
            
            # Combine
            if income_q is not None and income_a is not None:
                df = pd.concat([income_q, income_a]).sort_index()
            elif income_q is not None:
                df = income_q
            elif income_a is not None:
                df = income_a
            else:
                df = pd.DataFrame()
            
            return df
        
        except Exception as e:
            print(f"[FUNDAMENTALS] Error fetching income statement: {e}")
            return pd.DataFrame()
    
    def _fetch_balance_sheet(self, ticker) -> pd.DataFrame:
        """Fetch balance sheet."""
        try:
            balance_q = ticker.quarterly_balance_sheet
            if balance_q is not None and not balance_q.empty:
                balance_q = balance_q.T
                balance_q.index.name = 'date'
                balance_q['period'] = 'Q'
            
            balance_a = ticker.balance_sheet
            if balance_a is not None and not balance_a.empty:
                balance_a = balance_a.T
                balance_a.index.name = 'date'
                balance_a['period'] = 'A'
            
            if balance_q is not None and balance_a is not None:
                df = pd.concat([balance_q, balance_a]).sort_index()
            elif balance_q is not None:
                df = balance_q
            elif balance_a is not None:
                df = balance_a
            else:
                df = pd.DataFrame()
            
            return df
        
        except Exception as e:
            print(f"[FUNDAMENTALS] Error fetching balance sheet: {e}")
            return pd.DataFrame()
    
    def _fetch_cash_flow(self, ticker) -> pd.DataFrame:
        """Fetch cash flow statement."""
        try:
            cf_q = ticker.quarterly_cashflow
            if cf_q is not None and not cf_q.empty:
                cf_q = cf_q.T
                cf_q.index.name = 'date'
                cf_q['period'] = 'Q'
            
            cf_a = ticker.cashflow
            if cf_a is not None and not cf_a.empty:
                cf_a = cf_a.T
                cf_a.index.name = 'date'
                cf_a['period'] = 'A'
            
            if cf_q is not None and cf_a is not None:
                df = pd.concat([cf_q, cf_a]).sort_index()
            elif cf_q is not None:
                df = cf_q
            elif cf_a is not None:
                df = cf_a
            else:
                df = pd.DataFrame()
            
            return df
        
        except Exception as e:
            print(f"[FUNDAMENTALS] Error fetching cash flow: {e}")
            return pd.DataFrame()
    
    def _fetch_info(self, ticker) -> dict:
        """Fetch company info and key metrics."""
        try:
            info = ticker.info
            return info
        except Exception as e:
            print(f"[FUNDAMENTALS] Error fetching info: {e}")
            return {}
    
    def _fetch_all_financials(self, ticker) -> dict:
        """Fetch all financial statements."""
        return {
            'income_statement': self._fetch_income_statement(ticker),
            'balance_sheet': self._fetch_balance_sheet(ticker),
            'cash_flow': self._fetch_cash_flow(ticker),
            'info': self._fetch_info(ticker),
        }
    
    def calculate_financial_ratios(self, financials: dict) -> pd.DataFrame:
        """
        Calculate key financial ratios from statements.
        
        Parameters
        ----------
        financials : dict
            Dictionary with financial statements
        
        Returns
        -------
        pd.DataFrame
            Calculated ratios by period
        """
        income = financials.get('income_statement')
        balance = financials.get('balance_sheet')
        
        if income is None or balance is None or income.empty or balance.empty:
            print("[FUNDAMENTALS] Insufficient data for ratio calculation")
            return pd.DataFrame()
        
        # Align by date
        combined = pd.merge(
            income,
            balance,
            left_index=True,
            right_index=True,
            suffixes=('_income', '_balance')
        )
        
        ratios = pd.DataFrame(index=combined.index)
        
        # Profitability ratios
        if 'Net Income' in combined.columns and 'Total Revenue' in combined.columns:
            ratios['net_margin'] = combined['Net Income'] / combined['Total Revenue']
        
        if 'EBITDA' in combined.columns and 'Total Revenue' in combined.columns:
            ratios['ebitda_margin'] = combined['EBITDA'] / combined['Total Revenue']
        
        if 'Net Income' in combined.columns and 'Total Assets' in combined.columns:
            ratios['roa'] = combined['Net Income'] / combined['Total Assets']
        
        if 'Net Income' in combined.columns and 'Stockholders Equity' in combined.columns:
            ratios['roe'] = combined['Net Income'] / combined['Stockholders Equity']
        
        # Leverage ratios
        if 'Total Debt' in combined.columns and 'Stockholders Equity' in combined.columns:
            ratios['debt_to_equity'] = combined['Total Debt'] / combined['Stockholders Equity']
        
        if 'Total Debt' in combined.columns and 'Total Assets' in combined.columns:
            ratios['debt_ratio'] = combined['Total Debt'] / combined['Total Assets']
        
        # Efficiency ratios
        if 'Total Revenue' in combined.columns and 'Total Assets' in combined.columns:
            ratios['asset_turnover'] = combined['Total Revenue'] / combined['Total Assets']
        
        return ratios
    
    def get_key_metrics_series(self, financials: dict) -> pd.DataFrame:
        """
        Extract time series of key metrics.
        
        Parameters
        ----------
        financials : dict
            Financial statements dictionary
        
        Returns
        -------
        pd.DataFrame
            Time series of key metrics
        """
        income = financials.get('income_statement')
        balance = financials.get('balance_sheet')
        cash_flow = financials.get('cash_flow')
        
        metrics = pd.DataFrame()
        
        if income is not None and not income.empty:
            if 'Total Revenue' in income.columns:
                metrics['revenue'] = income['Total Revenue']
            if 'EBITDA' in income.columns:
                metrics['ebitda'] = income['EBITDA']
            if 'Net Income' in income.columns:
                metrics['net_income'] = income['Net Income']
        
        if balance is not None and not balance.empty:
            if 'Total Assets' in balance.columns:
                metrics['total_assets'] = balance['Total Assets']
            if 'Total Debt' in balance.columns:
                metrics['total_debt'] = balance['Total Debt']
            if 'Stockholders Equity' in balance.columns:
                metrics['equity'] = balance['Stockholders Equity']
        
        if cash_flow is not None and not cash_flow.empty:
            if 'Operating Cash Flow' in cash_flow.columns:
                metrics['operating_cf'] = cash_flow['Operating Cash Flow']
            if 'Free Cash Flow' in cash_flow.columns:
                metrics['free_cf'] = cash_flow['Free Cash Flow']
        
        return metrics.sort_index()


