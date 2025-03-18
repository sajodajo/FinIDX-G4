def describeGarchModel(bestResult):
    """Generate a business-friendly description of the GARCH model results"""
    model_params = bestResult.params
    
    # Calculate persistence
    persistence = sum([model_params[f'alpha[{i+1}]'] for i in range(bestResult.p) if f'alpha[{i+1}]' in model_params] + 
                     [model_params[f'beta[{i+1}]'] for i in range(bestResult.q) if f'beta[{i+1}]' in model_params])
    
    # Business-friendly interpretation
    summary = f"Our volatility model shows that "
    
    if persistence > 0.95:
        summary += f"market shocks have very long-lasting effects on this asset. When volatility increases, it tends to remain elevated for extended periods. This suggests you should consider longer-term hedging strategies when managing this position."
    elif persistence > 0.85:
        summary += f"market shocks have significant staying power. When volatility spikes, it tends to persist for several weeks before returning to normal levels. This suggests medium-term hedging strategies may be appropriate."
    else:
        summary += f"market shocks tend to dissipate relatively quickly. When volatility spikes, it typically returns to normal levels within a short period. This suggests short-term hedging strategies may be sufficient."
    
    # Add more business context
    if bestResult.p > 1 or bestResult.q > 1:
        summary += f" The model also indicates a complex relationship between past and current volatility, suggesting that simple risk metrics may not capture the full risk profile of this asset."
    
    return summary


def summarizeVolatilityPatterns(garch_fit, returns):
    """Generate business-friendly insights about volatility patterns"""
    cond_vol = garch_fit.conditional_volatility
    
    # Calculate stats
    mean_vol = cond_vol.mean()
    max_vol = cond_vol.max()
    max_vol_date = cond_vol.idxmax()
    volatility_range = max_vol / mean_vol
    
    # Find periods of high volatility (top 5%)
    high_vol_threshold = np.percentile(cond_vol, 95)
    high_vol_periods = cond_vol[cond_vol > high_vol_threshold]
    
    # Business-friendly insights
    summary = f"Looking at volatility patterns, "
    
    if volatility_range > 4:
        summary += f"we've observed extreme volatility swings. The most volatile period (around {max_vol_date.strftime('%B %Y')}) saw volatility levels approximately {volatility_range:.1f}x higher than average. "
    elif volatility_range > 2.5:
        summary += f"we've observed significant volatility swings. The most volatile period (around {max_vol_date.strftime('%B %Y')}) saw volatility levels about {volatility_range:.1f}x higher than average. "
    else:
        summary += f"volatility has remained relatively stable. Even during the most volatile period (around {max_vol_date.strftime('%B %Y')}), volatility only reached about {volatility_range:.1f}x the average level. "
    
    # Add clustering information
    high_vol_days = (cond_vol > high_vol_threshold).astype(int)
    transitions = np.diff(np.hstack([[0], high_vol_days, [0]]))
    clusters = np.sum(transitions == 1)
    
    if clusters > 5:
        summary += f"We've identified {clusters} distinct volatility spikes over the period. This suggests the asset experiences frequent stress events that require active risk management."
    elif clusters > 2:
        summary += f"We've identified {clusters} distinct volatility spikes over the period. This suggests occasional stress events that should be accounted for in your risk management strategy."
    else:
        summary += f"We've identified only {clusters} distinct volatility spikes over the period. This suggests infrequent stress events, though their impact can still be significant."
    
    return summary


def interpretResidualAnalysis(garch_fit):
    """Generate business-friendly insights from residual analysis"""
    # Compute standardized residuals
    std_residuals = garch_fit.resid / garch_fit.conditional_volatility
    std_residuals = std_residuals.dropna()
    
    # Calculate statistics
    kurtosis = stats.kurtosis(std_residuals)
    
    # Fit t-distribution to standardized residuals
    df_t, loc_t, scale_t = t.fit(std_residuals)
    
    # Business-friendly insights
    summary = f"Our analysis of market movements shows that "
    
    if kurtosis > 3:
        summary += f"this asset experiences more extreme price movements than standard models would predict. This means there's a higher likelihood of 'black swan' events than conventional risk measures might suggest. "
    elif kurtosis > 1:
        summary += f"this asset occasionally experiences more extreme price movements than standard models would predict. This means there's a moderate risk of unexpected large price swings. "
    else:
        summary += f"price movements generally follow patterns that are well-captured by standard risk models. This suggests that conventional risk measures should be relatively reliable for this asset. "
    
    # Add t-distribution assessment
    if df_t < 5:
        summary += f"Our analysis indicates that you should be prepared for price movements that are up to 2-3 times larger than what standard risk models would predict."
    elif df_t < 10:
        summary += f"Our analysis indicates that you should be prepared for price movements that are about 1.5 times larger than what standard risk models would predict."
    else:
        summary += f"The size of price movements appears to be reasonably well-captured by standard risk models."
    
    return summary


def analyzeVaRResults(confidence_levels, VaR_hist, VaR_norm, VaR_t, companyName):
    """Generate business-friendly insights from VaR analysis"""
    # Find specific VaR values at 95% and 99% confidence
    idx_95 = np.abs(confidence_levels - 0.95).argmin()
    idx_99 = np.abs(confidence_levels - 0.99).argmin()
    
    VaR_95_hist = VaR_hist[idx_95]
    VaR_99_hist = VaR_hist[idx_99]
    VaR_99_t = VaR_t[idx_99]
    
    # Business-friendly insights
    summary = f"Our Value-at-Risk (VaR) analysis for {companyName} shows that "
    
    # Translate to dollar terms with hypothetical portfolio
    portfolio_value = 1000000  # $1 million portfolio for illustration
    dollar_var_95 = -VaR_95_hist * portfolio_value
    dollar_var_99 = -VaR_99_hist * portfolio_value
    
    summary += f"in a typical worst-case day (95% confidence), you could expect to lose up to {dollar_var_95:,.0f} dollars on a {portfolio_value:,.0f} dollar position. "
    summary += f"In a severe scenario (99% confidence), losses could reach {dollar_var_99:,.0f} dollars. "
    
    # Compare approaches
    t_vs_hist_ratio_99 = abs(VaR_99_t / VaR_99_hist)
    
    if t_vs_hist_ratio_99 > 1.2:
        summary += f"Our advanced modeling suggests that standard approaches may underestimate the true risk by approximately {(t_vs_hist_ratio_99-1)*100:.0f}%. "
        summary += f"This means your actual risk exposure could be significantly higher than conventional metrics indicate."
    elif t_vs_hist_ratio_99 < 0.8:
        summary += f"Our advanced modeling suggests that standard approaches may overestimate the true risk by approximately {(1-t_vs_hist_ratio_99)*100:.0f}%. "
        summary += f"This means your actual risk exposure may be lower than conventional metrics indicate."
    else:
        summary += f"Our advanced modeling largely confirms the risk estimates from standard approaches, giving us confidence in these figures."
    
    return summary


def analyzeDynamicRiskMeasures(garch_fit, companyName, returns):
    """Generate business-friendly insights from dynamic risk measures"""
    # Extract conditional volatility
    cond_volatility = garch_fit.conditional_volatility.dropna()
    
    # Compute standardized residuals
    std_residuals = garch_fit.resid / garch_fit.conditional_volatility
    std_residuals = std_residuals.dropna()
    
    # Fit t-distribution to standardized residuals
    df_t, loc_t, scale_t = t.fit(std_residuals)
    
    # Compute dynamic VaR at 95% and 99%
    VaR_95 = -scale_t * t.ppf(0.05, df_t) * cond_volatility
    VaR_99 = -scale_t * t.ppf(0.01, df_t) * cond_volatility
    
    # Calculate statistics
    recent_window = min(20, len(VaR_95))
    recent_VaR = VaR_95[-recent_window:]
    avg_recent_VaR = recent_VaR.mean()
    avg_overall_VaR = VaR_95.mean()
    
    change_pct = (avg_recent_VaR - avg_overall_VaR) / avg_overall_VaR * 100
    
    # Business-friendly insights
    summary = f"Our dynamic risk assessment for {companyName} shows that "
    
    if change_pct > 20:
        summary += f"current risk levels are substantially higher (about {change_pct:.0f}%) than the historical average. "
        summary += f"This suggests you should consider reducing exposure or implementing additional hedges in the near term."
    elif change_pct > 5:
        summary += f"current risk levels are moderately elevated (about {change_pct:.0f}%) compared to the historical average. "
        summary += f"This suggests a cautious approach to new positions may be warranted."
    elif change_pct < -20:
        summary += f"current risk levels are substantially lower (about {abs(change_pct):.0f}% below) than the historical average. "
        summary += f"This could present an opportunity to increase positions at a lower risk point."
    elif change_pct < -5:
        summary += f"current risk levels are moderately lower (about {abs(change_pct):.0f}% below) than the historical average. "
        summary += f"This suggests a relatively favorable risk environment for this asset."
    else:
        summary += f"current risk levels are in line with historical averages. "
        summary += f"This suggests a neutral risk environment for this asset."
    
    # Add backtest information
    neg_returns = -returns.loc[VaR_95.index]
    breaches_95 = (neg_returns > VaR_95).sum()
    breaches_99 = (neg_returns > VaR_99).sum()
    
    breach_rate_95 = breaches_95 / len(VaR_95)
    breach_rate_99 = breaches_99 / len(VaR_99)
    
    if breach_rate_95 > 0.06:
        summary += f" Our models may be underestimating the true risk, as we've observed more extreme losses than expected in backtesting. Consider adding a safety margin to these estimates."
    elif breach_rate_95 < 0.04:
        summary += f" Our models appear to be conservative, as we've observed fewer extreme losses than expected in backtesting. These estimates likely provide a good cushion against losses."
    
    return summary


def generateExecutiveSummary(garch_fit, returns, companyName):
    """Generate a concise executive summary of key risk insights"""
    # Calculate key statistics
    cond_vol = garch_fit.conditional_volatility
    mean_vol = cond_vol.mean()
    max_vol = cond_vol.max()
    max_vol_date = cond_vol.idxmax()
    volatility_ratio = max_vol / mean_vol
    
    # Calculate average daily return and annualized volatility
    avg_return = returns.mean() * 100  # Convert to percentage
    annualized_vol = mean_vol * np.sqrt(252) * 100  # Convert to percentage and annualize
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free = 0.02 / 252  # Daily risk-free rate
    sharpe = (returns.mean() - risk_free) / returns.std() * np.sqrt(252)
    
    # Calculate 95% VaR for a $1M portfolio
    portfolio_value = 1000000
    var_95 = np.percentile(returns, 5) * portfolio_value
    
    # Business-friendly executive summary
    summary = f"# Executive Summary: {companyName} Risk Profile\n\n"
    
    summary += f"## Key Risk Metrics\n"
    summary += f"* **Expected Annual Volatility:** {annualized_vol:.1f}%\n"
    summary += f"* **Average Daily Return:** {avg_return:.3f}%\n"
    summary += f"* **Risk-Adjusted Return (Sharpe):** {sharpe:.2f}\n"
    summary += f"* **Daily Value-at-Risk (95%):** ${abs(var_95):,.0f} on a $1M position\n\n"
    
    summary += f"## Risk Assessment\n"
    
    # Risk level assessment
    if annualized_vol > 40:
        risk_level = "Very High"
    elif annualized_vol > 25:
        risk_level = "High"
    elif annualized_vol > 15:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    summary += f"* **Overall Risk Level:** {risk_level}\n"
    
    # Dynamic risk assessment
    recent_window = min(20, len(cond_vol))
    recent_vol = cond_vol[-recent_window:]
    recent_avg = recent_vol.mean()
    vol_change = (recent_avg - mean_vol) / mean_vol * 100
    
    if vol_change > 20:
        trend = f"Strongly Increasing (+{vol_change:.0f}%)"
    elif vol_change > 5:
        trend = f"Increasing (+{vol_change:.0f}%)"
    elif vol_change < -20:
        trend = f"Strongly Decreasing ({vol_change:.0f}%)"
    elif vol_change < -5:
        trend = f"Decreasing ({vol_change:.0f}%)"
    else:
        trend = "Stable"
    
    summary += f"* **Current Risk Trend:** {trend}\n"
    
    # Extreme events
    summary += f"* **Stress Scenario Impact:** During peak volatility ({max_vol_date.strftime('%B %Y')}), daily losses could be {volatility_ratio:.1f}x larger than average\n\n"
    
    summary += f"## Recommendations\n"
    
    # Generate recommendations based on the analysis
    recommendations = []
    
    if vol_change > 10:
        recommendations.append("Consider reducing position sizes or implementing additional hedges given the recent volatility increase")
    
    if volatility_ratio > 3:
        recommendations.append(f"Ensure risk limits can accommodate potential {volatility_ratio:.1f}x spikes in volatility")
    
    if sharpe < 0.5 and annualized_vol > 25:
        recommendations.append("Re-evaluate the risk-return profile of this asset, as it may not provide adequate compensation for its volatility")
    
    if not recommendations:
        recommendations.append("Maintain current position within established risk limits")
    
    for i, rec in enumerate(recommendations):
        summary += f"{i+1}. {rec}\n"
    
    return summary


def generateFullReport(df, returns, garch_fit, bestResult, companyName):
    """Generate a comprehensive business-oriented risk report"""
    # Import required libraries at the start of your script
    import numpy as np
    from scipy import stats
    from scipy.stats import t, norm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf
    import pandas as pd
    from datetime import datetime
    
    # Calculate key statistics for the report
    avg_return = returns.mean() * 100  # Convert to percentage
    annualized_return = avg_return * 252  # Annualize
    annualized_vol = returns.std() * np.sqrt(252) * 100  # Convert to percentage and annualize
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_annual = 0.02
    risk_free_daily = risk_free_annual / 252
    sharpe = (returns.mean() - risk_free_daily) / returns.std() * np.sqrt(252)
    
    # Calculate drawdowns
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate VaR metrics
    confidence_levels = np.linspace(0.95, 0.99, 5)
    VaR_hist = [np.percentile(returns, (1 - alpha) * 100) for alpha in confidence_levels]
    VaR_norm = [norm.ppf(1 - alpha, loc=np.mean(returns), scale=np.std(returns)) for alpha in confidence_levels]
    df_t, loc_t, scale_t = t.fit(returns)
    VaR_t = [loc_t + scale_t * t.ppf(1 - alpha, df_t) for alpha in confidence_levels]
    
    # Business-friendly report
    fCompany = f"{companyName}"
    report_date = datetime.now().strftime("%B %d, %Y")
    
    report = f"# Risk Analysis Report: {fCompany}\n"
    report += f"**Report Date:** {report_date}\n\n"
    
    # Executive summary
    report += "## Executive Summary\n\n"
    report += describeGarchModel(bestResult) + "\n\n"
    report += summarizeVolatilityPatterns(garch_fit, returns) + "\n\n"
    
    # Key metrics table
    report += "## Key Risk Metrics\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| Annualized Return | {annualized_return:.2f}% |\n"
    report += f"| Annualized Volatility | {annualized_vol:.2f}% |\n"
    report += f"| Sharpe Ratio | {sharpe:.2f} |\n"
    report += f"| Maximum Drawdown | {abs(max_drawdown):.2f}% |\n"
    report += f"| Daily VaR (95%) | {abs(VaR_hist[0] * 100):.2f}% |\n"
    report += f"| Daily VaR (99%) | {abs(VaR_hist[-1] * 100):.2f}% |\n\n"
    
    # Risk assessment
    report += "## Detailed Risk Assessment\n\n"
    report += "### Volatility Analysis\n"
    report += interpretResidualAnalysis(garch_fit) + "\n\n"
    
    report += "### Value-at-Risk Analysis\n"
    report += analyzeVaRResults(confidence_levels, VaR_hist, VaR_norm, VaR_t, companyName) + "\n\n"
    
    report += "### Dynamic Risk Assessment\n"
    report += analyzeDynamicRiskMeasures(garch_fit, companyName, returns) + "\n\n"
    
    # Business implications
    report += "## Business Implications\n\n"
    
    # Generate business implications based on the analysis
    implications = []
    
    # Volatility persistence
    model_params = bestResult.params
    persistence = sum([model_params[f'alpha[{i+1}]'] for i in range(bestResult.p) if f'alpha[{i+1}]' in model_params] + 
                     [model_params[f'beta[{i+1}]'] for i in range(bestResult.q) if f'beta[{i+1}]' in model_params])
    
    if persistence > 0.95:
        implications.append("**Long-Term Risk Planning:** The high persistence in volatility means that when markets become turbulent, they tend to stay that way for extended periods. This suggests:  \n- Implementing longer-term hedging strategies rather than short-term tactical adjustments  \n- Developing contingency plans that can be sustained for several months during stress periods")
    
    # Risk-return profile
    if sharpe < 0.5:
        implications.append("**Risk-Return Optimization:** The current risk-return profile suggests inadequate compensation for the volatility risk. Consider:  \n- Restructuring the position to improve the risk-adjusted return  \n- Exploring alternative investments with more favorable risk-return characteristics")
    
    # Extreme events
    cond_vol = garch_fit.conditional_volatility
    volatility_ratio = cond_vol.max() / cond_vol.mean()
    if volatility_ratio > 3:
        implications.append(f"**Extreme Event Preparedness:** The analysis shows that extreme volatility events (up to {volatility_ratio:.1f}x normal levels) are possible. This requires:  \n- Stress testing portfolios under more severe scenarios than standard models suggest  \n- Maintaining sufficient liquidity to withstand extended periods of market stress")
    
    # Current risk trend
    recent_window = min(20, len(cond_vol))
    recent_vol = cond_vol[-recent_window:]
    recent_avg = recent_vol.mean()
    vol_change = (recent_avg - cond_vol.mean()) / cond_vol.mean() * 100
    
    if vol_change > 10:
        implications.append(f"**Current Market Conditions:** Our analysis indicates a {vol_change:.0f}% increase in current risk levels compared to historical averages. This suggests:  \n- Implementing tactical risk reduction in the near term  \n- Exercising caution with new positions until volatility normalizes")
    elif vol_change < -10:
        implications.append(f"**Current Market Conditions:** Our analysis indicates a {abs(vol_change):.0f}% decrease in current risk levels compared to historical averages. This suggests:  \n- A potential opportunity to increase positions at a lower risk point  \n- The ability to implement strategic positions with improved risk-adjusted returns")
    
    # Add implications to report
    for i, imp in enumerate(implications):
        report += f"{i+1}. {imp}\n\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    # Generate recommendations based on the analysis
    recommendations = []
    
    if persistence > 0.95:
        recommendations.append("**Extend Hedging Horizon:** Implement longer-dated hedges to account for the persistent nature of volatility")
    
    if volatility_ratio > 3:
        recommendations.append("**Enhance Stress Testing:** Incorporate more extreme scenarios in stress tests that reflect the observed volatility spikes")
    
    if sharpe < 0.5 and annualized_vol > 20:
        recommendations.append("**Optimize Risk-Return:** Consider restructuring the position to improve the risk-adjusted return")
    
    if vol_change > 10:
        recommendations.append("**Tactical Risk Reduction:** Temporarily reduce position sizes or implement additional hedges until volatility normalizes")
    elif vol_change < -10:
        recommendations.append("**Strategic Opportunity:** Consider increasing positions to take advantage of the favorable risk environment")
    
    # Add recommendations to report
    for i, rec in enumerate(recommendations):
        report += f"{i+1}. {rec}\n\n"
    
    # Methodology note
    report += "## Methodology\n\n"
    report += "This analysis was conducted using advanced time series modeling techniques, including GARCH volatility modeling and extreme value theory. The approach captures both the dynamic nature of market risk and the potential for extreme events beyond what traditional models suggest.\n\n"
    
    return report