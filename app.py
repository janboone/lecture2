import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve

st.set_page_config(page_title="Lecture 2", layout="wide")

st.title("Healthcare")
st.header("Interactive: Healthcare Expenditure Modeling")

# Load data
df = pd.read_csv('./data/gdp_healthcare_nl.csv')

st.markdown("#### Healthcare Expenditure per Head in the Netherlands")
st.line_chart(
    df.set_index("Year")["CHE_per_head"],
    use_container_width=True,
    height=350
)

st.markdown("""
Adjust the sliders below to fit a linear trend to the healthcare expenditure data.  
You can also see the effect of changing the intercept and slope on the fit and the residuals.
""")

# Default values: fit first and last point
intercept_default = float(df.CHE_per_head.iloc[0])
slope_default = (df.CHE_per_head.iloc[-1] - df.CHE_per_head.iloc[0]) / (df.Year.iloc[-1] - df.Year.iloc[0])

intercept = st.slider(
    "Intercept (CHE per head at first year)", 
    min_value=intercept_default-1000, 
    max_value=intercept_default+1000, 
    value=float(np.round(intercept_default, 2)), 
    step=10.0
)
slope = st.slider(
    "Slope (change per year)", 
    min_value=-100.0, 
    max_value=200.0, 
    value=float(np.round(slope_default, 2)), 
    step=1.0
)

calc = st.button("Calculate Linear Fit", key="health_linear_fit")

def linear_trend(years, intercept, slope):
    return intercept + slope * (years - years.iloc[0])


if calc:
    years = df.Year
    che = df.CHE_per_head
    trend = linear_trend(years, intercept, slope)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(years, che, marker='o', label='Data', color='tab:orange')
    ax.plot(years, trend, label='Linear Trend (slider)', color='tab:blue')
    for x, y, yhat in zip(years, che, trend):
        ax.vlines(x, min(y, yhat), max(y, yhat), color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Healthcare Expenditure per head (Euros)')
    ax.set_title('Data, Linear Trend, and Residuals')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("""
The blue line is your linear trend (from sliders). The gray vertical lines show the difference (residual) between the data and your trend for each year.
""")

    # OLS fit and visualization
    def rss(params, years, y):
        intercept, slope = params
        yhat = intercept + slope * (years - years.iloc[0])
        return np.sum((y - yhat) ** 2)

    res = minimize(rss, x0=[che.iloc[0], 0], args=(years, che))
    opt_intercept, opt_slope = res.x
    opt_trend = linear_trend(years, opt_intercept, opt_slope)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(years, che, marker='o', label='Data', color='tab:orange')
    ax2.plot(years, opt_trend, label='OLS Linear Fit', color='tab:green')
    for x, y, yhat in zip(years, che, opt_trend):
        ax2.vlines(x, min(y, yhat), max(y, yhat), color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Healthcare Expenditure per head (Euros)')
    ax2.set_title('OLS Linear Fit and Residuals')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    st.markdown(f"""
**OLS Linear Fit (green line):**  
Intercept: {opt_intercept:.2f}  
Slope: {opt_slope:.2f}  

The green line shows the optimal linear fit (using OLS). The gray vertical lines show the residuals for each year.
""")

    # 1. Extrapolation to 2050
    future_years = np.arange(df.Year.min(), 2051)
    future_years_df = pd.Series(future_years)
    future_trend = linear_trend(future_years_df, opt_intercept, opt_slope)

    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(df.Year, df.CHE_per_head, marker='o', label='Observed Data', color='tab:orange')
    ax3.plot(future_years, future_trend, label='OLS Linear Extrapolation (to 2050)', color='tab:blue')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Healthcare Expenditure per head (Euros)')
    ax3.set_title('OLS Linear Extrapolation of Healthcare Expenditure per Head (to 2050)')
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
    st.markdown("""
**Extrapolation to 2050:**  
The blue line shows the OLS linear trend extrapolated to 2050.  
Notice that this extrapolation does not look realistic for such a long time period!
""")

    # 2. Multi-variable regression (GDP, OOP)
    def rss_multi(params, years, gdp, oop, y):
        intercept, slope_year, slope_gdp, slope_oop = params
        year_base = years - years.iloc[0]
        yhat = intercept + slope_year * year_base + slope_gdp * gdp + slope_oop * oop
        return np.sum((y - yhat) ** 2)
 
    init_params = [che.iloc[0], 0, 0, 0]
    res_multi = minimize(
        rss_multi,
        x0=init_params,
        args=(df.Year, df.GDP_per_head, df.oop, df.CHE_per_head)
    )
    opt_intercept_m, opt_slope_year, opt_slope_gdp, opt_slope_oop = res_multi.x
 
    year_base = df.Year - df.Year.iloc[0]
    che_hat_multi = (
        opt_intercept_m
        + opt_slope_year * year_base
        + opt_slope_gdp * df.GDP_per_head
        + opt_slope_oop * df.oop
    )
 
    fig4, ax4 = plt.subplots(figsize=(10,5))
    ax4.plot(df.Year, df.CHE_per_head, marker='o', label='Observed Data', color='tab:orange')
    ax4.plot(df.Year, che_hat_multi, marker='s', label='Multi-variable Prediction', color='tab:green')
    ax4.plot(df.Year, opt_trend, marker='x', label='OLS Linear Fit', color='tab:blue', linestyle='--')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Healthcare Expenditure per head (Euros)')
    ax4.set_title('Observed vs Multi-variable Model Prediction')
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)
    st.markdown("""
**Multi-variable regression:**  
The green squares show the prediction from a model using year, GDP per head, and out-of-pocket (OOP) as covariates.  
This model fits the data much better than the simple OLS linear trend (blue crosses).
""")

    # 3. Use fsolve to find required OOP to hold CHE fixed at 2014 level

    che_2014 = df.loc[df.Year == 2014, 'CHE_per_head'].values[0]
    years_proj = np.arange(2015, 2025)
    idx_proj = df.Year.isin(years_proj)
    gdp_proj = df.loc[idx_proj, 'GDP_per_head'].values
    year_base_proj = df.loc[idx_proj, 'Year'].values - df.Year.iloc[0]

    def oop_to_hold_che(oop_guess, idx):
        yhat = (
            opt_intercept_m
            + opt_slope_year * year_base_proj[idx]
            + opt_slope_gdp * gdp_proj[idx]
            + opt_slope_oop * oop_guess
        )
        return yhat - che_2014

    oop_needed = []
    for i in range(len(years_proj)):
        oop_init = df.loc[df.Year == years_proj[i], 'oop'].values[0]
        required_oop = fsolve(oop_to_hold_che, oop_init, args=(i,))
        oop_needed.append(required_oop[0])

    oop_data = df.loc[idx_proj, 'oop'].values

    fig5, ax5 = plt.subplots(figsize=(8,5))
    ax5.plot(years_proj, oop_data, marker='o', label='Actual OOP')
    ax5.plot(years_proj, oop_needed, marker='s', label='Required OOP to hold CHE fixed\n(at 2014 level)')
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Out-of-pocket (% of CHE)')
    ax5.set_title('OOP Levels Needed to Hold Healthcare Expenditure per Head Constant (2015-2024)')
    ax5.legend()
    ax5.grid(True)
    st.pyplot(fig5)
    st.markdown("""
**Required OOP to hold CHE fixed:**  
The green squares show the level of out-of-pocket payments (OOP) needed in each year to keep healthcare expenditure per head at the 2014 level, according to the multi-variable model.  
The actual OOP values are shown as orange circles.

A numerical solver (`fsolve`) is used to find the required OOP for each year.  
This demonstrates how you can use numerical methods to solve for policy variables in a model.
""")
else:
    st.info("Set the intercept and slope, then press 'Calculate Linear Fit'.")
