import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#######################################
# 1) Callback functions to reset or set lab parameters
#######################################
def reset_parameters():
    st.session_state["portfolio_slider"] = 1000000.0
    st.session_state["mu_slider"] = 0.001
    st.session_state["sigma_slider"] = 0.02
    st.session_state["confidence_slider"] = 0.95
    st.session_state["window_slider"] = 250

def set_lab1_parameters():
    # Lab 1: Compare Parametric vs. Historical
    st.session_state["portfolio_slider"] = 1000000.0
    st.session_state["mu_slider"] = 0.0005
    st.session_state["sigma_slider"] = 0.015
    st.session_state["confidence_slider"] = 0.95
    st.session_state["window_slider"] = 250

def set_lab2_parameters():
    # Lab 2: Higher Volatility Impact
    st.session_state["portfolio_slider"] = 1000000.0
    st.session_state["mu_slider"] = 0.001
    st.session_state["sigma_slider"] = 0.05
    st.session_state["confidence_slider"] = 0.95
    st.session_state["window_slider"] = 250

def set_lab3_parameters():
    # Lab 3: Changing Confidence Level
    st.session_state["portfolio_slider"] = 1000000.0
    st.session_state["mu_slider"] = 0.001
    st.session_state["sigma_slider"] = 0.02
    st.session_state["confidence_slider"] = 0.99
    st.session_state["window_slider"] = 250

def set_lab4_parameters():
    # Lab 4: Extreme Event Simulation (Short window)
    st.session_state["portfolio_slider"] = 1000000.0
    st.session_state["mu_slider"] = 0.001
    st.session_state["sigma_slider"] = 0.03
    st.session_state["confidence_slider"] = 0.95
    st.session_state["window_slider"] = 50

def set_lab5_parameters():
    # Lab 5: Backtesting scenario with longer history
    st.session_state["portfolio_slider"] = 1000000.0
    st.session_state["mu_slider"] = 0.001
    st.session_state["sigma_slider"] = 0.02
    st.session_state["confidence_slider"] = 0.95
    st.session_state["window_slider"] = 500

#######################################
# 2) Define functions for VaR and ES calculations
#######################################
def calculate_parametric_VaR_ES(portfolio, mu, sigma, confidence):
    alpha = 1 - confidence  # tail probability (e.g., 0.05 for 95% CL)
    # VaR: - (mu + sigma * norm.ppf(alpha)) will be positive because norm.ppf(alpha) is negative
    var_parametric = portfolio * ( - (mu + sigma * norm.ppf(alpha)) )
    # ES (Expected Shortfall) for a normal distribution:
    es_parametric = portfolio * ( - (mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha) )
    return var_parametric, es_parametric

def calculate_historical_VaR_ES(portfolio, mu, sigma, confidence, window):
    alpha = (1 - confidence) * 100  # percentile (e.g., 5 for 95% CL)
    # Simulate historical returns
    hist_returns = np.random.normal(mu, sigma, int(window))
    # Calculate VaR: the negative of the alpha-th percentile
    var_hist = portfolio * (-np.percentile(hist_returns, alpha))
    # Calculate ES: average of returns below the VaR threshold (make positive)
    threshold = np.percentile(hist_returns, alpha)
    es_hist = portfolio * (-np.mean(hist_returns[hist_returns <= threshold]))
    return var_hist, es_hist, hist_returns

#######################################
# 3) Configure the Streamlit app layout and sidebar
#######################################
st.set_page_config(layout="wide")
st.title("📊 VaR & ES Playground: Measuring Downside Risk")
st.markdown("Explore how different portfolio return assumptions affect Value at Risk (VaR) and Expected Shortfall (ES) using both parametric and historical simulation methods.")

with st.sidebar:
    st.header("⚙️ Parameters")
    st.button("↺ Reset Parameters", on_click=reset_parameters)
    st.markdown("### Portfolio & Return Assumptions")
    portfolio = st.slider("Portfolio Value (€)", 10000.0, 10000000.0, 1000000.0, step=10000.0, key='portfolio_slider')
    mu = st.slider("Daily Mean Return (μ)", -0.05, 0.05, 0.001, key='mu_slider', format="%.4f")
    sigma = st.slider("Daily Volatility (σ)", 0.005, 0.1, 0.02, key='sigma_slider', format="%.3f")
    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, key='confidence_slider', format="%.2f")
    window = st.slider("Historical Window (Days)", 50, 1000, 250, key='window_slider')
    
    st.markdown("---")
    st.markdown(
    """
    **Disclaimer**  
    *This tool is for educational purposes only. The simulated data and risk measures do not represent actual portfolio performance and should not be used for investment decisions.*
    """)
    
    st.markdown("""
    <div style="margin-top: 20px;">
        <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank">
            <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
        </a>
        <br>
        <span style="font-size: 0.8em;">Luís Simões da Cunha (2025)</span>
    </div>
    """, unsafe_allow_html=True)

#######################################
# 4) Create tabs for different sections
#######################################
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Tool", 
    "📚 Theory Behind the Model", 
    "📖 Comprehensive Tutorial", 
    "🛠️ Practical Labs",
    "🧠 The Very Basics of Risk Management"
])

#######################################
# Tab 1: Interactive Tool
#######################################
with tab1:
    st.subheader("Interactive VaR & ES Calculator")
    
    # Calculate risk measures with the parametric method
    var_parametric, es_parametric = calculate_parametric_VaR_ES(portfolio, mu, sigma, confidence)
    # Calculate risk measures with the historical simulation method
    var_hist, es_hist, hist_returns = calculate_historical_VaR_ES(portfolio, mu, sigma, confidence, window)
    
    # Display the results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"### Parametric Method")
        st.markdown(f"""
        - **VaR:** €{var_parametric:,.2f}  
        - **ES:** €{es_parametric:,.2f}  
        
        *Assumes normally distributed returns.*
        """)
    with col2:
        st.success(f"### Historical Simulation")
        st.markdown(f"""
        - **VaR:** €{var_hist:,.2f}  
        - **ES:** €{es_hist:,.2f}  
        
        *Based on simulated historical returns (n={int(window)} days).*
        """)
    
    # Plot histogram of simulated historical returns with VaR and ES markers
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(hist_returns, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    # Calculate threshold for historical VaR
    alpha_threshold = np.percentile(hist_returns, (1 - confidence) * 100)
    ax.axvline(alpha_threshold, color='red', linestyle='--', linewidth=2, label='Historical VaR Threshold')
    ax.axvline(np.mean(hist_returns[hist_returns <= alpha_threshold]), color='darkorange', linestyle='--', linewidth=2, label='Historical ES')
    ax.set_title("Histogram of Simulated Daily Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

#######################################
# Tab 2: Theory Behind the Model
#######################################
with tab2:
    st.markdown("""
    ## VaR & ES: Theoretical Foundations
    
    **Value at Risk (VaR):**  
    - **Definition:** Estimates the maximum expected loss over a specified time period at a given confidence level.  
    - **Interpretation:** A 1-day 95% VaR of €50,000 means that there is a 5% chance the portfolio will lose more than €50,000 in one day.
    
    **Expected Shortfall (ES):**  
    - **Definition:** Also known as Conditional VaR (CVaR), it represents the average loss when losses exceed the VaR threshold.  
    - **Interpretation:** It gives a measure of the tail risk by averaging the worst losses.
    
    **Methods of Calculation:**  
    1. **Parametric (Variance-Covariance):**  
       - Assumes returns follow a normal (or log-normal) distribution.  
       - Simple and quick to compute using the mean and volatility of returns.
    
    2. **Non-Parametric (Historical Simulation):**  
       - Uses actual or simulated historical returns to derive the loss distribution.  
       - No assumption about the underlying distribution, making it more flexible.
    
    **Why It Matters:**  
    VaR and ES are critical tools in risk management for financial institutions, helping them understand and control potential losses.
    """)

#######################################
# Tab 3: Comprehensive Tutorial
#######################################
with tab3:
    st.markdown("""
    ## Comprehensive Tutorial on VaR & ES
    
    **Step 1: Understand the Basics**  
    - **VaR:** Determine the quantile of the loss distribution at the desired confidence level.
    - **ES:** Compute the average loss beyond the VaR cutoff.
    
    **Step 2: Parametric Method Calculation**  
    For normally distributed returns:  
    - VaR = *Portfolio Value* × *[-(μ + σ×z)]*  
      where *z* is the z‑score for the tail probability (e.g., z≈–1.645 for 5% tail).  
    - ES = *Portfolio Value* × *[-(μ – σ×(pdf(z)/α))]*  
      where *pdf(z)* is the probability density function of the standard normal at *z* and α is the tail probability.
    
    **Step 3: Historical Simulation**  
    - Collect (or simulate) a series of daily returns.  
    - Find the (1–Confidence)% percentile (e.g., 5th percentile for 95% CL) to get VaR.
    - Average all returns that fall below this percentile to get ES.
    
    **Step 4: Practice**  
    - Use the interactive sliders to adjust the portfolio value, mean return, volatility, confidence level, and historical window.
    - Observe how changes in these parameters affect the risk measures.
    
    **Example:**  
    - With a €1,000,000 portfolio, a daily mean return of 0.1%, and daily volatility of 2% at a 95% confidence level:
      - **Parametric VaR** might be around €33,000.
      - **Parametric ES** might be slightly higher, reflecting the average of the worst losses.
    
    Follow along and experiment with different scenarios to build your intuition!
    """)

#######################################
# Tab 4: Practical Labs
#######################################
with tab4:
    st.header("🔬 Practical Risk Labs")
    st.markdown("""
    In these labs, you will explore real-world scenarios using VaR and ES:
    
    - **Lab 1: Parametric vs. Historical**  
      Compare the two methods by changing assumptions and the historical window.
    
    - **Lab 2: Impact of Volatility**  
      See how increased volatility affects the risk measures.
    
    - **Lab 3: Confidence Level Experiment**  
      Observe the changes when shifting from 95% to 99% confidence.
    
    - **Lab 4: Extreme Events**  
      Simulate a shorter historical window to mimic turbulent market conditions.
    
    - **Lab 5: Backtesting VaR**  
      Increase the historical window to assess risk over a longer period.
    """)
    
    lab_choice = st.radio(
        "Select a lab to view:",
        ("Lab 1: Parametric vs. Historical",
         "Lab 2: Impact of Volatility",
         "Lab 3: Confidence Level Experiment",
         "Lab 4: Extreme Events",
         "Lab 5: Backtesting VaR"),
        index=0
    )
    
    if lab_choice == "Lab 1: Parametric vs. Historical":
        st.subheader("📊 Lab 1: Parametric vs. Historical Comparison")
        st.markdown("""
        **Objective:**  
        Compare the VaR and ES values calculated by the parametric method and historical simulation.
        
        **Steps:**  
        1. Click **Set Lab 1 Parameters** to initialize with typical values.
        2. Compare the values on the Interactive Tool tab.
        3. Adjust the historical window to see how the historical simulation changes.
        """)
        st.button("⚡ Set Lab 1 Parameters", on_click=set_lab1_parameters, key="lab1_setup")
        
    elif lab_choice == "Lab 2: Impact of Volatility":
        st.subheader("📈 Lab 2: Impact of Volatility on Risk")
        st.markdown("""
        **Objective:**  
        Observe how increasing volatility affects VaR and ES.
        
        **Steps:**  
        1. Click **Set Lab 2 Parameters** to use a higher volatility setting.
        2. Note the increase in both risk measures.
        3. Experiment by further increasing volatility.
        """)
        st.button("⚡ Set Lab 2 Parameters", on_click=set_lab2_parameters, key="lab2_setup")
        
    elif lab_choice == "Lab 3: Confidence Level Experiment":
        st.subheader("🔍 Lab 3: Effects of Changing Confidence Level")
        st.markdown("""
        **Objective:**  
        See how risk measures change when shifting from 95% to 99% confidence.
        
        **Steps:**  
        1. Click **Set Lab 3 Parameters** to set confidence at 99%.
        2. Compare the VaR and ES with the standard settings.
        3. Discuss why higher confidence leads to higher risk estimates.
        """)
        st.button("⚡ Set Lab 3 Parameters", on_click=set_lab3_parameters, key="lab3_setup")
        
    elif lab_choice == "Lab 4: Extreme Events":
        st.subheader("🌩️ Lab 4: Simulating Extreme Market Conditions")
        st.markdown("""
        **Objective:**  
        Mimic turbulent markets by reducing the historical window.
        
        **Steps:**  
        1. Click **Set Lab 4 Parameters** to use a shorter historical window.
        2. Notice the variability in historical VaR and ES.
        3. Reflect on how limited data can affect risk estimates.
        """)
        st.button("⚡ Set Lab 4 Parameters", on_click=set_lab4_parameters, key="lab4_setup")
        
    else:  # Lab 5
        st.subheader("💹 Lab 5: Backtesting VaR Over a Longer Period")
        st.markdown("""
        **Objective:**  
        Evaluate risk over a longer historical period.
        
        **Steps:**  
        1. Click **Set Lab 5 Parameters** to increase the historical window.
        2. Compare the smoother risk estimates with those from a shorter period.
        3. Discuss the importance of having ample historical data for backtesting.
        """)
        st.button("⚡ Set Lab 5 Parameters", on_click=set_lab5_parameters, key="lab5_setup")

#######################################
# Tab 5: The Very Basics of Risk Management
#######################################
with tab5:
    st.header("🧠 The Very Basics of Risk Management")
    st.markdown("""
    **What are VaR & ES?**  
    - **VaR (Value at Risk):**  
      A measure that tells you the maximum loss you might expect over a set time period at a given confidence level.
      
    - **ES (Expected Shortfall):**  
      Provides the average loss given that the loss has exceeded the VaR threshold.
      
    **Why are they important?**  
    - They help financial institutions and risk managers quantify and control downside risk.
    - VaR offers a simple risk threshold while ES gives insight into extreme losses.
      
    **Key Points to Remember:**  
    - **Parametric methods** assume a certain distribution (often normal), which simplifies calculations but may miss tail risks.
    - **Historical simulation** uses past data to compute risk, capturing real market behavior without distributional assumptions.
    
    **Risk Management in Practice:**  
    - These measures are used to set capital reserves, stress test portfolios, and guide trading decisions.
    - Understanding their limitations and assumptions is as critical as knowing how to compute them.
    """)
