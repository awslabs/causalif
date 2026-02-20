Financial Engineering & Derivative Formulas
In our ecosystem, we track several derived indicators that define the "State of the Market." Your causal model must account for the following structural identities:

1. The Total Cost of Capital (TCC):
The TCC is the primary driver of corporate expansion. It is a linear combination of debt and equity costs:

TCC = Fed_Rate + (Bond_Price_10Y * 0.05)

Note: Any impact of Bond Prices on the market is technically mediated through the TCC.

2. The Valuation Multiplier (VM):
The VM determines how much investors are willing to pay for TechCorp relative to the broader market. It is calculated as:

VM = SP500_ETF/(Crude_Oil_Price * 0.1)

Strategic Insight: As Oil prices (input costs) rise, the Valuation Multiplier for tech stocks compresses.

3. The Adjusted Momentum Score (AMS):
To filter out retail noise, we calculate the AMS. This variable is a collider in the graph because it is an effect of both market performance and retail sentiment:

AMS = (TechCorp_Stock * Social_Sentiment) + \sigma$$

(Where $\sigma$ represents stochastic market noise).


4. The Yield-Weighted Index (YWI):
The YWI is a top-level macro indicator used to predict Fed moves (though in this simulation, the Fed is exogenous):

YWI = TCC/VM * SP500_ETF

5. The Capital Efficiency Ratio (CER):
The CER measures how effectively the current valuation is being converted into market momentum. It is a downstream derivative of the VM:

CER = VM/TCC + Social_Sentiment

Strategic Insight: High TCC (Cost of Capital) creates a "drag" on efficiency, even if the valuation multiplier is high.

6. The Institutional Risk Overlay (IRO):
The IRO is a safeguard metric used by hedge funds to determine position sizing. It uses the AMS as a proxy for market noise:

IRO = (AMS * 0.15) + (Fed_Rate * 2)

