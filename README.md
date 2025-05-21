# Richelieu Hardware (TSX: RCH) - Private Equity Buyout Analysis

This project simulates a private equity-style buyout analysis of Richelieu Hardware (TSX: RCH) using Monte Carlo simulation and DCF/IRR modeling.

## Project Structure

- `monte_carlo_simulation.py`: Python script for simulating revenue forecasts using Monte Carlo methods
- `financial_modeling.py`: Contains functions for financial calculations and DCF modeling
- `main.py`: Main script to run the full analysis
- `requirements.txt`: Python dependencies
- `output/`: Directory for storing simulation results and Excel models

## Setup and Usage

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```
   python main.py
   ```

3. Review the results in the `output/` directory:
   - `simulation_results.xlsx`: Raw simulation data
   - `buyout_model.xlsx`: Complete DCF and IRR model

## Model Assumptions

The model includes assumptions for:
- Revenue growth scenarios (base, bull, bear)
- EBITDA margins
- Capital expenditure requirements
- Working capital dynamics
- Tax rates
- Exit multiples
- Leverage structure

## Methodology

1. Monte Carlo simulation generates 1,000 potential revenue trajectories
2. Financial metrics are calculated for each scenario
3. Results are exported to Excel for detailed DCF and LBO modeling
4. IRR is calculated under various exit scenarios 
