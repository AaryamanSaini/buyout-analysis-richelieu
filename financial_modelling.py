import numpy as np
import pandas as pd
import os
from scipy.optimize import newton

class BuyoutModel:
    def __init__(self, 
                 simulation_results,
                 enterprise_value=2100,  # Enterprise value in CAD millions
                 debt_percentage=0.6,    # Percentage of purchase price financed by debt
                 ebitda_margin_base=0.12,
                 ebitda_margin_improvement=0.015,  # Annual improvement of EBITDA margin
                 exit_multiple_base=11.0,
                 exit_multiple_bull=13.0,
                 exit_multiple_bear=9.0,
                 tax_rate=0.27,
                 capex_percent_revenue=0.03,
                 nwc_percent_revenue=0.15,
                 nwc_change_percent_revenue_growth=0.6,
                 debt_interest_rate=0.065,
                 debt_repayment_percent_fcf=0.6,
                 minimum_cash=50):  # Minimum cash in CAD millions
        """
        Initialize the Buyout Model for LBO and DCF calculations.
        
        Args:
            simulation_results: DataFrame with Monte Carlo simulation results
            enterprise_value: Current enterprise value in CAD millions
            debt_percentage: Percentage of purchase price financed by debt
            ebitda_margin_base: Base EBITDA margin
            ebitda_margin_improvement: Annual improvement in EBITDA margin
            exit_multiple_base: Exit EV/EBITDA multiple in base case
            exit_multiple_bull: Exit EV/EBITDA multiple in bull case
            exit_multiple_bear: Exit EV/EBITDA multiple in bear case
            tax_rate: Corporate tax rate
            capex_percent_revenue: CapEx as a percentage of revenue
            nwc_percent_revenue: Net working capital as a percentage of revenue
            nwc_change_percent_revenue_growth: NWC change as a percentage of revenue growth
            debt_interest_rate: Interest rate on debt
            debt_repayment_percent_fcf: Debt repayment as a percentage of FCF
            minimum_cash: Minimum cash balance maintained
        """
        self.simulation_results = simulation_results
        self.enterprise_value = enterprise_value
        self.debt_percentage = debt_percentage
        self.equity_percentage = 1 - debt_percentage
        self.ebitda_margin_base = ebitda_margin_base
        self.ebitda_margin_improvement = ebitda_margin_improvement
        self.exit_multiple_base = exit_multiple_base
        self.exit_multiple_bull = exit_multiple_bull
        self.exit_multiple_bear = exit_multiple_bear
        self.tax_rate = tax_rate
        self.capex_percent_revenue = capex_percent_revenue
        self.nwc_percent_revenue = nwc_percent_revenue
        self.nwc_change_percent_revenue_growth = nwc_change_percent_revenue_growth
        self.debt_interest_rate = debt_interest_rate
        self.debt_repayment_percent_fcf = debt_repayment_percent_fcf
        self.minimum_cash = minimum_cash
        
        # Calculate initial conditions
        self.equity_contribution = self.enterprise_value * self.equity_percentage
        self.initial_debt = self.enterprise_value * self.debt_percentage
        
        # Initialize results
        self.model_results = {}
        
    def calculate_ebitda(self, revenue, year):
        """Calculate EBITDA based on revenue and improving margins over time."""
        margin = min(self.ebitda_margin_base + year * self.ebitda_margin_improvement, 0.20)
        return revenue * margin
    
    def calculate_capex(self, revenue):
        """Calculate capital expenditures as a percentage of revenue."""
        return revenue * self.capex_percent_revenue
    
    def calculate_nwc_change(self, current_revenue, previous_revenue):
        """Calculate change in net working capital based on revenue growth."""
        revenue_growth = current_revenue - previous_revenue
        return revenue_growth * self.nwc_change_percent_revenue_growth
    
    def calculate_exit_value(self, final_ebitda, scenario):
        """Calculate exit enterprise value based on EBITDA and scenario."""
        if scenario == 'Bull':
            exit_multiple = self.exit_multiple_bull
        elif scenario == 'Bear':
            exit_multiple = self.exit_multiple_bear
        else:  # Base case
            exit_multiple = self.exit_multiple_base
            
        return final_ebitda * exit_multiple
    
    def calculate_irr(self, cash_flows):
        """Calculate IRR from a series of cash flows starting with negative investment."""
        # Make sure the first cash flow is negative (initial investment)
        if cash_flows[0] >= 0:
            cash_flows[0] = -cash_flows[0]
            
        # Define function to find root of NPV equation
        def npv_equation(r):
            return sum([cf / (1 + r) ** t for t, cf in enumerate(cash_flows)])
        
        try:
            # Find IRR using Newton's method
            irr = newton(npv_equation, x0=0.15, tol=1e-6, maxiter=1000)
            return max(min(irr, 1.0), -0.5)  # Constrain IRR to reasonable range
        except:
            # Return a very negative number if IRR calculation fails
            return -1.0
    
    def build_model(self, sample_size=100):
        """
        Build the full LBO model using the simulation results.
        
        Args:
            sample_size: Number of simulation paths to model in detail
        """
        # Reset results dictionary
        self.model_results = {}
        
        # Sample from the simulation results
        sample_indices = np.random.choice(
            range(len(self.simulation_results)), 
            size=min(sample_size, len(self.simulation_results)), 
            replace=False
        )
        
        # For each sampled simulation path
        for i, idx in enumerate(sample_indices):
            sim_data = self.simulation_results.iloc[idx].copy()
            scenario = sim_data['Scenario']
            
            # Create a path ID
            path_id = f"Path_{i+1}_{scenario}"
            
            # Extract revenue projections for each year
            years = range(len([col for col in sim_data.index if col.startswith('Year_')]))
            revenues = [sim_data[f'Year_{year}'] for year in range(len(years))]
            
            # Initialize financial metrics
            ebitdas = []
            capex = []
            nwc_changes = []
            fcfs = [] # Free Cash Flows
            debt_balances = [self.initial_debt]
            interest_expenses = []
            debt_repayments = []
            cash_to_equity = []
            
            # Calculate financial metrics for each projection year
            for year in range(1, len(years)):
                # Calculate EBITDA
                ebitda = self.calculate_ebitda(revenues[year], year)
                ebitdas.append(ebitda)
                
                # Calculate CapEx
                current_capex = self.calculate_capex(revenues[year])
                capex.append(current_capex)
                
                # Calculate change in net working capital
                nwc_change = self.calculate_nwc_change(revenues[year], revenues[year-1])
                nwc_changes.append(nwc_change)
                
                # Calculate interest expense
                interest_expense = debt_balances[-1] * self.debt_interest_rate
                interest_expenses.append(interest_expense)
                
                # Calculate taxable income
                taxable_income = max(0, ebitda - current_capex - interest_expense)
                
                # Calculate taxes
                taxes = taxable_income * self.tax_rate
                
                # Calculate free cash flow
                fcf = ebitda - current_capex - nwc_change - taxes
                fcfs.append(fcf)
                
                # Calculate debt repayment
                debt_repayment = min(fcf * self.debt_repayment_percent_fcf, debt_balances[-1])
                debt_repayment = max(debt_repayment, 0)  # Ensure non-negative
                debt_repayments.append(debt_repayment)
                
                # Update debt balance
                new_debt_balance = debt_balances[-1] - debt_repayment
                debt_balances.append(new_debt_balance)
                
                # Calculate cash flow to equity
                cf_to_equity = fcf - debt_repayment
                cash_to_equity.append(cf_to_equity)
            
            # Calculate exit value
            exit_value = self.calculate_exit_value(ebitdas[-1], scenario)
            
            # Calculate proceeds to equity at exit
            equity_proceeds = exit_value - debt_balances[-1]
            
            # Combine cash flows for IRR calculation
            equity_cash_flows = [-self.equity_contribution] + cash_to_equity + [equity_proceeds]
            
            # Calculate IRR
            irr = self.calculate_irr(equity_cash_flows)
            
            # Store the results
            self.model_results[path_id] = {
                'Revenues': revenues,
                'EBITDAs': [0] + ebitdas,  # Add 0 for year 0
                'CapEx': [0] + capex,  # Add 0 for year 0
                'NWC_Changes': [0] + nwc_changes,  # Add 0 for year 0
                'FCFs': [0] + fcfs,  # Add 0 for year 0
                'Debt_Balances': debt_balances,
                'Interest_Expenses': [0] + interest_expenses,  # Add 0 for year 0
                'Debt_Repayments': [0] + debt_repayments,  # Add 0 for year 0
                'Cash_to_Equity': [0] + cash_to_equity,  # Add 0 for year 0
                'Exit_Value': exit_value,
                'Final_Debt': debt_balances[-1],
                'Equity_Proceeds': equity_proceeds,
                'IRR': irr,
                'Scenario': scenario
            }
        
        return self.model_results
    
    def summarize_results(self):
        """Summarize the model results by scenario."""
        if not self.model_results:
            raise ValueError("Run build_model first before summarizing results")
        
        # Create summary DataFrames
        summary = {
            'All': {'Count': 0, 'Avg_IRR': 0, 'Median_IRR': 0, 'Min_IRR': 0, 'Max_IRR': 0},
            'Base': {'Count': 0, 'Avg_IRR': 0, 'Median_IRR': 0, 'Min_IRR': 0, 'Max_IRR': 0},
            'Bull': {'Count': 0, 'Avg_IRR': 0, 'Median_IRR': 0, 'Min_IRR': 0, 'Max_IRR': 0},
            'Bear': {'Count': 0, 'Avg_IRR': 0, 'Median_IRR': 0, 'Min_IRR': 0, 'Max_IRR': 0}
        }
        
        # Collect IRRs by scenario
        irrs = {'All': [], 'Base': [], 'Bull': [], 'Bear': []}
        
        for path_id, results in self.model_results.items():
            scenario = results['Scenario']
            irr = results['IRR']
            
            irrs['All'].append(irr)
            irrs[scenario].append(irr)
        
        # Calculate summary statistics
        for scenario in summary.keys():
            if irrs[scenario]:
                summary[scenario]['Count'] = len(irrs[scenario])
                summary[scenario]['Avg_IRR'] = np.mean(irrs[scenario])
                summary[scenario]['Median_IRR'] = np.median(irrs[scenario])
                summary[scenario]['Min_IRR'] = min(irrs[scenario])
                summary[scenario]['Max_IRR'] = max(irrs[scenario])
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary).T
        
        # Format IRR values as percentages
        for col in ['Avg_IRR', 'Median_IRR', 'Min_IRR', 'Max_IRR']:
            summary_df[col] = summary_df[col] * 100
        
        return summary_df
    
    def export_to_excel(self, output_path='output'):
        """Export model results to Excel."""
        if not self.model_results:
            raise ValueError("Run build_model first before exporting results")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # File path for the output Excel file
        file_path = os.path.join(output_path, 'buyout_model.xlsx')
        
        # Create a writer object
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        
        # Export summary
        summary_df = self.summarize_results()
        summary_df.to_excel(writer, sheet_name='Summary')
        
        # Export detailed results for each path
        for path_id, results in self.model_results.items():
            # Create a DataFrame for this path
            years = range(len(results['Revenues']))
            
            # Main financial metrics
            data = {
                'Year': list(years),
                'Revenue': results['Revenues'],
                'EBITDA': results['EBITDAs'],
                'CapEx': results['CapEx'],
                'NWC Change': results['NWC_Changes'],
                'FCF': results['FCFs'],
                'Debt Balance': results['Debt_Balances'],
                'Interest Expense': results['Interest_Expenses'],
                'Debt Repayment': results['Debt_Repayments'],
                'Cash to Equity': results['Cash_to_Equity']
            }
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add IRR and other summary info to the bottom
            summary_data = pd.DataFrame({
                'Metric': ['Scenario', 'Exit Value', 'Final Debt', 'Equity Proceeds', 'IRR'],
                'Value': [
                    results['Scenario'],
                    results['Exit_Value'],
                    results['Final_Debt'],
                    results['Equity_Proceeds'],
                    results['IRR'] * 100  # Convert to percentage
                ]
            })
            
            # Export to Excel
            df.to_excel(writer, sheet_name=path_id, index=False)
            
            # Write summary at the bottom
            workbook = writer.book
            worksheet = writer.sheets[path_id]
            
            # Format for percentage
            pct_format = workbook.add_format({'num_format': '0.00%'})
            
            # Get the row position for the summary (after the main table)
            start_row = len(df) + 3
            
            # Write the summary data
            for i, (metric, value) in enumerate(zip(summary_data['Metric'], summary_data['Value'])):
                worksheet.write(start_row + i, 0, metric)
                
                # Apply percentage format to IRR
                if metric == 'IRR':
                    worksheet.write(start_row + i, 1, value / 100, pct_format)
                else:
                    worksheet.write(start_row + i, 1, value)
        
        # Save the Excel file
        writer.close()
        
        print(f"Buyout model exported to {file_path}")
        return file_path


if __name__ == "__main__":
    # Test with dummy simulation results
    print("This module should be imported and used with actual simulation results.") 
