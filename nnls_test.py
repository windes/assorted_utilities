import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp

# Set the title of the Streamlit app
st.title("Bundle Value Calculator")

# File uploader for Excel (.xlsx) input
uploaded_file = st.file_uploader("Upload Excel (.xlsx) file", type="xlsx")

if uploaded_file is not None:
    # Read the Excel file
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    
    # Read the bundle data from the first sheet
    if len(sheet_names) < 1:
        st.error("Excel file must contain at least one sheet with bundle data.")
    else:
        df_bundles = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
        
        # Validate required columns in bundle data
        if 'bundle_name' not in df_bundles.columns or 'cost' not in df_bundles.columns:
            st.error("Bundle sheet must contain 'bundle_name' and 'cost' columns.")
        else:
            # Extract item names (all columns except 'bundle_name' and 'cost')
            item_names = [col for col in df_bundles.columns if col not in ['bundle_name', 'cost']]
            
            # Validate that there is at least one item column and all relevant columns are numeric
            if len(item_names) == 0:
                st.error("Bundle sheet must contain at least one item column.")
            elif not all(df_bundles[col].dtype.kind in 'biufc' for col in item_names + ['cost']):
                st.error("Item quantities and costs in bundle sheet must be numeric.")
            else:
                # Prepare data for optimization
                A = df_bundles[item_names].values  # Matrix of item quantities (bundles x items)
                c = df_bundles['cost'].values      # Vector of bundle costs
                n_items = len(item_names)
                
                # Scale the item quantities to improve numerical stability
                scaling_factors = np.max(np.abs(A), axis=0)
                scaling_factors[scaling_factors == 0] = 1  # Avoid division by zero
                A_scaled = A / scaling_factors
                p_scaling = scaling_factors  # To rescale prices later
                
                # Read trade data from the second sheet, if it exists
                trades = []
                if len(sheet_names) > 1:
                    df_trades = pd.read_excel(uploaded_file, sheet_name=sheet_names[1])
                    required_trade_cols = ['give_item', 'give_quantity', 'receive_item', 'receive_quantity']
                    if not all(col in df_trades.columns for col in required_trade_cols):
                        st.warning("Trades sheet must contain 'give_item', 'give_quantity', 'receive_item', and 'receive_quantity' columns. Ignoring trade data.")
                    else:
                        for _, row in df_trades.iterrows():
                            if row['give_item'] in item_names and row['receive_item'] in item_names:
                                trades.append({
                                    'give_item': row['give_item'],
                                    'give_quantity': row['give_quantity'],
                                    'receive_item': row['receive_item'],
                                    'receive_quantity': row['receive_quantity']
                                })
                            else:
                                st.warning(f"Trade involving {row['give_item']} or {row['receive_item']} ignored: Items not found in bundle data.")
                
                # Set up the optimization problem with cvxpy
                p = cp.Variable(n_items)  # Item prices (scaled)
                lambda_reg = 0.1  # Regularization parameter
                lambda_trade = 1.0  # Trade constraint penalty
                
                # Objective: ||A @ p - c||^2 + lambda_reg * ||p||^2
                objective = cp.sum_squares(A_scaled @ p - c) + lambda_reg * cp.sum_squares(p)
                
                # Add soft trade constraints as penalty terms
                if trades:
                    trade_penalties = []
                    for trade in trades:
                        give_idx = item_names.index(trade['give_item'])
                        receive_idx = item_names.index(trade['receive_item'])
                        # Scale the quantities
                        give_qty = trade['give_quantity'] / scaling_factors[give_idx]
                        receive_qty = trade['receive_quantity'] / scaling_factors[receive_idx]
                        # Penalty: (give_quantity * p[give_item] - receive_quantity * p[receive_item])^2
                        penalty = cp.sum_squares(give_qty * p[give_idx] - receive_qty * p[receive_idx])
                        trade_penalties.append(penalty)
                    if trade_penalties:
                        objective += lambda_trade * cp.sum(trade_penalties)
                
                # Constraints: p >= 0
                constraints = [p >= 0]
                
                # Solve the problem
                problem = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    problem.solve(solver=cp.SCS, eps=1e-5)  # Use SCS solver with relaxed tolerance
                    if problem.status == cp.OPTIMAL:
                        p_values = p.value
                        # Clip to non-negative to handle numerical precision issues
                        p_values = np.maximum(p_values, 0)
                    else:
                        st.warning("Optimization did not converge to an optimal solution. Falling back to NNLS.")
                        p_values = None
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}. Falling back to bundle data only.")
                    p_values = None
                
                # If optimization fails or did not converge, fall back to regularized NNLS
                if p_values is None:
                    from scipy.optimize import nnls
                    A_reg = np.vstack([A_scaled, np.sqrt(lambda_reg) * np.eye(n_items)])
                    c_reg = np.concatenate([c, np.zeros(n_items)])
                    p_values, _ = nnls(A_reg, c_reg)
                    # NNLS ensures non-negative values, but clip for consistency
                    p_values = np.maximum(p_values, 0)
                
                # Rescale the prices
                p_values = p_values * p_scaling
                
                # Normalize prices to better match total bundle costs
                predicted_costs = A @ p_values
                nonzero_mask = predicted_costs > 0
                if np.any(nonzero_mask):
                    scaling_factor = np.mean(c[nonzero_mask] / predicted_costs[nonzero_mask])
                    p_values = p_values * scaling_factor
                
                # Check for negative prices and warn the user (should not occur with clipping)
                if any(price < 0 for price in p_values):
                    st.warning("Some items have negative estimated prices after processing, which should not happen.")
                
                # Function to format price based on its value
                def format_price(price, item_name):
                    if price < 0.001:
                        scaled_price = price * 1000
                        return f"${scaled_price:.2f} (1k {item_name})"
                    elif price < 0.01:
                        scaled_price = price * 100
                        return f"${scaled_price:.2f} (100 {item_name})"
                    else:
                        return f"${price:.2f}"
                
                # Create tabs
                tab1, tab2, tab3 = st.tabs(["Bundle Analysis", "Item Prices", "Bundle Breakdown"])
                
                # Tab 1: Bundle Analysis
                with tab1:
                    # Dropdown menu to select an item
                    selected_item = st.selectbox("Select an item", item_names)
                    p_i = p_values[item_names.index(selected_item)]  # Estimated price of the selected item
                    
                    # Display the estimated price with scaling if necessary
                    st.write(f"Estimated price of {selected_item}: {format_price(p_i, selected_item)}")
                    
                    # Warn if the estimated price is very close to zero
                    if p_i < 1e-6:
                        st.warning(f"The estimated price of {selected_item} is effectively $0.00. This may indicate insufficient data to accurately estimate the price.")
                    
                    # Filter bundles that contain the selected item
                    bundles_with_item = df_bundles[df_bundles[selected_item] > 0].copy()
                    
                    if not bundles_with_item.empty:
                        # Compute effective cost per unit for the selected item in each bundle
                        A_sub = bundles_with_item[item_names].values
                        c_sub = bundles_with_item['cost'].values
                        sum_all = A_sub @ p_values  # Total estimated cost of all items in each bundle
                        sum_selected = bundles_with_item[selected_item] * p_i  # Estimated cost of selected item
                        sum_other = sum_all - sum_selected  # Estimated cost of other items
                        effective_cost = c_sub - sum_other  # Effective cost of selected item
                        effective_cost_per_unit = effective_cost / bundles_with_item[selected_item]
                        
                        # Calculate effective units per $1 (reciprocal of effective cost per unit)
                        effective_units_per_dollar = 1 / effective_cost_per_unit
                        
                        # Add effective units per $1 to the DataFrame
                        bundles_with_item['effective_units_per_dollar'] = effective_units_per_dollar
                        
                        # Sort bundles by effective units per dollar (descending = best value first)
                        sorted_bundles = bundles_with_item.sort_values('effective_units_per_dollar', ascending=False)
                        
                        # Prepare display DataFrame with renamed columns for clarity
                        display_df = sorted_bundles[['bundle_name', 'cost', selected_item, 'effective_units_per_dollar']]
                        display_df = display_df.rename(columns={
                            'bundle_name': 'Bundle Name',
                            'cost': 'Bundle Cost',
                            selected_item: 'Quantity',
                            'effective_units_per_dollar': 'Effective Units per $1'
                        })
                        
                        # Round the effective units per $1 for better readability
                        display_df['Effective Units per $1'] = display_df['Effective Units per $1'].round(2)
                        
                        # Display the sorted list of bundles
                        st.dataframe(display_df)
                    else:
                        st.write("No bundles contain this item.")
                
                # Tab 2: Item Prices
                with tab2:
                    # Create a DataFrame for item prices with scaled prices
                    item_price_data = []
                    for item, price in zip(item_names, p_values):
                        if price < 0.001:
                            scaled_price = price * 1000
                            display_price = f"${scaled_price:.2f} (1k)"
                        elif price < 0.01:
                            scaled_price = price * 100
                            display_price = f"${scaled_price:.2f} (100)"
                        else:
                            scaled_price = price
                            display_price = f"${scaled_price:.2f}"
                        item_price_data.append({'Item': item, 'Estimated Price ($)': display_price})
                    
                    item_price_df = pd.DataFrame(item_price_data)
                    
                    # Display a sortable table of items and their estimated prices
                    st.dataframe(item_price_df, use_container_width=True)
                
                # Tab 3: Bundle Breakdown
                with tab3:
                    bundle_names = df_bundles['bundle_name'].tolist()
                    selected_bundle = st.selectbox("Select a bundle", bundle_names, key='bundle_select')
                    bundle_row = df_bundles[df_bundles['bundle_name'] == selected_bundle].iloc[0]
                    item_data = []
                    total_estimated_value = 0
                    for item in item_names:
                        quantity = bundle_row[item]
                        if quantity > 0:
                            item_value = quantity * p_values[item_names.index(item)]
                            total_estimated_value += item_value
                            item_data.append({
                                'Item': item,
                                'Quantity': quantity,
                                'Fraction of Total Estimated Value': item_value / total_estimated_value if total_estimated_value > 0 else 0,
                                'Estimated Value': item_value
                            })
                    if item_data:
                        bundle_df = pd.DataFrame(item_data)
                        bundle_df['Estimated Value'] = bundle_df['Estimated Value'].round(2)
                        bundle_df['Fraction of Total Estimated Value'] = (bundle_df['Fraction of Total Estimated Value'] * 100).round(2).astype(str) + '%'
                        st.dataframe(bundle_df)
                        actual_price = bundle_row['cost']
                        st.write(f"Total Estimated Value: ${total_estimated_value:.2f}")
                        st.write(f"Actual Price: ${actual_price:.2f}")
                        if total_estimated_value > actual_price:
                            st.write("Good Deal")
                        elif total_estimated_value < actual_price:
                            st.write("Bad Deal")
                        else:
                            st.write("Fair Deal")
                    else:
                        st.write("This bundle has no items.")
