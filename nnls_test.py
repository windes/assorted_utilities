import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

# Set the app title
st.title("Bundle Value Calculator")

# File uploader for bundle CSV (required)
bundle_file = st.file_uploader("Upload Bundle CSV file", type="csv")

# File uploader for trade CSV (optional)
trade_file = st.file_uploader("Upload Trade CSV file (optional)", type="csv")

if bundle_file is not None:
    # Load bundle data
    df_bundle = pd.read_csv(bundle_file)
    
    # Check for required columns
    if 'bundle_name' not in df_bundle.columns or 'cost' not in df_bundle.columns:
        st.error("Bundle CSV must have 'bundle_name' and 'cost' columns.")
    else:
        # Get item names (exclude 'bundle_name' and 'cost')
        item_names = [col for col in df_bundle.columns if col not in ['bundle_name', 'cost']]
        
        if len(item_names) == 0:
            st.error("Bundle CSV must have at least one item column.")
        elif not all(df_bundle[col].dtype.kind in 'biufc' for col in item_names + ['cost']):
            st.error("Item quantities and costs must be numeric.")
        else:
            # Prepare bundle data for modeling
            A = df_bundle[item_names].values  # Item quantities matrix
            c = df_bundle['cost'].values      # Bundle costs vector
            
            # Load and process trade data if provided
            trades = []
            if trade_file is not None:
                df_trade = pd.read_csv(trade_file)
                if not all(col in df_trade.columns for col in ['give_item', 'give_quantity', 'receive_item', 'receive_quantity']):
                    st.error("Trade CSV must have 'give_item', 'give_quantity', 'receive_item', and 'receive_quantity' columns.")
                else:
                    item_index = {item: idx for idx, item in enumerate(item_names)}
                    for _, row in df_trade.iterrows():
                        give_item = row['give_item']
                        give_qty = row['give_quantity']
                        receive_item = row['receive_item']
                        receive_qty = row['receive_quantity']
                        if give_item in item_index and receive_item in item_index:
                            trades.append((item_index[give_item], give_qty, item_index[receive_item], receive_qty))
                        else:
                            st.warning(f"Skipping trade with unknown items: {give_item} or {receive_item}")
            
            # Build Bayesian model
            with pm.Model() as model:
                # Item values (non-negative prior)
                p = pm.HalfNormal('p', sigma=1, shape=len(item_names))
                
                # Bundle price likelihood
                sigma_bundle = pm.HalfNormal('sigma_bundle', sigma=1)
                bundle_pred = pm.math.dot(A, p)
                pm.Normal('bundle_obs', mu=bundle_pred, sigma=sigma_bundle, observed=c)
                
                # Trade likelihood (if trade data exists)
                if trades:
                    sigma_trade = pm.HalfNormal('sigma_trade', sigma=1)
                    for give_idx, give_qty, receive_idx, receive_qty in trades:
                        trade_pred = receive_qty * p[receive_idx]
                        pm.Normal(f'trade_{give_idx}_{receive_idx}', mu=trade_pred, sigma=sigma_trade, observed=give_qty * p[give_idx])
            
            # Fit the model
            with model:
                trace = pm.sample(1000, tune=1000, chains=2, cores=1)
            
            # Get mean item values from posterior
            p_mean = trace.posterior['p'].mean(dim=['chain', 'draw']).values
            
            # Create tabs for analysis
            tab1, tab2 = st.tabs(["Bundle Analysis", "Item Prices"])
            
            # Tab 1: Bundle Analysis
            with tab1:
                selected_item = st.selectbox("Select an item", item_names)
                p_i = p_mean[item_names.index(selected_item)]
                
                st.write(f"Estimated price of {selected_item}: ${p_i:.2f}")
                
                # Filter bundles with the selected item
                bundles_with_item = df_bundle[df_bundle[selected_item] > 0].copy()
                
                if not bundles_with_item.empty:
                    # Calculate bundle metrics
                    A_sub = bundles_with_item[item_names].values
                    c_sub = bundles_with_item['cost'].values
                    sum_all = A_sub @ p_mean  # Estimated total value
                    sum_selected = bundles_with_item[selected_item] * p_i
                    sum_other = sum_all - sum_selected
                    effective_cost = c_sub - sum_other
                    effective_cost_per_unit = effective_cost / bundles_with_item[selected_item]
                    effective_units_per_dollar = 1 / effective_cost_per_unit
                    value_relative_to_price = (sum_all / c_sub) * 100  # As percentage
                    
                    # Add to DataFrame
                    bundles_with_item['effective_units_per_dollar'] = effective_units_per_dollar
                    bundles_with_item['value_relative_to_price'] = value_relative_to_price
                    
                    # Sort by value
                    sorted_bundles = bundles_with_item.sort_values('effective_units_per_dollar', ascending=False)
                    
                    # Prepare display table
                    display_df = sorted_bundles[['bundle_name', 'cost', selected_item, 'effective_units_per_dollar', 'value_relative_to_price']]
                    display_df = display_df.rename(columns={
                        'bundle_name': 'Bundle Name',
                        'cost': 'Bundle Cost',
                        selected_item: 'Quantity',
                        'effective_units_per_dollar': 'Effective Units per $1',
                        'value_relative_to_price': 'Value Relative to Price (%)'
                    })
                    
                    # Round for readability
                    display_df['Effective Units per $1'] = display_df['Effective Units per $1'].round(2)
                    display_df['Value Relative to Price (%)'] = display_df['Value Relative to Price (%)'].round(2)
                    
                    st.dataframe(display_df)
                else:
                    st.write("No bundles contain this item.")
            
            # Tab 2: Item Prices
            with tab2:
                item_price_df = pd.DataFrame({
                    'Item': item_names,
                    'Estimated Price ($)': [round(price, 2) for price in p_mean]
                })
                st.dataframe(item_price_df, use_container_width=True)