import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# Set the title of the Streamlit app
st.title("Bundle Value Calculator")

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Validate required columns
    if 'bundle_name' not in df.columns or 'cost' not in df.columns:
        st.error("CSV must contain 'bundle_name' and 'cost' columns.")
    else:
        # Extract item names (all columns except 'bundle_name' and 'cost')
        item_names = [col for col in df.columns if col not in ['bundle_name', 'cost']]
        
        # Validate that there is at least one item column and all relevant columns are numeric
        if len(item_names) == 0:
            st.error("CSV must contain at least one item column.")
        elif not all(df[col].dtype.kind in 'biufc' for col in item_names + ['cost']):
            st.error("Item quantities and costs must be numeric.")
        else:
            # Prepare data for non-negative least squares
            A = df[item_names].values  # Matrix of item quantities (bundles x items)
            c = df['cost'].values      # Vector of bundle costs
            
            # Solve for individual item prices using non-negative least squares
            p, _ = nnls(A, c)
            
            # Check for negative prices and warn the user
            if any(price < 0 for price in p):
                st.warning("Some items have negative estimated prices, which should not happen with non-negative least squares. This may indicate numerical instability or an issue with the dataset.")
            
            # Create tabs
            tab1, tab2 = st.tabs(["Bundle Analysis", "Item Prices"])
            
            # Tab 1: Bundle Analysis (existing functionality)
            with tab1:
                # Dropdown menu to select an item
                selected_item = st.selectbox("Select an item", item_names)
                p_i = p[item_names.index(selected_item)]  # Estimated price of the selected item
                
                # Display the estimated price
                st.write(f"Estimated price of {selected_item}: ${p_i:.2f}")
                
                # Warn if the estimated price is zero
                if p_i == 0:
                    st.warning(f"The estimated price of {selected_item} is $0.00. This may indicate insufficient data variation to accurately estimate the price, leading to unreliable results.")
                
                # Filter bundles that contain the selected item
                bundles_with_item = df[df[selected_item] > 0].copy()
                
                if not bundles_with_item.empty:
                    # Compute effective cost per unit for the selected item in each bundle
                    A_sub = bundles_with_item[item_names].values
                    c_sub = bundles_with_item['cost'].values
                    sum_all = A_sub @ p  # Total estimated cost of all items in each bundle
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
                # Create a DataFrame for item prices
                item_price_df = pd.DataFrame({
                    'Item': item_names,
                    'Estimated Price ($)': [round(price, 2) for price in p]
                })
                
                # Display a sortable table of items and their estimated prices
                st.dataframe(item_price_df, use_container_width=True)
