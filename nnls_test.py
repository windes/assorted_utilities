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
            
            # Dropdown menu to select an item
            selected_item = st.selectbox("Select an item", item_names)
            p_i = p[item_names.index(selected_item)]  # Estimated price of the selected item
            
            # Display the estimated price
            st.write(f"Estimated price of {selected_item}: ${p_i:.2f}")
            
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
                
                # Add effective cost per unit to the DataFrame
                bundles_with_item['effective_cost_per_unit'] = effective_cost_per_unit
                
                # Sort bundles by effective cost per unit (ascending = best value first)
                sorted_bundles = bundles_with_item.sort_values('effective_cost_per_unit')
                
                # Prepare display DataFrame with renamed columns for clarity
                display_df = sorted_bundles[['bundle_name', 'cost', selected_item, 'effective_cost_per_unit']]
                display_df = display_df.rename(columns={
                    'bundle_name': 'Bundle Name',
                    'cost': 'Bundle Cost',
                    selected_item: 'Quantity',
                    'effective_cost_per_unit': 'Effective Cost per Unit'
                })
                
                # Round the effective cost for better readability
                display_df['Effective Cost per Unit'] = display_df['Effective Cost per Unit'].round(2)
                
                # Display the sorted list of bundles
                st.dataframe(display_df)
            else:
                st.write("No bundles contain this item.")
