import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="CSV Data Viewer & Analyzer", layout="centered")

def main():
    # Create a title for the page
    st.title("CSV Data Viewer & Analyzer")
    
    # Add some description text
    st.write("Upload your CSV file below to view its contents and analyze the data.")
    
    # Create a file upload widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read the CSV file using pandas
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            
            # Display the dataframe as a table
            st.write("### Data Preview:")
            
            # Add sorting functionality
            column_to_sort = st.selectbox('Sort by:', df.columns.tolist())
            ascending = st.checkbox('Sort in ascending order')
            df_sorted = df.sort_values(by=column_to_sort, ascending=ascending)
            
            # Limit the number of rows displayed
            max_rows = st.slider('Max rows to display:', 1, len(df), 1000)
            df_display = df_sorted.head(max_rows)
            
            # Display the dataframe with sorting and row limiting
            st.dataframe(
                data=df_display,
                column_config=None,
                use_container_width=True,
                hide_index=False
            )
            
            # Add styling to make the table more readable
            st.write("### Table Styling:")
            st.markdown("""
                <style>
                .table-container {
                    overflow-x: auto;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                </style>
                """, unsafe_allow_html=True)
            
            st.write("### Data Analysis & Visualization:")
            
            # Section for Basic Statistics
            with st.expander("Basic Statistics"):
                st.subheader("Summary Statistics")
                if df.select_dtypes(include='number').columns.tolist():
                    st.write(df.describe())
                
            # Section for Variable Types
            with st.expander("Variable Types"):
                st.subheader("Data Types")
                dtypes = df.dtypes
                st.write(dtypes)
                
            # Data Visualization Section
            st.write("### Data Visualizations:")
            
            if len(df) > 0:
                # Numerical Analysis Section
                st.subheader("Numerical Analysis")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    selected_numeric_col = st.selectbox('Select Numeric Column:', numeric_cols)
                    fig_dist = px.histogram(df, x=selected_numeric_col)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Pairwise relationships for numerical columns
                    with st.expander("Relationships Between Numerical Variables"):
                        if len(numeric_cols) >= 2:
                            col_a = st.selectbox('Select Column A:', numeric_cols)
                            col_b = st.selectbox('Select Column B:', numeric_cols, 
                                               index=1 if len(numeric_cols) > 1 else 0)
                            
                            fig_relationships = px.scatter(df, x=col_a, y=col_b)
                            st.plotly_chart(fig_relationships, use_container_width=True)
                    
                # Categorical Analysis Section
                st.subheader("Categorical Analysis")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if categorical_cols:
                    selected_categorical_col = st.selectbox('Select Categorical Column:', categorical_cols)
                    fig_cat_dist = px.bar(df, x=selected_categorical_col)
                    st.plotly_chart(fig_cat_dist, use_container_width=True)
                    
                # Time Series Analysis Section
                st.subheader("Time Series Analysis")
                date_cols = [col for col in df.columns if ('date' in col.lower()) and 
                            (pd.api.types.is_datetime64_dtype(df[col]) or pd.to_numeric(df[col], errors='coerce').notna().all())]
                
                if date_cols:
                    selected_date_col = st.selectbox('Select Date Column:', date_cols)
                    
                    # Convert the date column to datetime
                    try:
                        df[selected_date_col] = pd.to_datetime(df[selected_date_col])
                        fig_time_series = px.line(df, x=selected_date_col, 
                                                y=df.columns.tolist())
                        st.plotly_chart(fig_time_series, use_container_width=True)
                    except ValueError:
                        st.error(f"Could not convert '{selected_date_col}' to datetime. Ensure it is in a proper date format.")
                
                # Geographical Data Section
                st.subheader("Geographical Analysis")
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    fig_map = px.scatter_geo(df, lat='latitude', lon='longitude')
                    st.plotly_chart(fig_map, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading CSV file or analyzing data: {str(e)}")
    
if __name__ == "__main__":
    main()