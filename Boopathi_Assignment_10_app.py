
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Excel Sheet Merger",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Multi-Sheet Excel Data Integration")
st.write("Upload an Excel file to merge all sheets into one CSV file.")


# Reusable function to merge Excel sheets
def merge_excel_sheets(uploaded_file):
    excel_file = pd.ExcelFile(uploaded_file)

    sheet_names = excel_file.sheet_names
    all_data = []

    for sheet in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet)

        # Skip empty sheets
        if df.empty:
            continue

        # Remove completely empty rows
        df = df.dropna(how="all")

        # Add source sheet information
        df["Source_Sheet"] = sheet

        all_data.append(df)

    if not all_data:
        return pd.DataFrame(), sheet_names

    # Merge all sheets
    merged_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove duplicate rows within the same sheet
    merged_df = merged_df.drop_duplicates()

    return merged_df, sheet_names


# Convert DataFrame into CSV
def convert_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# Upload Excel file
uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    try:
        # Validate uploaded file
        if uploaded_file.size == 0:
            st.error("Uploaded file is empty.")
            st.stop()

        # Process Excel file
        merged_df, sheet_names = merge_excel_sheets(
            uploaded_file
        )

        st.success("Excel file uploaded successfully!")

        # Display sheet information
        st.subheader("📁 Excel Sheet Details")

        st.write("Total Sheets:", len(sheet_names))
        st.write("Sheet Names:", ", ".join(sheet_names))

        if not merged_df.empty:

            # Dataset statistics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Sheets",
                len(sheet_names)
            )

            col2.metric(
                "Total Rows",
                len(merged_df)
            )

            col3.metric(
                "Total Columns",
                len(merged_df.columns)
            )

            # Preview merged data
            st.subheader("📋 Merged Data Preview")

            st.dataframe(
                merged_df,
                use_container_width=True
            )

            # Download CSV
            csv_data = convert_to_csv(merged_df)

            st.download_button(
                label="⬇ Download Merged CSV",
                data=csv_data,
                file_name="merged_excel_data.csv",
                mime="text/csv"
            )

            st.success("CSV file is ready for download!")

        else:
            st.warning("No valid data found in Excel sheets.")

    except Exception as e:
        st.error(f"Error processing Excel file: {e}")

else:
    st.info("Please upload an Excel file to continue.")
