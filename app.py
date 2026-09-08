import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="EDA Interface",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Exploratory Data Analysis Interface")
st.write("Upload a CSV dataset to inspect its metadata and explore individual attributes.")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dataset Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is not None:

    # Read CSV file safely
    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read the uploaded CSV file: {e}")
        st.stop()

    # Check whether dataset is empty
    if df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    # --------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------

    st.header("Dataset Preview")

    st.dataframe(
        df.head(5),
        use_container_width=True
    )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    st.header("Dataset Metadata")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Rows", df.shape[0])

    with col2:
        st.metric("Number of Columns", df.shape[1])

    # Data types and missing values
    metadata = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.subheader("Column Information")

    st.dataframe(
        metadata,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # STATISTICAL SUMMARY
    # --------------------------------------------------

    st.subheader("Numerical Statistics")

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numerical_columns) > 0:

        statistics = pd.DataFrame({
            "Mean": df[numerical_columns].mean(),
            "Median": df[numerical_columns].median(),
            "Minimum": df[numerical_columns].min(),
            "Maximum": df[numerical_columns].max()
        })

        st.dataframe(
            statistics,
            use_container_width=True
        )

    else:
        st.info("No numerical attributes were found in this dataset.")

    # --------------------------------------------------
    # ATTRIBUTE SELECTION
    # --------------------------------------------------

    st.sidebar.subheader("Attribute Selection")

    selected_column = st.sidebar.selectbox(
        "Select a column for analysis",
        df.columns
    )

    # --------------------------------------------------
    # AUTOMATIC ATTRIBUTE CLASSIFICATION
    # --------------------------------------------------

    selected_data = df[selected_column]

    if pd.api.types.is_numeric_dtype(selected_data):
        attribute_type = "Numerical"
    else:
        attribute_type = "Categorical"

    st.sidebar.write(
        f"**Detected Attribute Type:** {attribute_type}"
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------

    st.header("Visualization")

    if attribute_type == "Numerical":

        st.subheader(f"Distribution of {selected_column}")

        # Remove missing values for histogram
        plot_data = selected_data.dropna()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            plot_data,
            bins=20,
            color="steelblue",
            edgecolor="black"
        )

        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Distribution of {selected_column}"
        )

        st.pyplot(fig)

    else:

        st.subheader(f"Frequency of {selected_column}")

        # Count categorical values
        value_counts = selected_data.fillna(
            "Missing"
        ).value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))

        value_counts.plot(
            kind="bar",
            ax=ax,
            color="teal",
            edgecolor="black"
        )

        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Frequency Distribution of {selected_column}"
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

else:

    st.info(
        "👈 Upload a CSV file from the sidebar to begin EDA."
    )

    st.markdown("""
    ### Supported File Format

    - CSV (`.csv`)

    ### Test Dataset

    You can test this application using **titanic.csv**.
    """)