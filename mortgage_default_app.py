import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(
    page_title="Advanced FICO Score Bucketing",
    page_icon="📊",
    layout="wide"
)

class OptimizedFICOBucketing:
    def __init__(self):
        self.data = None
        self.results = None
    
    def parse_csv(self, text):
        """Parse CSV data with robust error handling and auto-detection"""
        try:
            # Try multiple delimiters and encodings
            for delimiter in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(StringIO(text), delimiter=delimiter)
                    if len(df.columns) > 1:
                        st.info(f"Detected delimiter: '{delimiter}' with {len(df.columns)} columns")
                        break
                except Exception as e:
                    continue
            
            if len(df.columns) <= 1:
                st.error("Could not parse CSV with common delimiters. Please check file format.")
                return None
            
            # Clean column names and handle case sensitivity
            df.columns = [str(col).lower().strip().replace('"', '').replace("'", "").replace(' ', '_') 
                         for col in df.columns]
            
            st.info(f"Columns found: {list(df.columns)}")
            return df
        except Exception as e:
            raise Exception(f"CSV parsing failed: {str(e)}")
    
    def detect_columns(self, df):
        """Automatically detect FICO and default columns with flexible matching"""
        fico_patterns = ['fico', 'score', 'credit', 'rating', 'creditscore', 'fico_score', 'credit_score']
        default_patterns = ['default', 'target', 'label', 'y', 'class', 'delinquent', 'bad', 'charged_off', 'default_flag', 'bad_flag']
        
        fico_col = None
        default_col = None
        
        # Find FICO column - check exact matches first
        for col in df.columns:
            col_lower = col.lower()
            # Exact matches
            if col_lower in ['fico', 'score', 'credit_score', 'fico_score']:
                fico_col = col
                break
            # Partial matches
            for pattern in fico_patterns:
                if pattern in col_lower:
                    fico_col = col
                    break
            if fico_col:
                break
        
        # Find default column
        for col in df.columns:
            col_lower = col.lower()
            # Exact matches
            if col_lower in ['default', 'target', 'bad', 'default_flag']:
                default_col = col
                break
            # Partial matches
            for pattern in default_patterns:
                if pattern in col_lower:
                    default_col = col
                    break
            if default_col:
                break
        
        st.info(f"Detected FICO column: {fico_col}, Default column: {default_col}")
        return fico_col, default_col
    
    def validate_and_prepare_data(self, df, fico_col, default_col):
        """Validate data and prepare for analysis with intelligent default rate correction"""
        st.info(f"Starting data validation with {len(df)} records")
        
        # Validate FICO column
        if not fico_col or fico_col not in df.columns:
            available_cols = list(df.columns)
            st.error(f"FICO column '{fico_col}' not found in data. Available columns: {available_cols}")
            raise ValueError(f"FICO column not found. Available columns: {available_cols}")
        
        st.info(f"Processing FICO column: {fico_col}")
        
        # Convert FICO scores to numeric, handling errors
        df[fico_col] = pd.to_numeric(df[fico_col], errors='coerce')
        
        # Show data quality info
        original_count = len(df)
        df = df[df[fico_col].notna()]
        na_count = original_count - len(df)
        
        if na_count > 0:
            st.warning(f"Removed {na_count} records with non-numeric FICO scores")
        
        # Check FICO score range
        fico_min = df[fico_col].min()
        fico_max = df[fico_col].max()
        st.info(f"FICO score range before filtering: {fico_min} - {fico_max}")
        
        # Remove rows with invalid FICO scores
        df = df[(df[fico_col] >= 300) & (df[fico_col] <= 850)]
        filtered_count = original_count - na_count - len(df)
        
        if filtered_count > 0:
            st.warning(f"Removed {filtered_count} records with FICO scores outside 300-850 range")
        
        if len(df) == 0:
            st.error("No valid FICO scores found after filtering. Please check your data.")
            raise ValueError("No valid FICO scores found (must be between 300-850)")
        
        st.info(f"After validation: {len(df)} valid records, FICO range: {df[fico_col].min()}-{df[fico_col].max()}")
        
        # Handle default column with INTELLIGENT validation and correction
        if default_col:
            if default_col not in df.columns:
                st.warning(f"Default column '{default_col}' not found. Using zeros for default data.")
                df['default'] = 0
            else:
                # Show what's in the default column before processing
                unique_defaults_before = df[default_col].unique()
                st.info(f"Unique values in default column '{default_col}': {sorted(unique_defaults_before)}")
                
                # Convert to numeric and handle errors
                df['default_processed'] = pd.to_numeric(df[default_col], errors='coerce')
                
                # Check the distribution before any processing
                value_counts_before = df['default_processed'].value_counts().sort_index()
                st.info(f"Default value distribution before processing: {dict(value_counts_before)}")
                
                # Calculate initial default rate
                valid_defaults = df['default_processed'].notna()
                if valid_defaults.sum() > 0:
                    initial_default_rate = df.loc[valid_defaults, 'default_processed'].mean() * 100
                    st.info(f"Initial default rate: {initial_default_rate:.1f}%")
                    
                    # INTELLIGENT CORRECTION LOGIC
                    if initial_default_rate > 50:
                        st.warning("🚨 HIGH DEFAULT RATE DETECTED - Analyzing potential issues...")
                        
                        # Check if this might be reversed coding (1=good, 0=bad)
                        if len(value_counts_before) == 2 and set(value_counts_before.index) == {0, 1}:
                            st.error("⚠️ Likely reversed coding detected! Auto-correcting (swapping 0 and 1)...")
                            df['default'] = 1 - df['default_processed']
                        else:
                            # For non-binary or multiple values, let user decide
                            st.error("""
                            **POTENTIAL DATA ISSUE DETECTED:**
                            
                            The default rate appears unrealistically high (>50%). This could mean:
                            1. **Reversed coding**: 1 = Good, 0 = Bad (most common)
                            2. **Different definition**: Your 'default' might mean something else
                            3. **Data quality issue**: Incorrect values in the column
                            
                            **Attempting intelligent correction...**
                            """)
                            
                            # Try to detect the most common pattern
                            if 0 in value_counts_before.index and 1 in value_counts_before.index:
                                # If we have clear 0/1 values but high default rate, assume reversed
                                st.info("Detected binary 0/1 values with high default rate - assuming reversed coding")
                                df['default'] = 1 - df['default_processed']
                            else:
                                # For other cases, use median split or let user decide
                                st.warning("Using median-based correction for non-binary default values")
                                median_val = df['default_processed'].median()
                                df['default'] = (df['default_processed'] < median_val).astype(int)
                    else:
                        # Normal case - use as is
                        df['default'] = df['default_processed']
                    
                    # Fill NaN values with 0 (assuming no default for missing values)
                    df['default'] = df['default'].fillna(0)
                    
                    # Ensure binary (0/1)
                    df['default'] = (df['default'] > 0).astype(int)
                    
                    # Calculate final default rate
                    final_default_rate = df['default'].mean() * 100
                    value_counts_final = df['default'].value_counts().sort_index()
                    
                    st.success(f"✅ Default processing completed:")
                    st.info(f"Final default rate: {final_default_rate:.1f}%")
                    st.info(f"Final distribution: {dict(value_counts_final)}")
                    
                    # Final sanity check
                    if final_default_rate > 30:
                        st.warning("""
                        **Note:** Even after correction, default rate is >30%. 
                        In real credit data, typical default rates are 1-10%.
                        Your data might have a different definition of 'default'.
                        """)
                    
                else:
                    st.warning("Default column contains no valid numeric values. Using zeros.")
                    df['default'] = 0
                
                # Clean up temporary column
                df = df.drop('default_processed', axis=1)
                    
        else:
            st.warning("No default column detected. Using zeros for default data.")
            df['default'] = 0
        
        return df.rename(columns={fico_col: 'fico'})[['fico', 'default']]
    
    def vectorized_mse_bucketing(self, scores, k):
        """Vectorized MSE-based bucketing with optimized convergence"""
        scores = np.sort(scores)
        n = len(scores)
        
        # Smart initialization using percentiles
        boundaries = np.percentile(scores, np.linspace(0, 100, k + 1))[1:-1]
        boundaries = np.append(boundaries, scores[-1] + 1)
        
        max_iterations = 20
        tolerance = 0.01
        
        for iteration in range(max_iterations):
            # Vectorized bucket assignment
            bucket_indices = np.digitize(scores, boundaries)
            bucket_indices = np.clip(bucket_indices, 0, k - 1)
            
            # Vectorized centroid calculation
            centroids = np.array([scores[bucket_indices == i].mean() if np.sum(bucket_indices == i) > 0 
                                else boundaries[i] for i in range(k)])
            
            # Update boundaries (midpoints between centroids)
            new_boundaries = (centroids[:-1] + centroids[1:]) / 2
            
            # Check convergence
            if np.max(np.abs(new_boundaries - boundaries[:-1])) < tolerance:
                break
                
            boundaries = np.append(new_boundaries, scores[-1] + 1)
        
        # Final bucket assignment and MSE calculation
        bucket_indices = np.digitize(scores, boundaries)
        bucket_indices = np.clip(bucket_indices, 0, k - 1)
        
        centroids = np.array([scores[bucket_indices == i].mean() for i in range(k)])
        mse = np.mean((scores - centroids[bucket_indices]) ** 2)
        
        # Calculate bucket counts
        bucket_counts = np.bincount(bucket_indices, minlength=k)
        
        return {
            'boundaries': np.concatenate([[scores[0]], boundaries]),
            'centroids': centroids,
            'mse': mse,
            'bucket_counts': bucket_counts,
            'iterations': iteration + 1
        }
    
    def optimized_log_likelihood_bucketing(self, data_points, k):
        """Optimized log-likelihood with smarter initialization"""
        df_sorted = data_points.sort_values('fico').reset_index(drop=True)
        n = len(df_sorted)
        
        # Precompute cumulative sums for O(1) bucket statistics
        cumsum_fico = df_sorted['fico'].cumsum().values
        cumsum_default = df_sorted['default'].cumsum().values
        
        def get_bucket_stats(start, end):
            """Get bucket statistics in O(1) time using precomputed cumulative sums"""
            if start >= end:
                return 0, 0, 0, 0
            n_bucket = end - start
            sum_fico = cumsum_fico[end-1] - (cumsum_fico[start-1] if start > 0 else 0)
            sum_default = cumsum_default[end-1] - (cumsum_default[start-1] if start > 0 else 0)
            avg_fico = sum_fico / n_bucket
            return n_bucket, sum_default, avg_fico, sum_default / n_bucket if n_bucket > 0 else 0
        
        def bucket_log_likelihood(start, end):
            """Calculate log-likelihood for a bucket"""
            n_bucket, sum_default, _, p_default = get_bucket_stats(start, end)
            if n_bucket == 0 or sum_default == 0 or sum_default == n_bucket:
                return 0
            return n_bucket * (p_default * math.log(p_default) + (1 - p_default) * math.log(1 - p_default))
        
        # Smart initialization using default rate changes
        default_rates = []
        window_size = max(10, n // 50)
        for i in range(0, n, window_size):
            end = min(i + window_size, n)
            _, sum_default, _, _ = get_bucket_stats(i, end)
            default_rates.append((i, sum_default / (end - i)))
        
        # Find points with significant default rate changes
        candidate_splits = []
        for i in range(1, len(default_rates)):
            if abs(default_rates[i][1] - default_rates[i-1][1]) > 0.05:
                candidate_splits.append(default_rates[i][0])
        
        # Use K-means on candidate splits for smart initialization
        if len(candidate_splits) >= k - 1:
            kmeans = KMeans(n_clusters=k-1, n_init=5, random_state=42)
            cluster_labels = kmeans.fit_predict(np.array(candidate_splits).reshape(-1, 1))
            boundaries = []
            for i in range(k-1):
                cluster_points = [candidate_splits[j] for j in range(len(candidate_splits)) if cluster_labels[j] == i]
                if cluster_points:
                    boundaries.append(int(np.median(cluster_points)))
        else:
            # Fallback to equal frequency
            boundaries = [int((i/k) * n) for i in range(1, k)]
        
        boundaries = sorted(set(boundaries))
        boundaries.append(n)
        
        # Refine boundaries using local optimization
        best_likelihood = -np.inf
        best_boundaries = boundaries.copy()
        
        for refinement_iter in range(10):
            current_likelihood = 0
            start = 0
            new_boundaries = []
            
            for boundary in boundaries[:-1]:
                end = boundary
                # Local optimization around current boundary
                def objective(offset):
                    test_end = max(start + 1, min(end + offset, boundaries[boundaries.index(boundary) + 1] - 1))
                    return -bucket_log_likelihood(start, test_end)
                
                try:
                    result = minimize_scalar(objective, bounds=(-50, 50), method='bounded')
                    optimized_end = max(start + 1, min(end + result.x, boundaries[boundaries.index(boundary) + 1] - 1))
                except:
                    optimized_end = end
                
                current_likelihood += bucket_log_likelihood(start, optimized_end)
                new_boundaries.append(optimized_end)
                start = optimized_end
            
            new_boundaries.append(n)
            
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_boundaries = new_boundaries.copy()
            
            boundaries = new_boundaries
        
        # Convert indices to FICO scores
        fico_boundaries = [df_sorted.iloc[0]['fico']]
        for boundary in best_boundaries[:-1]:
            fico_boundaries.append(df_sorted.iloc[min(boundary, n-1)]['fico'])
        fico_boundaries.append(df_sorted.iloc[n-1]['fico'] + 1)
        
        # Calculate final statistics
        bucket_stats = []
        start = 0
        for boundary in best_boundaries:
            n_bucket, sum_default, avg_fico, p_default = get_bucket_stats(start, boundary)
            bucket_stats.append({
                'ni': n_bucket,
                'ki': sum_default,
                'pi': p_default,
                'avg_fico': avg_fico
            })
            start = boundary
        
        return {
            'boundaries': fico_boundaries,
            'log_likelihood': best_likelihood,
            'bucket_stats': bucket_stats
        }
    
    def create_rating_map(self, boundaries, num_buckets):
        """Create rating map with CORRECT FICO descriptions (higher score = better credit)"""
        rating_map = []
        
        for i in range(num_buckets):
            lower_bound = int(math.ceil(boundaries[i]))
            upper_bound = int(math.floor(boundaries[i + 1])) - (1 if i < num_buckets - 1 else 0)
            
            # CORRECT: Higher rating number = better credit (industry standard)
            # Rating 1 = lowest score, Rating 5/7 = highest score
            rating = i + 1
            
            # Get appropriate description
            if num_buckets == 5:
                if rating == 1:
                    description = "Poor Credit (300-579)"
                elif rating == 2:
                    description = "Fair Credit (580-669)"
                elif rating == 3:
                    description = "Good Credit (670-739)"
                elif rating == 4:
                    description = "Very Good Credit (740-799)"
                else:  # rating == 5
                    description = "Exceptional Credit (800-850)"
            elif num_buckets == 7:
                if rating == 1:
                    description = "Very Poor Credit (300-579)"
                elif rating == 2:
                    description = "Poor Credit (580-629)"
                elif rating == 3:
                    description = "Below Average Credit (630-679)"
                elif rating == 4:
                    description = "Average Credit (680-719)"
                elif rating == 5:
                    description = "Good Credit (720-739)"
                elif rating == 6:
                    description = "Very Good Credit (740-799)"
                else:  # rating == 7
                    description = "Exceptional Credit (800-850)"
            else:
                # Generic descriptions for other bucket counts
                if rating == 1:
                    description = "Highest Risk"
                elif rating == num_buckets:
                    description = "Lowest Risk"
                else:
                    description = f"Rating {rating}"
            
            rating_map.append({
                'Rating': rating,
                'FICO Range': f"{lower_bound} - {upper_bound}",
                'Description': description
            })
        
        return rating_map
    
    def generate_sample_data(self):
        """Generate realistic sample FICO data with proper distribution"""
        np.random.seed(42)
        n_samples = 1000
        
        # Realistic FICO distribution parameters with tighter std to stay within bounds
        fico_params = [
            {'mean': 500, 'std': 30, 'weight': 0.10, 'default_prob': 0.30},   # Very Poor (300-579)
            {'mean': 600, 'std': 20, 'weight': 0.15, 'default_prob': 0.20},   # Poor (580-629)
            {'mean': 650, 'std': 15, 'weight': 0.20, 'default_prob': 0.12},   # Below Average (630-679)
            {'mean': 700, 'std': 15, 'weight': 0.25, 'default_prob': 0.06},   # Average (680-719)
            {'mean': 730, 'std': 10, 'weight': 0.15, 'default_prob': 0.03},   # Good (720-739)
            {'mean': 770, 'std': 10, 'weight': 0.10, 'default_prob': 0.01},   # Very Good (740-799)
            {'mean': 820, 'std': 8, 'weight': 0.05, 'default_prob': 0.005}    # Exceptional (800-850)
        ]
        
        fico_scores = []
        defaults = []
        
        for params in fico_params:
            n_tier = int(n_samples * params['weight'])
            # Generate scores with tighter bounds to ensure they stay within range
            tier_scores = np.random.normal(params['mean'], params['std'], n_tier)
            # More aggressive clipping with buffer
            tier_scores = np.clip(tier_scores, 350, 840).astype(int)
            tier_defaults = np.random.binomial(1, params['default_prob'], n_tier)
            
            fico_scores.extend(tier_scores)
            defaults.extend(tier_defaults)
        
        # Create DataFrame and ensure all scores are valid
        df = pd.DataFrame({'fico_score': fico_scores, 'default_flag': defaults})
        
        # Final validation - remove any scores that somehow got outside bounds
        df = df[(df['fico_score'] >= 300) & (df['fico_score'] <= 850)]
        
        return df

def main():
    st.title("Advanced FICO Score Bucketing Analysis")
    st.markdown("Optimize credit score buckets using vectorized algorithms and smart optimization")
    
    # Initialize tool
    if 'bucketing_tool' not in st.session_state:
        st.session_state.bucketing_tool = OptimizedFICOBucketing()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'detected_columns' not in st.session_state:
        st.session_state.detected_columns = None
    
    tool = st.session_state.bucketing_tool
    
    # Data Upload Section
    st.header("Step 1: Upload Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload any CSV file. The app will automatically detect FICO score and default columns."
        )
    
    with col2:
        if st.button("Use Sample Data", use_container_width=True):
            try:
                df = tool.generate_sample_data()
                st.session_state.data = df
                st.session_state.detected_columns = {'fico': 'fico_score', 'default': 'default_flag'}
                
                # Show data validation info
                st.success(f"Generated {len(df)} sample records")
                st.info(f"FICO Range: {df['fico_score'].min()}-{df['fico_score'].max()}, "
                       f"Default Rate: {(df['default_flag'].mean() * 100):.1f}%")
                
            except Exception as e:
                st.error(f"Error generating sample data: {e}")
    
    if uploaded_file is not None:
        try:
            # Read the file content
            text = uploaded_file.getvalue().decode('utf-8')
            st.info(f"File uploaded successfully. Size: {len(text)} characters")
            
            # Parse CSV
            df = tool.parse_csv(text)
            if df is None:
                st.error("Failed to parse CSV file. Please check the file format.")
                return
                
            st.info(f"CSV parsed successfully. Shape: {df.shape}")
            
            # Auto-detect columns
            fico_col, default_col = tool.detect_columns(df)
            
            if not fico_col:
                st.error("""
                **Could not automatically detect FICO score column.** 
                
                Please ensure your data has a column with FICO scores (300-850).
                
                **Common column names:** 
                - fico, score, credit_score, fico_score, creditscore
                - credit, rating, cscore
                
                **Available columns in your file:**
                """ + str(list(df.columns)))
                
                # Let user manually select columns
                st.subheader("Manual Column Selection")
                col1, col2 = st.columns(2)
                with col1:
                    fico_col = st.selectbox("Select FICO Score Column", options=df.columns)
                with col2:
                    default_col = st.selectbox("Select Default Column (optional)", options=["None"] + list(df.columns))
                    if default_col == "None":
                        default_col = None
            else:
                # Show detected columns
                st.session_state.detected_columns = {'fico': fico_col, 'default': default_col}
                
                # Validate and prepare data
                df_clean = tool.validate_and_prepare_data(df, fico_col, default_col)
                st.session_state.data = df_clean
                
                st.success(f"Successfully loaded {len(df_clean)} records")
                
                # Show data preview
                with st.expander("View Data Preview and Detected Columns"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Detected Columns:**")
                        st.write(f"- FICO Score: {fico_col}")
                        st.write(f"- Default: {default_col if default_col else 'Not detected (using zeros)'}")
                    
                    with col2:
                        st.write("**Data Preview:**")
                        st.dataframe(df_clean.head())
                        
                    st.write(f"**Data Summary:** {len(df_clean)} records, FICO range: {df_clean['fico'].min()}-{df_clean['fico'].max()}, "
                            f"Default rate: {(df_clean['default'].mean() * 100):.1f}%")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("""
            **Troubleshooting tips:**
            1. Ensure your CSV has a column with FICO scores (numeric values between 300-850)
            2. Check that the file is a valid CSV format
            3. Make sure the FICO score column has numeric values
            4. Try opening the file in Excel or another spreadsheet program to verify the data
            """)
    
    # Data Quality Alert Section
    if st.session_state.data is not None:
        df = st.session_state.data
        default_rate = df['default'].mean() * 100
        
        # Show data quality alerts
        if default_rate > 30:
            st.error("🚨 DATA QUALITY ALERT: High Default Rate Detected")
            st.warning(f"Current default rate: {default_rate:.1f}%")
            st.warning("""
            **Expected Range:** Typical default rates in credit data are 1-10%
            
            **Possible Issues:**
            - Default column might be coded backwards (1 = Good, 0 = Bad)
            - Different definition of 'default' in your data
            - Data quality issues in the default column
            
            **The system has attempted automatic correction, but please verify your data.**
            """)
        
        elif default_rate == 0:
            st.warning("⚠️ Note: Default rate is 0%. Using zeros for all default data.")
            st.info("This might affect the Log-Likelihood optimization results.")
    
    # Configuration Section
    if st.session_state.data is not None:
        st.header("Step 2: Configure Bucketing")
        
        df = st.session_state.data
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_buckets = st.selectbox(
                "Number of Rating Categories",
                options=[5, 7],
                index=0,
                help="5-category: Standard FICO model | 7-category: Extended granularity"
            )
            
            # Show rating scale explanation
            if num_buckets == 5:
                st.info("""
                **5-Category Scale:**
                - Rating 1: Poor Credit (300-579)
                - Rating 2: Fair Credit (580-669) 
                - Rating 3: Good Credit (670-739)
                - Rating 4: Very Good Credit (740-799)
                - Rating 5: Exceptional Credit (800-850)
                """)
            else:
                st.info("""
                **7-Category Scale:**
                - Rating 1: Very Poor Credit (300-579)
                - Rating 2: Poor Credit (580-629)
                - Rating 3: Below Average Credit (630-679)
                - Rating 4: Average Credit (680-719)
                - Rating 5: Good Credit (720-739)
                - Rating 6: Very Good Credit (740-799)
                - Rating 7: Exceptional Credit (800-850)
                """)
        
        with col2:
            method = st.selectbox(
                "Optimization Method",
                options=['both', 'mse', 'll'],
                format_func=lambda x: {
                    'both': 'Both Methods (Compare)',
                    'mse': 'Mean Squared Error',
                    'll': 'Log-Likelihood (with defaults)'
                }[x]
            )
        
        # Custom CSS for light green button
        st.markdown("""
        <style>
        .stButton>button {
            background-color: #90EE90 !important;
            color: black !important;
            border: 1px solid #32CD32 !important;
        }
        .stButton>button:hover {
            background-color: #32CD32 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Optimal Buckets", use_container_width=True):
            with st.spinner("Running optimized bucketing analysis..."):
                try:
                    fico_scores = df['fico'].values
                    data_points = df[['fico', 'default']].copy()
                    
                    mse_result = None
                    ll_result = None
                    
                    if method in ['mse', 'both']:
                        st.info("Running MSE bucketing...")
                        mse_result = tool.vectorized_mse_bucketing(fico_scores, num_buckets)
                        mse_result['rating_map'] = tool.create_rating_map(mse_result['boundaries'], num_buckets)
                    
                    if method in ['ll', 'both']:
                        st.info("Running Log-Likelihood bucketing...")
                        ll_result = tool.optimized_log_likelihood_bucketing(data_points, num_buckets)
                        ll_result['rating_map'] = tool.create_rating_map(ll_result['boundaries'], num_buckets)
                    
                    st.session_state.results = {
                        'mse': mse_result,
                        'll': ll_result,
                        'data_stats': {
                            'total_records': len(df),
                            'avg_fico': df['fico'].mean(),
                            'default_rate': df['default'].mean(),
                            'fico_range': (df['fico'].min(), df['fico'].max())
                        }
                    }
                    
                    st.success("Bucketing analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during bucketing: {e}")
    
    # Results Section
    if st.session_state.results is not None:
        st.header("Step 3: Analysis Results")
        results = st.session_state.results
        
        # Important note about rating system
        st.info("""
        **Rating System Explanation:** 
        - **Higher Rating Numbers = Better Credit Quality** (Industry Standard)
        - Rating 1 represents the lowest credit scores (highest risk)
        - Rating 5/7 represents the highest credit scores (lowest risk)
        """)
        
        # MSE Results
        if results['mse']:
            mse_result = results['mse']
            
            st.subheader("Mean Squared Error Method")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.metric("Mean Squared Error", f"{mse_result['mse']:.2f}")
            with col2:
                st.metric("Iterations", mse_result['iterations'])
            with col3:
                mse_df = pd.DataFrame(mse_result['rating_map'])
                st.download_button(
                    "Export MSE Map",
                    mse_df.to_csv(index=False),
                    "fico_rating_map_mse.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # Rating map table in correct format
            st.write("**FICO Rating Map**")
            display_df = pd.DataFrame(mse_result['rating_map'])
            st.dataframe(display_df[['Rating', 'FICO Range', 'Description']], use_container_width=True)
            
            # Show risk interpretation
            st.write("**Risk Interpretation:**")
            best_rating = display_df['Rating'].max()
            worst_rating = display_df['Rating'].min()
            st.write(f"- **Rating {best_rating}**: Lowest Risk (Best Credit Quality)")
            st.write(f"- **Rating {worst_rating}**: Highest Risk (Poorest Credit Quality)")
        
        # Log-Likelihood Results
        if results['ll']:
            ll_result = results['ll']
            
            st.subheader("Log-Likelihood Method")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.metric("Log-Likelihood Score", f"{ll_result['log_likelihood']:.2f}")
            with col2:
                ll_df = pd.DataFrame(ll_result['rating_map'])
                st.download_button(
                    "Export LL Map",
                    ll_df.to_csv(index=False),
                    "fico_rating_map_loglikelihood.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # Rating map table with default statistics
            st.write("**FICO Rating Map with Default Statistics**")
            display_data = []
            for rating_info in ll_result['rating_map']:
                # Find the corresponding bucket stats
                bucket_idx = rating_info['Rating'] - 1  # Rating 1 = index 0, etc.
                stats = ll_result['bucket_stats'][bucket_idx]
                display_data.append({
                    'Rating': rating_info['Rating'],
                    'FICO Range': rating_info['FICO Range'],
                    'Description': rating_info['Description'],
                    'Count': stats['ni'],
                    'Defaults': stats['ki'],
                    'Default Rate': f"{(stats['pi'] * 100):.1f}%"
                })
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True)
            
            # Show risk pattern
            if len(display_df) > 0:
                highest_risk_rate = display_df[display_df['Rating'] == display_df['Rating'].min()]['Default Rate'].iloc[0]
                lowest_risk_rate = display_df[display_df['Rating'] == display_df['Rating'].max()]['Default Rate'].iloc[0]
                st.write(f"**Risk Gradient:** Rating {display_df['Rating'].min()} ({highest_risk_rate}) → Rating {display_df['Rating'].max()} ({lowest_risk_rate})")
    
    # Model Performance and Conclusions Section
    if st.session_state.results is not None:
        st.header("Step 4: Model Performance and Conclusions")
        
        results = st.session_state.results
        data_stats = results['data_stats']
        
        # Data Overview
        st.subheader("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", data_stats['total_records'])
        with col2:
            st.metric("Average FICO", f"{data_stats['avg_fico']:.1f}")
        with col3:
            st.metric("Default Rate", f"{(data_stats['default_rate'] * 100):.1f}%")
        with col4:
            st.metric("FICO Range", f"{data_stats['fico_range'][0]}-{data_stats['fico_range'][1]}")
        
        # Data Quality Assessment
        st.subheader("Data Quality Assessment")
        default_rate = data_stats['default_rate'] * 100
        
        if default_rate > 30:
            st.error("⚠️ **Data Quality Issue**: Unusually high default rate detected")
            st.warning("""
            **Recommendations:**
            1. Verify your default column coding (0=Good, 1=Bad)
            2. Check if your data uses a different definition of 'default'
            3. Consider using MSE method instead of Log-Likelihood for this dataset
            """)
        elif default_rate == 0:
            st.warning("⚠️ **Note**: No defaults in dataset")
            st.info("Log-Likelihood method may not provide meaningful results without default variation")
        elif default_rate <= 10:
            st.success("✅ **Good Data Quality**: Default rate within expected range (1-10%)")
        else:
            st.info("ℹ️ **Moderate Default Rate**: Consider verifying data definitions")
        
        # FICO Score Interpretation Guide
        st.subheader("FICO Score Interpretation Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Traditional FICO Ranges:**")
            st.write("- 800-850: Exceptional")
            st.write("- 740-799: Very Good")
            st.write("- 670-739: Good")
            st.write("- 580-669: Fair")
            st.write("- 300-579: Poor")
        
        with col2:
            st.write("**Typical Implications:**")
            st.write("- 750+: Easy loan approval, best rates")
            st.write("- 700-749: Good approval odds, competitive rates")
            st.write("- 650-699: May require higher down payments")
            st.write("- 600-649: Subprime, higher interest rates")
            st.write("- Below 600: Difficult to get approved")
        
        # Model Performance Metrics
        st.subheader("Model Performance Metrics")
        
        if results['mse'] and results['ll']:
            # Performance Comparison
            comparison_data = {
                'Metric': ['Optimization Goal', 'Processing Speed', 'Data Utilization', 
                          'Default Pattern Capture', 'Recommended Use Case'],
                'MSE Method': [
                    'Minimize score approximation error',
                    'Very Fast (vectorized)',
                    'Uses only FICO scores',
                    'No explicit default pattern consideration',
                    'Score approximation, general bucketing'
                ],
                'Log-Likelihood Method': [
                    'Maximize default prediction likelihood',
                    'Fast (optimized)',
                    'Uses FICO scores + default labels',
                    'Explicitly models default patterns',
                    'Risk prediction, PD modeling'
                ]
            }
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Conclusions and Recommendations
        st.subheader("Conclusions and Recommendations")
        
        if results['mse'] and results['ll']:
            # Smart recommendation based on data quality
            default_rate = data_stats['default_rate'] * 100
            
            if default_rate > 30 or default_rate == 0:
                recommendation = """
                **Primary Recommendation: Use MSE Method**
                
                Due to data quality concerns (unrealistic default rate), the **MSE method** is recommended 
                as it doesn't rely on default patterns and provides stable numerical bucketing.
                """
            else:
                recommendation = """
                **Primary Recommendation: Use Log-Likelihood Method**
                
                For **default prediction and risk modeling**, use the **Log-Likelihood method** as it explicitly 
                optimizes bucket boundaries based on actual default patterns, providing better separation 
                between risk categories.
                """
            
            st.success(recommendation)
        
        # Implementation Guidance
        st.subheader("Implementation Guidance")
        
        st.markdown("""
        **Next Steps:**
        1. **Export your preferred rating map** using the download buttons above
        2. **Integrate into your ML pipeline** by replacing raw FICO scores with bucket ratings
        3. **Monitor model performance** and consider re-bucketing if portfolio characteristics change significantly
        
        **Expected Benefits:**
        - Improved model interpretability through categorical features
        - Better handling of non-linear FICO score effects
        - Reduced overfitting compared to using raw scores
        - More stable model performance across different populations
        
        **Remember:** Higher rating numbers indicate better credit quality and lower risk.
        """)

if __name__ == "__main__":
    main()