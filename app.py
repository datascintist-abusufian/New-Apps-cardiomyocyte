import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import umap.umap_ as umap
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import re

# --------------------------
# Safe fallbacks for newer Streamlit APIs used in this app
# --------------------------
if not hasattr(st, "fragment"):
    # Streamlit <= 1.31: make decorator a no-op
    def _fragment(func):
        return func
    st.fragment = _fragment

# Try importing nd2reader for .nd2 files
try:
    from nd2reader import ND2Reader
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False
    print("Warning: nd2reader not installed. Install with: pip install nd2reader")

from analyzers.aging_analyzer import AdvancedCardiomyocyteAnalyzer
from config import SUPPORTED_IMAGE_FORMATS, SUPPORTED_VIDEO_FORMATS

# Update configuration for 300MB limit and add .nd2
SUPPORTED_IMAGE_FORMATS_EXTENDED = list(SUPPORTED_IMAGE_FORMATS) + ['nd2']
MAX_FILE_SIZE_MB = 300  # Increased from 200MB

# Set page config
st.set_page_config(
    page_title="🧬 Cardiomyocyte Aging Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Best-effort server.maxUploadSize (private API; guard to avoid crashes)
try:
    if hasattr(st, "_config"):
        st._config.set_option('server.maxUploadSize', MAX_FILE_SIZE_MB)
except Exception:
    pass

def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    if 'plot_cache' not in st.session_state:
        st.session_state.plot_cache = {}
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'last_files' not in st.session_state:
        st.session_state.last_files = []
    if 'last_sensitivity' not in st.session_state:
        st.session_state.last_sensitivity = 1.0
    if 'temporal_data' not in st.session_state:
        st.session_state.temporal_data = {}
    if 'video_cache' not in st.session_state:
        st.session_state.video_cache = {}

@st.cache_data
def configure_streamlit():
    """Configure Streamlit custom CSS."""
    return """
        <style>
            .main-header { font-size: 2.5em; color: #1f77b4; text-align: center; margin-bottom: 20px; }
            .aging-score { font-size: 2em; font-weight: bold; text-align: center; padding: 15px; border-radius: 8px; margin: 10px 0; }
            .critical-marker { background-color: #ffe6e6; padding: 8px; border-radius: 4px; border-left: 3px solid #d62728; margin: 5px 0; }
            .tab-content { padding: 15px; background-color: #ffffff; border-radius: 6px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
            .stSpinner > div { text-align: center; }
            .temporal-header { font-size: 1.2em; color: #2c3e50; font-weight: bold; margin: 10px 0; }
        </style>
    """

@st.cache_resource
def get_analyzer():
    """Get cached analyzer instance."""
    return AdvancedCardiomyocyteAnalyzer()

def read_nd2_file(file_path: str) -> Optional[np.ndarray]:
    """Read ND2 file and return as numpy array."""
    if not ND2_AVAILABLE:
        raise ImportError("nd2reader is required for .nd2 files. Install with: pip install nd2reader")
    try:
        with ND2Reader(file_path) as images:
            # Pick a representative frame: time>0, else middle z, else first
            if images.sizes.get('t', 1) > 1:
                frame = np.array(images[0])
            elif images.sizes.get('z', 1) > 1:
                zmid = int(images.sizes['z'] // 2)
                frame = np.array(images[zmid])
            else:
                frame = np.array(images[0])

            # Normalize to uint8
            if frame.dtype != np.uint8:
                fmin, fmax = float(frame.min()), float(frame.max())
                if fmax > fmin:
                    frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            return frame
    except Exception as e:
        st.error(f"Error reading ND2 file: {str(e)}")
        return None

def extract_day_from_filename(filename: str) -> Optional[int]:
    """Extract day number from filename (e.g., 'sample_day3.jpg' -> 3)."""
    patterns = [
        r'day[\s_-]?(\d+)',      # day1, day_1, day-1, day 1
        r'(?:^|[_\W])d(\d+)(?:[_\W]|$)',  # d1, d-1 etc (safer word boundaries)
        r'[\s_-](\d+)[\s_-]?d',  # 1d, 1_d, 1-d
    ]
    low = filename.lower()
    for p in patterns:
        m = re.search(p, low)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None

def get_file_hash(file):
    """Generate hash for file content with size check."""
    file_size = file.size / (1024 * 1024)  # MB
    if file_size > MAX_FILE_SIZE_MB:
        st.error(f"File {file.name} exceeds {MAX_FILE_SIZE_MB}MB limit ({file_size:.1f}MB)")
        return None
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

def create_sidebar() -> Dict[str, Any]:
    """Create sidebar with analysis parameters and visualization options."""
    with st.sidebar:
        st.header("🔧 Parameters")
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Comprehensive", "Morphology", "Sarcomere", "Mitochondrial", "Motion"]
        )
        sensitivity = st.slider("Sensitivity", 0.1, 2.0, 1.0, 0.1)

        st.header("🔎 Filters")
        stage_filter = st.multiselect("Filter by Stage (optional)", [], help="Will populate after analysis.")
        day_filter = st.multiselect("Filter by Day (optional)", [], help="Will populate after analysis.")

        st.header("📊 Visualizations")
        options = {
            'show_preprocessing': st.checkbox("Preprocessed Image", True),
            'show_feature_radar': st.checkbox("Feature Radar Plot", True),
            'show_comparisons': st.checkbox("Reference Comparisons", True),
            'show_biomarkers': st.checkbox("Top Biomarkers", True),
            'show_motion': st.checkbox("Motion Analysis", True),
            'show_morphology': st.checkbox("Morphology Analysis", True),
            'show_temporal': st.checkbox("Temporal Pattern Analysis", False),
            'analysis_depth': analysis_depth,
            'sensitivity': sensitivity,
            'stage_filter': stage_filter,
            'day_filter': day_filter
        }

        # Temporal analysis settings
        if options['show_temporal']:
            st.subheader("📅 Temporal Settings")
            options['temporal_days'] = st.multiselect(
                "Select Days for Analysis",
                options=list(range(1, 9)),
                default=list(range(1, 9))
            )
        return options

def create_file_upload_section() -> tuple[List, bool, bool]:
    """Create file upload section with unlimited uploads."""
    col1, col2 = st.columns(2)
    with col1:
        st.header("📁 Upload Files")
        st.info(f"✅ Unlimited files | Max size: {MAX_FILE_SIZE_MB}MB each | Formats: JPG, PNG, TIFF, ND2, MP4, AVI")
        uploaded_files = st.file_uploader(
            "Upload Images/Videos",
            type=SUPPORTED_IMAGE_FORMATS_EXTENDED + SUPPORTED_VIDEO_FORMATS,
            accept_multiple_files=True,
            help=f"Upload unlimited files up to {MAX_FILE_SIZE_MB}MB each"
        )
    with col2:
        st.header("📊 Analysis Options")
        c1, c2 = st.columns(2)
        with c1:
            analysis_button = st.button("🔬 Analyze All", type="primary", disabled=not uploaded_files)
        with c2:
            temporal_button = st.button(
                "📅 Temporal Analysis",
                type="secondary",
                disabled=not uploaded_files,
                help="Analyze day 1-8 temporal patterns"
            )

        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} file(s) ready for analysis")
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.info(f"Total size: {total_size:.1f}MB")
        else:
            st.info("Upload images or videos to start analysis")
    return uploaded_files or [], analysis_button, temporal_button

def process_video_optimized(video_path: str, max_frames: int = 10) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """Process video with optimized frame sampling."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            return None, []
        frames_to_read = list(range(total_frames)) if total_frames <= max_frames else np.linspace(0, total_frames-1, max_frames, dtype=int)
        for frame_idx in frames_to_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames[0] if frames else None, frames

def process_single_file(analyzer, file_data, temp_dir, sensitivity, cache_dict):
    """Process a single file in parallel with ND2 support."""
    file, file_hash = file_data
    if file_hash is None:
        return None

    # Check cache first
    cache_key = f"{file_hash}_{sensitivity}"
    if cache_key in cache_dict:
        return cache_dict[cache_key]

    try:
        file_ext = file.name.lower().rsplit('.', 1)[-1]
        if file_ext not in SUPPORTED_IMAGE_FORMATS_EXTENDED + SUPPORTED_VIDEO_FORMATS:
            return None

        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            file.seek(0)
            f.write(file.read())

        # Handle different file types
        is_video = False
        if file_ext == 'nd2':
            image = read_nd2_file(file_path)
            if image is None:
                return None
            temp_tiff = os.path.join(temp_dir, f"{file.name}.tiff")
            cv2.imwrite(temp_tiff, image)
            file_path = temp_tiff
        elif file_ext in SUPPORTED_VIDEO_FORMATS:
            is_video = True
            first_frame, frames = process_video_optimized(file_path)
            if first_frame is None:
                return None

        day = extract_day_from_filename(file.name)

        # Analyzer returns: features (dict), orig (ndarray), proc (ndarray|None),
        # vit_features (np.ndarray or list), motion_signals (list|None), contours (list|None)
        features, orig, proc, vit_features, motion_signals, contours = analyzer.comprehensive_aging_analysis(
            file_path, is_video, sensitivity
        )
        if not features or orig is None:
            return None

        score, feature_names, lime_values, stage, stage_color, critical, confidence = analyzer.calculate_advanced_aging_score(
            features, vit_features
        )

        # Ensure safe image type
        if isinstance(orig, np.ndarray) and orig.ndim == 2:
            display_img = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        else:
            display_img = orig

        # Normalize lime_values shape: wrap to list for consistency
        if isinstance(lime_values, (np.ndarray, list)):
            lv = np.array(lime_values).tolist()
        else:
            lv = [lime_values]

        result = {
            'name': file.name,
            'features': features,
            'vit_features': vit_features,
            'score': float(score),
            'stage': stage,
            'stage_color': stage_color,
            'critical': ", ".join(critical) if critical else "None",
            'lime_values': lv,
            'confidence': float(confidence) if isinstance(confidence, (int, float)) else confidence,
            'motion_signals': motion_signals or [],
            'contours': contours or [],
            'processed_image': proc,
            'feature_names': feature_names or [],
            'image': display_img,
            'day': day
        }
        return result
    except Exception as e:
        return {'error': f"Error processing {file.name}: {str(e)}"}

def process_uploaded_files_parallel(analyzer: AdvancedCardiomyocyteAnalyzer, uploaded_files: List, sensitivity: float) -> Dict[str, Any]:
    """Process uploaded files in parallel with no limit."""
    results = {
        'features': [], 'vit_features': [], 'names': [], 'scores': [], 'stages': [],
        'stage_colors': [], 'critical': [], 'lime_values': [], 'confidences': [],
        'images': [], 'motion_signals': [], 'contours': [], 'processed_images': [],
        'feature_names': None, 'days': []
    }

    cache_dict = dict(st.session_state.analysis_cache) if 'analysis_cache' in st.session_state else {}

    file_hashes = []
    for f in uploaded_files:
        h = get_file_hash(f)
        if h:
            file_hashes.append((f, h))
    if not file_hashes:
        st.error("No valid files to process")
        return results

    with tempfile.TemporaryDirectory() as temp_dir:
        progress_bar = st.progress(0)
        progress_text = st.empty()
        max_workers = min(8, max(4, len(file_hashes) // 2))
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_single_file, analyzer, file_data, temp_dir, sensitivity, cache_dict): idx
                for idx, file_data in enumerate(file_hashes)
            }
            completed = 0
            for future in as_completed(future_map):
                completed += 1
                progress_bar.progress(completed / len(file_hashes))
                progress_text.text(f"Processing file {completed}/{len(file_hashes)}...")
                result = future.result()
                if result and 'error' not in result:
                    results['features'].append(result['features'])
                    results['vit_features'].append(result['vit_features'])
                    results['names'].append(result['name'])
                    results['scores'].append(result['score'])
                    results['stages'].append(result['stage'])
                    results['stage_colors'].append(result['stage_color'])
                    results['critical'].append(result['critical'])
                    results['lime_values'].append(result['lime_values'])
                    results['confidences'].append(result['confidence'])
                    results['motion_signals'].append(result['motion_signals'])
                    results['contours'].append(result['contours'])
                    results['processed_images'].append(result['processed_image'])
                    results['images'].append(result['image'])
                    results['days'].append(result.get('day'))
                    if result['feature_names']:
                        results['feature_names'] = result['feature_names']

                    # Update session cache
                    fhash = file_hashes[future_map[future]][1]
                    st.session_state.analysis_cache[f"{fhash}_{sensitivity}"] = result
                elif result and 'error' in result:
                    errors.append(result['error'])

        progress_text.empty()
        if errors:
            with st.expander(f"⚠️ Processing Warnings ({len(errors)})", expanded=False):
                for e in errors[:10]:
                    st.warning(e)
    return results

def create_feature_radar_plot(features: Dict, feature_names: List[str], name: str) -> go.Figure:
    """Create radar plot for feature visualization."""
    radar_features = [
        'sarcomere_regularity', 'mitochondrial_density', 'motion_entropy',
        'motion_vector_magnitude', 'circularity_standard', 'aspect_ratio',
        'solidity', 'shape_complexity'
    ]
    values, labels = [], []
    for feat in radar_features:
        if feat in features:
            values.append(float(features.get(feat, 0.0)))
            labels.append(feat.replace('_', ' ').title())

    if values:
        vmin, vmax = min(values), max(values)
        if vmax > vmin:
            values = [(v - vmin) / (vmax - vmin) for v in values]

    fig = go.Figure(data=go.Scatterpolar(r=values, theta=labels, fill='toself', name=name))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title=f"Feature Radar - {name}", height=400
    )
    return fig

def create_temporal_analysis(results: Dict[str, Any], selected_days: List[int]) -> Dict[str, go.Figure]:
    """Create temporal pattern analysis for days 1-8."""
    temporal_figs = {}
    day_data = {day: {'diameter': [], 'circularity': [], 'aspect_ratio': [], 'names': [], 'scores': []}
                for day in selected_days}

    for idx, day in enumerate(results['days']):
        if day and day in selected_days:
            features = results['features'][idx]
            area = float(features.get('area', 0) or 0.0)
            diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0.0
            day_data[day]['diameter'].append(diameter)
            day_data[day]['circularity'].append(float(features.get('circularity_standard', 0.0)))
            day_data[day]['aspect_ratio'].append(float(features.get('aspect_ratio', 0.0)))
            day_data[day]['names'].append(results['names'][idx])
            day_data[day]['scores'].append(float(results['scores'][idx]))

    avg_data = {
        'days': [], 'avg_diameter': [], 'std_diameter': [],
        'avg_circularity': [], 'std_circularity': [],
        'avg_aspect_ratio': [], 'std_aspect_ratio': [],
        'sample_count': [], 'avg_score': []
    }

    for day in sorted(selected_days):
        if day_data[day]['diameter']:
            avg_data['days'].append(day)
            for key, agg in [
                ('avg_diameter', np.mean(day_data[day]['diameter'])),
                ('std_diameter', np.std(day_data[day]['diameter'])),
                ('avg_circularity', np.mean(day_data[day]['circularity'])),
                ('std_circularity', np.std(day_data[day]['circularity'])),
                ('avg_aspect_ratio', np.mean(day_data[day]['aspect_ratio'])),
                ('std_aspect_ratio', np.std(day_data[day]['aspect_ratio'])),
                ('avg_score', np.mean(day_data[day]['scores']))
            ]:
                avg_data[key].append(float(agg))
            avg_data['sample_count'].append(int(len(day_data[day]['diameter'])))

    if not avg_data['days']:
        return temporal_figs

    # Main 3-row figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Average Diameter vs Day', 'Average Circularity vs Day', 'Aspect Ratio vs Day'),
        vertical_spacing=0.12
    )
    fig.add_trace(go.Scatter(
        x=avg_data['days'], y=avg_data['avg_diameter'], mode='lines+markers',
        name='Avg Diameter', error_y=dict(type='data', array=avg_data['std_diameter'])
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=avg_data['days'], y=avg_data['avg_circularity'], mode='lines+markers',
        name='Avg Circularity', error_y=dict(type='data', array=avg_data['std_circularity'])
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=avg_data['days'], y=avg_data['avg_aspect_ratio'], mode='lines+markers',
        name='Avg Aspect Ratio', error_y=dict(type='data', array=avg_data['std_aspect_ratio'])
    ), row=3, col=1)
    fig.update_xaxes(title_text="Day", row=3, col=1)
    fig.update_yaxes(title_text="Diameter (px)", row=1, col=1)
    fig.update_yaxes(title_text="Circularity", row=2, col=1)
    fig.update_yaxes(title_text="Aspect Ratio", row=3, col=1)
    fig.update_layout(height=900, title_text="Temporal Pattern Analysis (Day 1-8)", showlegend=False, hovermode='x unified')
    temporal_figs['main'] = fig

    # Sample count
    fig_samples = go.Figure(data=[go.Bar(
        x=avg_data['days'], y=avg_data['sample_count'], text=avg_data['sample_count'], textposition='auto'
    )])
    fig_samples.update_layout(title="Samples per Day", xaxis_title="Day", yaxis_title="Number of Samples", height=300)
    temporal_figs['samples'] = fig_samples

    # NEW: Score trend with linear fit
    if len(avg_data['days']) >= 2:
        x = np.array(avg_data['days'], dtype=float)
        y = np.array(avg_data['avg_score'], dtype=float)
        try:
            m, b = np.polyfit(x, y, deg=1)
            yfit = m * x + b
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', name='Avg Score'))
            fig_trend.add_trace(go.Scatter(x=x, y=yfit, mode='lines', name=f'Linear Fit (slope={m:.2f})'))
            fig_trend.update_layout(title="Average Score vs Day (with linear trend)", xaxis_title="Day", yaxis_title="Avg Score", height=350)
            temporal_figs['score_trend'] = fig_trend
        except Exception:
            pass

    return temporal_figs

@lru_cache(maxsize=32)
def create_cluster_analysis_cached(features_hash: str, vit_features_hash: str, names_tuple: tuple,
                                  scores_tuple: tuple, stages_tuple: tuple, confidences_tuple: tuple) -> pd.DataFrame:
    """Cached cluster analysis."""
    try:
        features = pickle.loads(bytes.fromhex(features_hash))
        vit_features = pickle.loads(bytes.fromhex(vit_features_hash))

        feature_df = pd.DataFrame(features).select_dtypes(include=[np.number]).fillna(0)
        if feature_df.empty:
            return pd.DataFrame()
        vit_arr = np.array(vit_features, dtype=float)
        if vit_arr.ndim > 2:
            vit_arr = vit_arr.reshape(len(vit_arr), -1)
        elif vit_arr.ndim == 1:
            vit_arr = vit_arr.reshape(-1, 1)

        min_features = min(feature_df.shape[1], vit_arr.shape[1]) if vit_arr.size else feature_df.shape[1]
        if vit_arr.size and min_features > 0:
            merged = np.hstack([feature_df.values[:, :min_features], vit_arr[:, :min_features]])
        else:
            merged = feature_df.values

        if merged.shape[0] < 2:
            return pd.DataFrame()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(merged)

        n_neighbors = max(2, min(15, scaled.shape[0] - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(scaled)

        n_clusters = min(3, scaled.shape[0])
        # Use integer n_init for older scikit-learn compatibility
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)

        return pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'Cluster': labels.astype(str),
            'Sample': list(names_tuple),
            'Score': list(scores_tuple),
            'Stage': list(stages_tuple),
            'Confidence': list(confidences_tuple)
        })
    except Exception as e:
        st.error(f"Cluster analysis error: {str(e)}")
        return pd.DataFrame()

def create_cluster_analysis(results: Dict[str, Any]) -> pd.DataFrame:
    """Wrapper for cluster analysis with caching."""
    if len(results['features']) < 2:
        return pd.DataFrame()
    features_hash = pickle.dumps(results['features']).hex()
    vit_features_hash = pickle.dumps(results['vit_features']).hex()
    return create_cluster_analysis_cached(
        features_hash,
        vit_features_hash,
        tuple(results['names']),
        tuple(results['scores']),
        tuple(results['stages']),
        tuple(results['confidences'])
    )

# --------------------------
# DISPLAY SECTIONS (existing)
# --------------------------
@st.fragment
def display_single_file_analysis(results: Dict[str, Any], options: Dict[str, Any]):
    st.header("🔬 Single File Analysis")
    img_col, metrics_col = st.columns([2, 1])

    with img_col:
        st.image(results['images'][0], caption="Uploaded Image/Video Frame", use_container_width=True)
        if options['show_feature_radar']:
            radar_fig = create_feature_radar_plot(
                results['features'][0], results['feature_names'] or [], results['names'][0]
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    with metrics_col:
        features, score, stage, stage_color, critical, confidence = (
            results['features'][0], results['scores'][0], results['stages'][0],
            results['stage_colors'][0], results['critical'][0].split(", ") if results['critical'][0] != "None" else [],
            results['confidences'][0]
        )
        st.markdown(f"""
        <div class="aging-score" style="background-color: {stage_color}20; border: 2px solid {stage_color};">
            <div style="color: {stage_color};">Aging Score: {score:.1f}/100</div>
            <div style="font-size: 0.6em; color: {stage_color};">{stage}</div>
            <div style="font-size: 0.5em;">Confidence: {float(confidence):.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        if critical:
            st.subheader("⚠️ Critical Markers")
            for marker in critical[:3]:
                st.markdown(f'<div class="critical-marker">{marker}</div>', unsafe_allow_html=True)

        st.subheader("📊 Key Metrics")
        metrics = {
            'Sarcomere': float(features.get('sarcomere_regularity', 0)),
            'Mitochondrial': float(features.get('mitochondrial_density', 0)),
            'Motion Entropy': float(features.get('motion_entropy', 0)),
            'Motion Vector': float(features.get('motion_vector_magnitude', 0))
        }
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.3f}")

    with st.expander("📊 Detailed Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if options['show_preprocessing'] and results['processed_images'][0] is not None:
                st.subheader("🔧 Preprocessed")
                st.image(results['processed_images'][0], caption="Enhanced Image", use_container_width=True)
        with col2:
            st.subheader("📊 Additional Metrics")
            add_metrics = {
                'Circularity': float(features.get('circularity_standard', 0)),
                'Aspect Ratio': float(features.get('aspect_ratio', 0)),
                'Solidity': float(features.get('solidity', 0)),
                'Complexity': float(features.get('shape_complexity', 0))
            }
            for metric, value in add_metrics.items():
                st.metric(metric, f"{value:.3f}")
        with col3:
            st.subheader("🎯 Risk Assessment")
            risk_level = "High" if results['stages'][0] in ['Moderate Aging', 'Advanced Aging'] else ("Moderate" if results['stages'][0] == 'Early Aging' else "Low")
            risk_color = "#d62728" if risk_level == "High" else "#ff7f0e" if risk_level == "Moderate" else "#2ca02c"
            st.markdown(f"""
            <div style="padding: 10px; background-color: {risk_color}20; border-radius: 6px; border-left: 3px solid {risk_color};">
                <h4 style="color: {risk_color}; margin: 0;">Risk: {risk_level}</h4>
            </div>
            """, unsafe_allow_html=True)

    if options['show_biomarkers'] and results['feature_names']:
        with st.expander("🧬 Top Biomarkers", expanded=False):
            plot_key = f"biomarkers_{results['names'][0]}"
            if plot_key not in st.session_state.plot_cache:
                lv0 = results['lime_values'][0]
                lv0 = lv0 if isinstance(lv0, list) else [lv0]
                topk = min(5, len(results['feature_names']))
                lime_df = pd.DataFrame({
                    'Feature': results['feature_names'][:topk],
                    'LIME Value': lv0[:topk]
                }).sort_values(by='LIME Value', ascending=False)
                fig_lime = px.bar(lime_df, x='Feature', y='LIME Value', title="Top Biomarkers", color='LIME Value', color_continuous_scale='Reds')
                fig_lime.update_layout(xaxis_tickangle=45, height=350)
                st.session_state.plot_cache[plot_key] = fig_lime
            st.plotly_chart(st.session_state.plot_cache[plot_key], use_container_width=True)

@st.fragment
def display_multiple_file_analysis(results: Dict[str, Any], options: Dict[str, Any]):
    st.header(f"📊 Multiple File Analysis ({len(results['names'])} files)")

    # Apply filters if specified (sidebar gets populated after first run)
    names, scores, stages, confidences, critical, days, images = (
        results['names'], results['scores'], results['stages'], results['confidences'],
        results['critical'], results['days'], results['images']
    )

    # Populate sidebar filters dynamically
    with st.sidebar:
        st.session_state.available_stages = sorted(list(set(stages)))
        st.session_state.available_days = sorted([d for d in set(days) if d is not None])
        if not st.session_state.get('stage_filter_initialized', False):
            st.session_state.stage_filter_initialized = True

    mask = np.ones(len(names), dtype=bool)
    if options.get('stage_filter'):
        mask &= np.array([s in options['stage_filter'] for s in stages])
    if options.get('day_filter'):
        mask &= np.array([(d in options['day_filter']) for d in days])

    filt_idx = np.where(mask)[0]
    if len(filt_idx) == 0:
        st.info("No samples match the selected filters.")
        return

    # Summary statistics
    fscores = np.array(scores)[filt_idx]
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Avg Score", f"{np.mean(fscores):.1f}")
    with col2: st.metric("Max Score", f"{np.max(fscores):.1f}")
    with col3: st.metric("Min Score", f"{np.min(fscores):.1f}")
    with col4: st.metric("Std Dev", f"{np.std(fscores):.2f}")

    # Radar comparison
    if options['show_feature_radar'] and len(filt_idx) > 1:
        with st.expander("📊 Feature Radar Comparison", expanded=True):
            fig = go.Figure()
            comparison_limit = min(5, len(filt_idx))
            radar_features = ['sarcomere_regularity', 'mitochondrial_density', 'motion_entropy',
                              'motion_vector_magnitude', 'circularity_standard', 'aspect_ratio']
            for idx in filt_idx[:comparison_limit]:
                vals, labels = [], []
                for feat in radar_features:
                    vals.append(float(results['features'][idx].get(feat, 0.0)))
                    labels.append(feat.replace('_', ' ').title())
                if vals:
                    vmin, vmax = min(vals), max(vals)
                    if vmax > vmin:
                        vals = [(v - vmin) / (vmax - vmin) for v in vals]
                fig.add_trace(go.Scatterpolar(r=vals, theta=labels, fill='toself', name=results['names'][idx][:20]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Feature Radar Comparison", height=500)
            st.plotly_chart(fig, use_container_width=True)

    # Table with pagination
    with st.expander("📋 Detailed Results Table", expanded=True):
        items_per_page = 20
        total_pages = (len(filt_idx) - 1) // items_per_page + 1
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀ Previous", disabled=(st.session_state.current_page == 1)):
                st.session_state.current_page -= 1
        with c2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with c3:
            if st.button("Next ▶", disabled=(st.session_state.current_page == total_pages)):
                st.session_state.current_page += 1

        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filt_idx))
        sel = filt_idx[start_idx:end_idx]
        results_df = pd.DataFrame({
            'Sample': [names[i] for i in sel],
            'Score': [scores[i] for i in sel],
            'Stage': [stages[i] for i in sel],
            'Confidence': [f"{float(confidences[i]):.1%}" for i in sel],
            'Critical': [critical[i] for i in sel],
            'Day': [days[i] if days[i] else 'N/A' for i in sel]
        })
        st.dataframe(results_df, use_container_width=True, height=400)

    # Images grid
    with st.expander("📸 Sample Images", expanded=False):
        display_limit = min(12, len(filt_idx))
        if len(filt_idx) > display_limit:
            st.info(f"Showing first {display_limit} of {len(filt_idx)} files")
        cols = st.columns(4)
        for j, idx in enumerate(filt_idx[:display_limit]):
            with cols[j % 4]:
                st.image(images[idx], caption=f"{names[idx][:15]}...", use_container_width=True)
                st.caption(f"Score: {scores[idx]:.1f}")

@st.fragment
def display_temporal_analysis(results: Dict[str, Any], options: Dict[str, Any]):
    if not options['show_temporal']:
        return
    st.header("📅 Temporal Pattern Analysis (Day 1-8)")
    days_with_data = [d for d in results['days'] if d is not None]
    if not days_with_data:
        st.warning("No day information found in filenames. Please include 'day1', 'day2', etc. in filenames.")
        return

    selected_days = options.get('temporal_days', list(range(1, 9)))
    temporal_figs = create_temporal_analysis(results, selected_days)
    if not temporal_figs:
        st.warning("Insufficient data for temporal analysis")
        return

    if 'main' in temporal_figs:
        st.plotly_chart(temporal_figs['main'], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if 'samples' in temporal_figs:
            st.plotly_chart(temporal_figs['samples'], use_container_width=True)
    with col2:
        st.subheader("📊 Temporal Statistics")
        temporal_stats = []
        for day in selected_days:
            day_indices = [i for i, d in enumerate(results['days']) if d == day]
            if day_indices:
                day_scores = [results['scores'][i] for i in day_indices]
                temporal_stats.append({
                    'Day': day,
                    'Count': len(day_indices),
                    'Avg Score': float(np.mean(day_scores)),
                    'Std Score': float(np.std(day_scores))
                })
        if temporal_stats:
            stats_df = pd.DataFrame(temporal_stats)
            st.dataframe(stats_df, use_container_width=True, height=300)

    if 'score_trend' in temporal_figs:
        st.plotly_chart(temporal_figs['score_trend'], use_container_width=True)

@st.fragment
def display_cluster_analysis(cluster_df: pd.DataFrame, results: Dict[str, Any], options: Dict[str, Any]):
    if cluster_df.empty:
        st.info("Cluster analysis requires at least 2 samples")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗺️ UMAP Clustering")
        plot_key = "umap_plot"
        if plot_key not in st.session_state.plot_cache:
            fig_umap = px.scatter(
                cluster_df, x='UMAP1', y='UMAP2', color='Cluster', size='Score',
                hover_data=['Sample', 'Score', 'Stage', 'Confidence'], symbol='Stage'
            )
            fig_umap.update_layout(height=400, title="UMAP Embedding")
            st.session_state.plot_cache[plot_key] = fig_umap
        st.plotly_chart(st.session_state.plot_cache[plot_key], use_container_width=True)
    with col2:
        st.subheader("📉 Cluster Summary")
        cluster_summary = cluster_df.groupby('Cluster')['Score'].agg(['mean', 'min', 'max', 'count']).reset_index()
        cluster_summary.columns = ['Cluster', 'Mean Score', 'Min Score', 'Max Score', 'Count']
        st.dataframe(cluster_summary, use_container_width=True, height=220)

        if options['show_biomarkers'] and results['feature_names'] and len(results['lime_values']) > 0:
            # Mean across samples (pad to same length)
            maxk = min(5, len(results['feature_names']))
            lv_mat = []
            for lv in results['lime_values']:
                lv = lv if isinstance(lv, list) else [lv]
                if len(lv) < maxk:
                    lv = lv + [0.0] * (maxk - len(lv))
                lv_mat.append(lv[:maxk])
            lime_df = pd.DataFrame({
                'Feature': results['feature_names'][:maxk],
                'Mean LIME': np.mean(np.array(lv_mat), axis=0)
            }).sort_values(by='Mean LIME', ascending=False)
            fig_lime = px.bar(lime_df, x='Feature', y='Mean LIME', title="Top Dataset Biomarkers", color='Mean LIME', color_continuous_scale='Reds')
            fig_lime.update_layout(xaxis_tickangle=45, height=250)
            st.plotly_chart(fig_lime, use_container_width=True)

@st.fragment
def display_motion_analysis(results: Dict[str, Any], options: Dict[str, Any]):
    if not options['show_motion'] or not any(results['motion_signals']):
        return
    st.subheader("🏃‍♂️ Motion Analysis")
    st.markdown("Motion features capture dynamic behaviors like contraction irregularity.")

    motion_features = ['motion_entropy', 'motion_variability', 'motion_smoothness',
                       'motion_amplitude_std', 'motion_vector_magnitude']

    items_per_page = 5
    total_items = len([s for s in results['motion_signals'] if s])
    if total_items == 0:
        st.info("No motion data available")
        return

    total_pages = (total_items - 1) // items_per_page + 1
    if 'motion_page' not in st.session_state:
        st.session_state.motion_page = 1

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Prev", key="motion_prev", disabled=(st.session_state.motion_page == 1)):
            st.session_state.motion_page -= 1
    with col2:
        st.write(f"Page {st.session_state.motion_page} of {total_pages}")
    with col3:
        if st.button("Next ▶", key="motion_next", disabled=(st.session_state.motion_page == total_pages)):
            st.session_state.motion_page += 1

    start_idx = (st.session_state.motion_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(results['names']))

    for idx in range(start_idx, end_idx):
        if idx >= len(results['motion_signals']) or not results['motion_signals'][idx]:
            continue
        with st.expander(f"📊 {results['names'][idx]}", expanded=(idx == start_idx)):
            c1, c2 = st.columns(2)
            with c1:
                plot_key = f"motion_signal_{idx}"
                if plot_key not in st.session_state.plot_cache:
                    fig_motion = go.Figure()
                    for i, signal in enumerate(results['motion_signals'][idx][:2]):
                        sig = np.array(signal).astype(float)
                        fig_motion.add_trace(go.Scatter(x=np.arange(len(sig)), y=sig, mode='lines', name=f'Signal {i+1}'))
                    fig_motion.update_layout(title="Motion Signals", xaxis_title="Time", yaxis_title="Intensity", height=250)
                    st.session_state.plot_cache[plot_key] = fig_motion
                st.plotly_chart(st.session_state.plot_cache[plot_key], use_container_width=True)
            with c2:
                features = results['features'][idx]
                motion_data = {k: float(features.get(k, 0.0)) for k in motion_features}
                fig_hist = px.bar(x=list(motion_data.keys()), y=list(motion_data.values()), title="Motion Features")
                fig_hist.update_layout(xaxis_tickangle=45, height=250, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

@st.fragment
def display_morphology_analysis(results: Dict[str, Any], options: Dict[str, Any]):
    if not options['show_morphology']:
        return
    st.subheader("🧬 Morphology Analysis")
    st.markdown("Morphology features describe cell shape and structure.")

    morphology_features = ['circularity_standard', 'aspect_ratio', 'solidity', 'shape_complexity', 'compactness']

    items_per_page = 5
    total_pages = (len(results['names']) - 1) // items_per_page + 1
    if 'morph_page' not in st.session_state:
        st.session_state.morph_page = 1

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Prev", key="morph_prev", disabled=(st.session_state.morph_page == 1)):
            st.session_state.morph_page -= 1
    with col2:
        st.write(f"Page {st.session_state.morph_page} of {total_pages}")
    with col3:
        if st.button("Next ▶", key="morph_next", disabled=(st.session_state.morph_page == total_pages)):
            st.session_state.morph_page += 1

    start_idx = (st.session_state.morph_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(results['names']))

    for idx in range(start_idx, end_idx):
        with st.expander(f"🔬 {results['names'][idx]}", expanded=(idx == start_idx)):
            c1, c2 = st.columns(2)
            with c1:
                if idx < len(results['contours']) and results['contours'][idx]:
                    contour_img = results['images'][idx].copy()
                    if isinstance(contour_img, np.ndarray) and contour_img.ndim == 2:
                        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)
                    try:
                        cv2.drawContours(contour_img, [max(results['contours'][idx], key=cv2.contourArea)], -1, (255, 0, 0), 2)
                    except Exception:
                        pass
                    st.image(contour_img, caption="Cell Contour", use_container_width=True)
            with c2:
                features = results['features'][idx]
                morph_data = {k: float(features.get(k, 0.0)) for k in morphology_features}
                df_morph = pd.DataFrame([morph_data]).T.rename(columns={0: 'Value'})
                st.dataframe(df_morph, height=200)

@st.fragment
def display_advanced_analysis(results: Dict[str, Any]):
    st.header("📈 Advanced Analysis")
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🎯 Feature Importance", "🔮 Prediction"])

    with tab1:
        feature_df = pd.DataFrame(results['features']).select_dtypes(include=[np.number]).fillna(0)
        if not feature_df.empty:
            stats_df = pd.DataFrame({
                'Feature': feature_df.columns[:8],
                'Mean': np.mean(feature_df.values, axis=0)[:8],
                'Std Dev': np.std(feature_df.values, axis=0)[:8],
                'Min': np.min(feature_df.values, axis=0)[:8],
                'Max': np.max(feature_df.values, axis=0)[:8]
            })
            st.dataframe(stats_df, use_container_width=True, height=300)
        else:
            st.info("No numeric features available for statistics.")

    with tab2:
        importance = {
            'Sarcomere': np.mean([f.get('sarcomere_regularity', 0) for f in results['features']]) * 0.15,
            'Motion': np.mean([f.get('motion_entropy', 0) for f in results['features']]) * 0.10,
            'Vector': np.mean([f.get('motion_vector_magnitude', 0) for f in results['features']]) * 0.10,
            'Mitochondrial': np.mean([f.get('mitochondrial_density', 0) for f in results['features']]) * 0.12
        }
        fig = px.bar(x=list(importance.keys()), y=list(importance.values()),
                     title="Feature Contributions to Aging Score",
                     color=list(importance.values()), color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=0, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        current_score = float(np.mean(results['scores'])) if len(results['scores']) else 0.0
        years = np.arange(0, 11)
        base_rate = 2.0 + (1.0 if np.mean([f.get('motion_entropy', 2) for f in results['features']]) > 3.0 else 0)
        fig_pred = go.Figure()
        scenarios = [
            ('Current Trajectory', base_rate, '#d62728'),
            ('With Intervention', max(base_rate * 0.4, 0.5), '#2ca02c'),
            ('Accelerated Aging', base_rate * 1.6, '#ff7f0e')
        ]
        for scenario, rate, color in scenarios:
            predicted = np.clip(current_score + years * rate, 0, 100)
            fig_pred.add_trace(go.Scatter(x=years, y=predicted, mode='lines+markers',
                                          name=scenario, line=dict(color=color, width=3)))
        fig_pred.update_layout(title="Ageing Trajectory Scenarios", xaxis_title="Years from Now",
                               yaxis_title="Aging Score", height=350, hovermode='x unified')
        st.plotly_chart(fig_pred, use_container_width=True)

# --------------------------
# NEW: Additional Analyses
# --------------------------
@st.fragment
def display_pca_and_correlations(results: Dict[str, Any]):
    st.header("🧠 PCA & Correlations")

    df = pd.DataFrame(results['features']).select_dtypes(include=[np.number]).fillna(0)
    if df.empty or len(df) < 2:
        st.info("Need at least 2 samples with numeric features for PCA/correlations.")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    pca = PCA(n_components=min(3, X.shape[1]))
    comps = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        'PC1': comps[:, 0],
        'PC2': comps[:, 1] if comps.shape[1] > 1 else 0.0,
        'Stage': results['stages'],
        'Score': results['scores'],
        'Sample': results['names']
    })
    st.subheader("PCA Scatter (PC1 vs PC2)")
    figp = px.scatter(pca_df, x='PC1', y='PC2', color='Stage', size='Score', hover_name='Sample')
    figp.update_layout(height=450)
    st.plotly_chart(figp, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    corr = np.corrcoef(df.values, rowvar=False)
    figc = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=df.columns.tolist(),
            y=df.columns.tolist(),
            zmin=-1, zmax=1,
            colorbar=dict(title='r')
        )
    )
    figc.update_layout(title="Correlation Matrix", height=600)
    st.plotly_chart(figc, use_container_width=True)

    # Also show top absolute correlations (excluding self)
    with st.expander("Top |r| feature pairs", expanded=False):
        cols = df.columns.tolist()
        pairs = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append((cols[i], cols[j], float(abs(corr[i, j]))))
        pairs.sort(key=lambda x: x[2], reverse=True)
        topk = pd.DataFrame(pairs[:20], columns=["Feature A", "Feature B", "|r|"])
        st.dataframe(topk, use_container_width=True, height=320)


@st.fragment
def display_outlier_detection(results: Dict[str, Any]):
    st.header("🚨 Outlier Detection (Isolation Forest)")
    df = pd.DataFrame(results['features']).select_dtypes(include=[np.number]).fillna(0)
    if df.empty or len(df) < 3:
        st.info("Need at least 3 samples with numeric features for outlier detection.")
        return

    c1, c2 = st.columns([1, 3])
    with c1:
        contamination = st.slider("Expected outlier fraction", 0.01, 0.30, 0.10, 0.01)
        n_estimators = st.slider("Trees", 50, 400, 200, 50)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Robust default: auto behavior varies across versions; use contamination from slider
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)          # -1 = outlier, 1 = inlier
    scores = -iso.score_samples(X)      # higher = more anomalous

    outlier_mask = (preds == -1)
    res_df = pd.DataFrame({
        "Sample": results['names'],
        "Score": results['scores'],
        "Stage": results['stages'],
        "AnomalyScore": scores,
        "IsOutlier": outlier_mask
    }).sort_values("AnomalyScore", ascending=False)

    with c2:
        st.metric("Detected outliers", f"{int(outlier_mask.sum())} / {len(outlier_mask)}")

    # Quick 2D projection (PCA) colored by outlier flag
    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(X)
    pca_plot = pd.DataFrame({
        "PC1": pc[:, 0], "PC2": pc[:, 1],
        "Outlier": np.where(outlier_mask, "Outlier", "Inlier"),
        "Sample": results['names'],
        "Score": results['scores']
    })
    figp = px.scatter(pca_plot, x="PC1", y="PC2", color="Outlier", size="Score", hover_name="Sample", symbol="Outlier")
    figp.update_layout(title="PCA projection with IsolationForest outliers", height=450)
    st.plotly_chart(figp, use_container_width=True)

    with st.expander("Outlier table (sorted by anomaly score)", expanded=True):
        st.dataframe(res_df, use_container_width=True, height=360)


def generate_report(results: Dict[str, Any]):
    """Generate analysis report."""
    st.header("📋 Report Generation")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.info(f"Generate a comprehensive analysis report for {len(results['names'])} samples")
    with c2:
        if st.button("📄 Generate", type="primary", key="generate_report"):
            try:
                feature_df = pd.DataFrame(results['features']).select_dtypes(include=[np.number]).fillna(0)
                sar_mean = np.mean([f.get('sarcomere_regularity', 0) for f in results['features']]) if len(results['features']) else 0
                mot_entropy_mean = np.mean([f.get('motion_entropy', 0) for f in results['features']]) if len(results['features']) else 0
                vec_mean = np.mean([f.get('motion_vector_magnitude', 0) for f in results['features']]) if len(results['features']) else 0
                mito_mean = np.mean([f.get('mitochondrial_density', 0) for f in results['features']]) if len(results['features']) else 0

                report = f"""# Cardiomyocyte Aging Analysis Report

## Executive Summary
- **Total Samples Analyzed**: {len(results['scores'])}
- **Average Aging Score**: {np.mean(results['scores']):.1f}/100
- **Score Range**: {np.min(results['scores']):.1f} - {np.max(results['scores']):.1f}
- **Primary Classification**: {results['stages'][0] if len(set(results['stages'])) == 1 and len(results['stages'])>0 else 'Multiple Classifications'}
- **Average Confidence**: {np.mean([float(c) for c in results['confidences']]):.2%}

## Key Metrics Summary
| Metric | Mean Value |
|--------|------------|
| Sarcomere Regularity | {sar_mean:.4f} |
| Motion Entropy | {mot_entropy_mean:.4f} |
| Motion Vector Magnitude | {vec_mean:.4f} |
| Mitochondrial Density | {mito_mean:.4f} |

## Sample Details (First 20)
| Sample | Score | Stage | Confidence | Day |
|--------|-------|-------|------------|-----|
""" + "\n".join([f"| {n[:30]} | {s:.1f} | {stg} | {float(c):.1%} | {d if d else 'N/A'} |"
                 for n, s, stg, c, d in zip(results['names'][:20], results['scores'][:20],
                                            results['stages'][:20], results['confidences'][:20],
                                            results['days'][:20])])

                if len(results['names']) > 20:
                    report += f"\n\n*Showing first 20 of {len(results['names'])} samples*"

                crit_list = [m for c in results['critical'] for m in (c.split(', ') if c != 'None' else [])]
                report += f"""

## Critical Markers Identified
{', '.join(sorted(set(crit_list))) or 'No critical markers identified'}

## Feature Columns
{', '.join(feature_df.columns.tolist()) if not feature_df.empty else 'No numeric features available.'}

---
*Generated by Cardiomyocyte Aging Analysis System*
"""
                with st.expander("📄 View Report", expanded=True):
                    st.text_area("Report Content", report, height=400, key="report_text")
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report,
                    file_name=f"cardiomyocyte_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    key="download_report_btn"
                )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")


def main():
    """Main application function with optimized flow."""
    initialize_session_state()
    st.markdown(configure_streamlit(), unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🧬 Cardiomyocyte Aging Analysis</h1>', unsafe_allow_html=True)

    analyzer = get_analyzer()

    # Create UI components
    options = create_sidebar()
    uploaded_files, analysis_button, temporal_button = create_file_upload_section()

    # Process files if button clicked
    if uploaded_files and (analysis_button or temporal_button):
        current_files = [f.name for f in uploaded_files]
        need_reprocess = (
            'last_files' not in st.session_state or
            st.session_state.last_files != current_files or
            'last_sensitivity' not in st.session_state or
            st.session_state.last_sensitivity != options['sensitivity']
        )
        if need_reprocess:
            with st.spinner(f"🔄 Processing {len(uploaded_files)} files..."):
                results = process_uploaded_files_parallel(analyzer, uploaded_files, options['sensitivity'])
                if results['features']:
                    st.session_state.current_results = results
                    st.session_state.last_files = current_files
                    st.session_state.last_sensitivity = options['sensitivity']
                else:
                    st.error("❌ No files could be processed successfully")
                    return
        else:
            results = st.session_state.current_results
            if not results or not results['features']:
                st.error("❌ No cached results available. Please reanalyze.")
                return

        st.success(f"✅ Successfully analyzed {len(results['features'])} file(s)")

        # Optional temporal analysis section gated by the toggle
        if temporal_button and options['show_temporal']:
            display_temporal_analysis(results, options)
            st.divider()

        # Tabs
        if len(results['features']) == 1:
            tab_names = ["📊 Main Analysis", "🗺️ Clustering", "🏃 Motion", "🔬 Morphology", "📈 Advanced", "🧠 PCA & Corr", "🚨 Outliers"]
        else:
            tab_names = ["📊 Overview", "🗺️ Clustering", "🏃 Motion", "🔬 Morphology", "📈 Advanced", "🧠 PCA & Corr", "🚨 Outliers"]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            if len(results['features']) == 1:
                display_single_file_analysis(results, options)
            else:
                display_multiple_file_analysis(results, options)

        with tabs[1]:
            if len(results['features']) > 1:
                with st.spinner("Computing clusters..."):
                    cluster_df = create_cluster_analysis(results)
                    display_cluster_analysis(cluster_df, results, options)
            else:
                st.info("📊 Cluster analysis requires at least 2 samples")

        with tabs[2]:
            display_motion_analysis(results, options)

        with tabs[3]:
            display_morphology_analysis(results, options)

        with tabs[4]:
            display_advanced_analysis(results)

        with tabs[5]:
            display_pca_and_correlations(results)

        with tabs[6]:
            display_outlier_detection(results)

        st.divider()
        generate_report(results)

    elif uploaded_files and not (analysis_button or temporal_button):
        st.info("👆 Click 'Analyze All' or 'Temporal Analysis' to start processing")

    # Footer
    st.divider()
    with st.expander("ℹ️ About This Application"):
        st.markdown(f"""
**Cardiomyocyte Aging Analysis System v2.1**

**Highlights**
- Unlimited file uploads, {MAX_FILE_SIZE_MB}MB per file
- ND2 support {"(Active)" if ND2_AVAILABLE else "(Install nd2reader)"}
- Temporal day 1–8 analysis (filename-based)
- UMAP clustering, PCA visualization
- LIME biomarker summaries
- Motion & morphology dashboards
- IsolationForest outlier detection
- Report generator
        """)


if __name__ == "__main__":
    # Windows event loop fix is unnecessary unless using asyncio; keep minimal
    main()
