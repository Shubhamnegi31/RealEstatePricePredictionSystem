"""
Indian Real Estate Price Prediction System - Interactive Streamlit Web Application

Features:
- Interactive Plotly Map with Map Click Selection & Satellite View Toggle.
- Dynamic State Landmark Visual Showcase Banner.
- Reactive Location Selectors (State -> City -> Locality).
- Machine Learning Price Estimation Engine with Market Gauge Meter.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.prediction import predict_price, format_inr_price
from src.data_loader import load_data

# Page Configuration
st.set_page_config(
    page_title="Indian Real Estate Valuation & Explorer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dynamic Design
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311b92 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0;
    }

    /* Dynamic State Visual Showcase Card */
    .state-banner-card {
        position: relative;
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 15px 25px rgba(0, 0, 0, 0.3);
        height: 240px;
        background-size: cover;
        background-position: center;
    }

    .state-banner-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.92) 0%, rgba(15, 23, 42, 0.75) 50%, rgba(15, 23, 42, 0.35) 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0 2.2rem;
    }

    .state-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.2);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 8px;
        width: fit-content;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .state-name-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    .state-desc {
        color: #e2e8f0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Valuation Result Card */
    .price-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 1.5px solid #10b981;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 25px rgba(16, 185, 129, 0.25);
    }

    .price-label {
        font-size: 1.15rem;
        color: #a7f3d0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    .price-amount-main {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.4rem 0;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .price-amount-sub {
        font-size: 1.3rem;
        color: #6ee7b7;
        font-weight: 500;
    }

    /* Metric Container */
    .metric-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Comprehensive 42 City Coordinate Database for All Dataset Cities
CITY_COORDINATES = {
    "Vijayawada": {"state": "Andhra Pradesh", "lat": 16.5062, "lon": 80.6480},
    "Vishakhapatnam": {"state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185},
    "Guwahati": {"state": "Assam", "lat": 26.1445, "lon": 91.7362},
    "Silchar": {"state": "Assam", "lat": 24.8333, "lon": 92.7789},
    "Gaya": {"state": "Bihar", "lat": 24.7914, "lon": 85.0002},
    "Patna": {"state": "Bihar", "lat": 25.5941, "lon": 85.1376},
    "Bilaspur": {"state": "Chhattisgarh", "lat": 22.0797, "lon": 82.1391},
    "Raipur": {"state": "Chhattisgarh", "lat": 21.2514, "lon": 81.6296},
    "Dwarka": {"state": "Delhi", "lat": 28.5921, "lon": 77.0460},
    "New Delhi": {"state": "Delhi", "lat": 28.6139, "lon": 77.2090},
    "Ahmedabad": {"state": "Gujarat", "lat": 23.0225, "lon": 72.5714},
    "Surat": {"state": "Gujarat", "lat": 21.1702, "lon": 72.8311},
    "Faridabad": {"state": "Haryana", "lat": 28.4089, "lon": 77.3178},
    "Gurgaon": {"state": "Haryana", "lat": 28.4595, "lon": 77.0266},
    "Jamshedpur": {"state": "Jharkhand", "lat": 22.8046, "lon": 86.2029},
    "Ranchi": {"state": "Jharkhand", "lat": 23.3441, "lon": 85.3096},
    "Bangalore": {"state": "Karnataka", "lat": 12.9716, "lon": 77.5946},
    "Mangalore": {"state": "Karnataka", "lat": 12.9141, "lon": 74.8560},
    "Mysore": {"state": "Karnataka", "lat": 12.2958, "lon": 76.6394},
    "Kochi": {"state": "Kerala", "lat": 9.9312, "lon": 76.2673},
    "Trivandrum": {"state": "Kerala", "lat": 8.5241, "lon": 76.9366},
    "Bhopal": {"state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126},
    "Indore": {"state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577},
    "Mumbai": {"state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    "Nagpur": {"state": "Maharashtra", "lat": 21.1458, "lon": 79.0882},
    "Pune": {"state": "Maharashtra", "lat": 18.5204, "lon": 73.8567},
    "Bhubaneswar": {"state": "Odisha", "lat": 20.2961, "lon": 85.8245},
    "Cuttack": {"state": "Odisha", "lat": 20.4625, "lon": 85.8828},
    "Amritsar": {"state": "Punjab", "lat": 31.6340, "lon": 74.8723},
    "Ludhiana": {"state": "Punjab", "lat": 30.9010, "lon": 75.8573},
    "Jaipur": {"state": "Rajasthan", "lat": 26.9124, "lon": 75.7873},
    "Jodhpur": {"state": "Rajasthan", "lat": 26.2389, "lon": 73.0243},
    "Chennai": {"state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    "Coimbatore": {"state": "Tamil Nadu", "lat": 11.0168, "lon": 76.9558},
    "Hyderabad": {"state": "Telangana", "lat": 17.3850, "lon": 78.4867},
    "Warangal": {"state": "Telangana", "lat": 17.9689, "lon": 79.5941},
    "Lucknow": {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
    "Noida": {"state": "Uttar Pradesh", "lat": 28.5355, "lon": 77.3910},
    "Dehradun": {"state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322},
    "Haridwar": {"state": "Uttarakhand", "lat": 29.9457, "lon": 78.1642},
    "Durgapur": {"state": "West Bengal", "lat": 23.5204, "lon": 87.3119},
    "Kolkata": {"state": "West Bengal", "lat": 22.5726, "lon": 88.3639}
}

# State Landmark Images & Descriptions Mapping
STATE_METADATA = {
    "Rajasthan": {
        "image": "https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Hawa Mahal & Pink City Palaces",
        "tagline": "Land of Forts, Royal Heritage & Rapidly Growing Housing Hubs"
    },
    "Maharashtra": {
        "image": "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Gateway of India & Financial Skyline",
        "tagline": "India's Commercial Capital & Premium Coastal Real Estate"
    },
    "Karnataka": {
        "image": "https://images.unsplash.com/photo-1596176530529-78163a4f7af2?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Bangalore Tech Parks & Heritage Palaces",
        "tagline": "Silicon Valley of India & High-Growth IT Real Estate Corridor"
    },
    "Delhi": {
        "image": "https://images.unsplash.com/photo-1587474260584-136574528ed5?auto=format&fit=crop&w=1200&q=80",
        "landmark": "India Gate & Capital Architecture",
        "tagline": "National Capital Region with Luxury Highrises & Metro Connectivity"
    },
    "Tamil Nadu": {
        "image": "https://images.unsplash.com/photo-1582510003544-4d00b7f74220?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Dravidian Heritage & Coastal Chennai",
        "tagline": "Industrial Titan & Automobile Capital Real Estate"
    },
    "Telangana": {
        "image": "https://images.unsplash.com/photo-1605649487212-47bdab064df7?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Charminar & Cyberabad IT District",
        "tagline": "Biotech & Cyber Corridor with Premium Gated Communities"
    },
    "West Bengal": {
        "image": "https://images.unsplash.com/photo-1558431382-27e303142255?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Howrah Bridge & Heritage Architecture",
        "tagline": "Cultural Capital & Emerging Eastern Financial Hub"
    },
    "Gujarat": {
        "image": "https://images.unsplash.com/photo-1609949279531-cf48d64bed89?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Sabarmati Riverfront & Textile Hubs",
        "tagline": "Vibrant Industrial & Mega Infrastructure Real Estate"
    },
    "Uttar Pradesh": {
        "image": "https://images.unsplash.com/photo-1564507592333-c60657eea523?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Taj Mahal & NCR Suburbs",
        "tagline": "Historic Heartland & Expanding Modern NCR Highrises"
    },
    "Kerala": {
        "image": "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Backwaters & Tropical Greenery",
        "tagline": "God's Own Country & Scenic Waterfront Villas"
    },
    "Haryana": {
        "image": "https://images.unsplash.com/photo-1586724237569-f3d0c1dee8c6?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Gurgaon Cyber City Skyline",
        "tagline": "Corporate Headquarters & Ultra-Luxury Highrise Residences"
    },
    "Punjab": {
        "image": "https://images.unsplash.com/photo-1588097281266-310ceea50820?auto=format&fit=crop&w=1200&q=80",
        "landmark": "Golden Temple & Rich Heritage",
        "tagline": "Prosperous Urban Centers & Agricultural Heritage"
    }
}

DEFAULT_METADATA = {
    "image": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=1200&q=80",
    "landmark": "Indian Real Estate Architecture",
    "tagline": "Exploring Premium Property Markets Across India"
}


@st.cache_data
def get_cached_dataset():
    """Loads dataset for dynamic selection options and metadata."""
    try:
        df = load_data(data_path="data/india_housing_prices.csv", sample_size=50000)
        return df
    except Exception as e:
        st.error(f"Error loading dataset options: {e}")
        return None


@st.cache_resource
def get_cached_model():
    """Loads trained model pipeline from disk."""
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Failed to load model file: {e}")
            return None
    return None


def build_interactive_map(selected_city: str, map_style_mode: str):
    """
    Builds an interactive Plotly Map with satellite/street layer choices,
    showing all Indian dataset cities. Clicking any pin on the map selects that city!
    """
    city_names = []
    state_names = []
    lats = []
    lons = []
    colors = []
    sizes = []

    for city, info in CITY_COORDINATES.items():
        city_names.append(city)
        state_names.append(info['state'])
        lats.append(info['lat'])
        lons.append(info['lon'])
        
        if city == selected_city:
            colors.append('#f43f5e')  # Glowing Rose Red for active selection
            sizes.append(22)
        else:
            colors.append('#38bdf8')  # Sky Blue for other cities
            sizes.append(11)

    df_map = pd.DataFrame({
        'city': city_names,
        'state': state_names,
        'lat': lats,
        'lon': lons,
        'color': colors,
        'size': sizes
    })

    # Mapbox Style Mapping
    style_mapping = {
        "📡 Satellite View": "open-street-map",
        "🗺️ Standard Terrain": "open-street-map",
        "🌙 Dark Mode": "carto-darkmatter",
        "☀️ Light Minimal": "carto-positron"
    }
    mapbox_style = style_mapping.get(map_style_mode, "open-street-map")

    # Target Lat/Lon for active city
    active_info = CITY_COORDINATES.get(selected_city, {"lat": 20.5937, "lon": 78.9629})

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        hover_name="city",
        hover_data={"state": True, "lat": False, "lon": False, "color": False, "size": False},
        color="color",
        color_discrete_map={"#f43f5e": "#f43f5e", "#38bdf8": "#38bdf8"},
        size="size",
        size_max=22,
        zoom=4.8,
        center={"lat": active_info["lat"], "lon": active_info["lon"]},
        mapbox_style=mapbox_style
    )

    fig.update_traces(
        marker=dict(opacity=0.9),
        customdata=df_map[['city', 'state']]
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_price_gauge(price_lakhs: float):
    """Creates a modern Plotly price gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price_lakhs,
        number={'suffix': " Lakhs", 'font': {'color': "#ffffff", 'size': 28, 'family': "Outfit"}},
        title={'text': "Market Valuation Gauge (Rs. Lakhs)", 'font': {'color': "#94a3b8", 'size': 14}},
        gauge={
            'axis': {'range': [10, 500], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#38bdf8", 'thickness': 0.3},
            'bgcolor': "#0f172a",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [10, 150], 'color': 'rgba(16, 185, 129, 0.25)'},
                {'range': [150, 300], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [300, 500], 'color': 'rgba(244, 63, 94, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#f43f5e", 'width': 3},
                'thickness': 0.75,
                'value': price_lakhs
            }
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Outfit"}
    )
    return fig


def main():
    # Hero Section
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🏡 Indian Real Estate Valuation & Explorer</div>
            <p class="hero-subtitle">Interactive machine learning price prediction across top Indian housing markets.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_data = get_cached_dataset()
    model = get_cached_model()

    # Dynamic Option Lists
    if df_data is not None:
        states = sorted(df_data['State'].dropna().unique().tolist())
        cities = sorted(df_data['City'].dropna().unique().tolist())
        localities = sorted(df_data['Locality'].dropna().unique().tolist())
        property_types = sorted(df_data['Property_Type'].dropna().unique().tolist())
        furnishing_options = sorted(df_data['Furnished_Status'].dropna().unique().tolist())
        facing_options = sorted(df_data['Facing'].dropna().unique().tolist())
        owner_options = sorted(df_data['Owner_Type'].dropna().unique().tolist())
        availability_options = sorted(df_data['Availability_Status'].dropna().unique().tolist())
        transport_options = sorted(df_data['Public_Transport_Accessibility'].dropna().unique().tolist())
    else:
        states = ["Rajasthan", "Maharashtra", "Tamil Nadu", "Karnataka", "Delhi", "Telangana", "West Bengal"]
        cities = ["Jaipur", "Mumbai", "Chennai", "Bangalore", "Delhi", "Hyderabad", "Kolkata"]
        localities = [f"Locality_{i}" for i in range(1, 50)]
        property_types = ["Apartment", "Independent House", "Villa"]
        furnishing_options = ["Furnished", "Semi-furnished", "Unfurnished"]
        facing_options = ["East", "West", "North", "South"]
        owner_options = ["Owner", "Builder", "Broker"]
        availability_options = ["Ready_to_Move", "Under_Construction"]
        transport_options = ["High", "Medium", "Low"]

    # Initialize Session State for State & City Selection
    if "selected_state" not in st.session_state:
        st.session_state["selected_state"] = "Rajasthan" if "Rajasthan" in states else states[0]
    if "selected_city" not in st.session_state:
        st.session_state["selected_city"] = "Jaipur" if "Jaipur" in cities else cities[0]

    # Sidebar
    st.sidebar.header("📊 Model Status")
    if model is not None:
        st.sidebar.success("✅ Machine Learning Pipeline Ready")
        st.sidebar.caption("Ensemble Random Forest Regressor")
        st.sidebar.caption("Trained on 100,000 Sample Records")
    else:
        st.sidebar.warning("⚠️ Model Pipeline Not Loaded")
        if st.sidebar.button("⚙️ Train Model Now"):
            with st.spinner("Training model pipeline..."):
                from src.model_training import train_model
                model, _, _ = train_model(sample_size=50000)
                st.sidebar.success("Model trained!")
                st.rerun()



    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Features")
    st.sidebar.markdown(
        """
        - **Interactive Map Selection**: Click any city marker on the map to switch location!
        - **Data Leakage Protection**: `Price_per_SqFt` removed
        - **Target Unit**: INR Lakhs & Crores
        """
    )

    st.subheader("📍 Location & Interactive India Map Explorer")
    st.caption("👈 Click any city marker pin on the map or use the dropdowns below to select your property location.")

    # Location Selectors Row
    loc_col1, loc_col2, loc_col3 = st.columns(3)

    with loc_col1:
        state_idx = states.index(st.session_state["selected_state"]) if st.session_state["selected_state"] in states else 0
        new_state = st.selectbox("Select State", states, index=state_idx, key="state_select_box")

        if new_state != st.session_state["selected_state"]:
            st.session_state["selected_state"] = new_state
            # Reset city to first city in new state
            if df_data is not None:
                avail_cities = sorted(df_data[df_data['State'] == new_state]['City'].dropna().unique().tolist())
                st.session_state["selected_city"] = avail_cities[0] if avail_cities else cities[0]
            st.rerun()

    with loc_col2:
        if df_data is not None:
            state_cities = sorted(df_data[df_data['State'] == st.session_state["selected_state"]]['City'].dropna().unique().tolist())
        else:
            state_cities = cities

        current_city = st.session_state["selected_city"]
        city_idx = state_cities.index(current_city) if current_city in state_cities else 0
        new_city = st.selectbox("Select City", state_cities if state_cities else cities, index=city_idx, key="city_select_box")

        if new_city != st.session_state["selected_city"]:
            st.session_state["selected_city"] = new_city
            st.rerun()

    with loc_col3:
        if df_data is not None:
            city_localities = sorted(df_data[df_data['City'] == st.session_state["selected_city"]]['Locality'].dropna().unique().tolist())
        else:
            city_localities = localities
        selected_locality = st.selectbox("Select Locality", city_localities if city_localities else localities, index=0)

    # Dynamic State Banner & Map Layout
    meta = STATE_METADATA.get(st.session_state["selected_state"], DEFAULT_METADATA)
    
    col_banner, col_map = st.columns([1.3, 1.2])

    with col_banner:
        st.markdown(
            f"""
            <div class="state-banner-card" style="background-image: url('{meta['image']}');">
                <div class="state-banner-overlay">
                    <div class="state-tag">✨ {meta['landmark']}</div>
                    <div class="state-name-title">{st.session_state['selected_state']} Market</div>
                    <div style="color: #38bdf8; font-weight: 600; font-size: 1.1rem; margin-bottom: 4px;">📍 Active City: {st.session_state['selected_city']}</div>
                    <p class="state-desc">{meta['tagline']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_map:
        st.markdown("##### 🗺️ Interactive Map & View Layers")
        map_style_mode = st.radio(
            "Select Map Style:",
            ["📡 Satellite View", "🗺️ Standard Terrain", "🌙 Dark Mode", "☀️ Light Minimal"],
            horizontal=True,
            index=0,
            key="main_map_style_radio"
        )
        map_fig = build_interactive_map(st.session_state["selected_city"], map_style_mode)
        
        # Enable Map Point Click Selection in Streamlit!
        map_event = st.plotly_chart(
            map_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="interactive_india_map"
        )


        # Handle Click Selection from Map Pin
        if map_event and "selection" in map_event and map_event["selection"]["points"]:
            point = map_event["selection"]["points"][0]
            if "customdata" in point:
                clicked_city = point["customdata"][0]
                clicked_state = point["customdata"][1]
                if clicked_city != st.session_state["selected_city"]:
                    st.session_state["selected_state"] = clicked_state
                    st.session_state["selected_city"] = clicked_city
                    st.rerun()

    st.markdown("---")

    # Property Specifications Form
    st.subheader("📋 Property Specification & Attributes")

    with st.form("price_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🏢 Building & Orientation")
            selected_prop_type = st.selectbox("Property Type", property_types, index=0)
            selected_facing = st.selectbox("Facing Direction", facing_options, index=0)

        with col2:
            st.markdown("##### 📐 Space & Layout")
            bhk = st.slider("BHK (Bedrooms)", min_value=1, max_value=6, value=3, step=1)
            size_sqft = st.number_input("Property Size (Sq. Ft.)", min_value=300, max_value=10000, value=1450, step=50)
            floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=5, step=1)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=15, step=1)
            
            if floor_no > total_floors:
                total_floors = max(floor_no, total_floors)

            year_built = st.slider("Year Built", min_value=1990, max_value=2024, value=2018, step=1)
            age_of_property = max(0, 2026 - year_built)

        with col3:
            st.markdown("##### ⚙️ Amenities & Details")
            furnishing = st.selectbox("Furnishing Status", furnishing_options, index=0)
            owner_type = st.selectbox("Owner / Seller Type", owner_options, index=0)
            availability = st.selectbox("Availability Status", availability_options, index=0)
            transport = st.selectbox("Public Transport Access", transport_options, index=0)
            
            parking = st.radio("Parking Space Available?", ["Yes", "No"], horizontal=True)
            security = st.radio("Security Guard / CCTV?", ["Yes", "No"], horizontal=True)

            nearby_schools = st.slider("Nearby Schools (within 3 km)", min_value=0, max_value=15, value=5)
            nearby_hospitals = st.slider("Nearby Hospitals (within 3 km)", min_value=0, max_value=15, value=4)

        st.markdown("##### 🏊 Selected Amenities")
        col_am1, col_am2, col_am3, col_am4, col_am5 = st.columns(5)
        has_gym = col_am1.checkbox("Gymnasium", value=True)
        has_pool = col_am2.checkbox("Swimming Pool", value=True)
        has_garden = col_am3.checkbox("Landscaped Garden", value=True)
        has_playground = col_am4.checkbox("Kids Playground", value=True)
        has_clubhouse = col_am5.checkbox("Clubhouse", value=True)

        amenities_list = []
        if has_gym: amenities_list.append("Gym")
        if has_pool: amenities_list.append("Pool")
        if has_garden: amenities_list.append("Garden")
        if has_playground: amenities_list.append("Playground")
        if has_clubhouse: amenities_list.append("Clubhouse")
        amenities_str = ", ".join(amenities_list) if amenities_list else "None"

        submit_button = st.form_submit_button("💰 Calculate Price Estimate", use_container_width=True)

    # Handle Form Submission & Prediction
    if submit_button:
        if model is None:
            st.error("Model pipeline is not loaded. Please train the model using the sidebar button.")
        else:
            input_dict = {
                "State": st.session_state["selected_state"],
                "City": st.session_state["selected_city"],
                "Locality": selected_locality,
                "Property_Type": selected_prop_type,
                "BHK": int(bhk),
                "Size_in_SqFt": int(size_sqft),
                "Year_Built": int(year_built),
                "Furnished_Status": furnishing,
                "Floor_No": int(floor_no),
                "Total_Floors": int(total_floors),
                "Age_of_Property": int(age_of_property),
                "Nearby_Schools": int(nearby_schools),
                "Nearby_Hospitals": int(nearby_hospitals),
                "Public_Transport_Accessibility": transport,
                "Parking_Space": parking,
                "Security": security,
                "Amenities": amenities_str,
                "Facing": selected_facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability
            }

            with st.spinner(f"Evaluating valuation for {st.session_state['selected_city']}, {st.session_state['selected_state']}..."):
                res = predict_price(input_dict, model=model)

            price_lakhs = res['price_lakhs']
            price_crores = res['price_crores']

            # Display Result Card
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-label">Estimated Market Value in {st.session_state['selected_city']}, {st.session_state['selected_state']}</div>
                    <div class="price-amount-main">{format_inr_price(price_lakhs)}</div>
                    <div class="price-amount-sub">Rs. {price_lakhs:,.2f} Lakhs &nbsp;|&nbsp; Rs. {price_crores:.4f} Crores</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Valuation & Price Gauge Visual Section
            st.markdown("### 📊 Valuation Breakdown & Market Spectrum")
            tab1, tab2 = st.tabs(["✨ Key Metrics Highlights", "📈 Price Gauge & Spectrum"])

            with tab1:
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-value">{bhk} BHK</div>
                            <div class="metric-label">{selected_prop_type}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with mc2:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-value">{size_sqft:,} sq ft</div>
                            <div class="metric-label">Built-up Area</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with mc3:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-value">Rs. {int((price_lakhs * 100000) / size_sqft):,}/sqft</div>
                            <div class="metric-label">Estimated Rate</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with mc4:
                    st.markdown(
                        f"""
                        <div class="metric-container">
                            <div class="metric-value">{len(amenities_list)} / 5</div>
                            <div class="metric-label">Amenities Included</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with tab2:
                gauge_fig = create_price_gauge(price_lakhs)
                st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("---")


if __name__ == "__main__":
    main()
