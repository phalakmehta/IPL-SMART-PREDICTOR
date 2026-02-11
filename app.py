import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="IPL Victory Predictor 2025",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR HIGH-VISIBILITY UI ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white; /* Default text color */
        font-family: 'Roboto', sans-serif;
    }

    /* TEXT VISIBILITY FIXES */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #ffffff !important; /* Force white text */
    }

    /* Input Widget Labels (The text above boxes) */
    .stNumberInput label p, .stSelectbox label p {
        color: #e0e0e0 !important; /* Slightly off-white for labels */
        font-weight: 600;
        font-size: 1rem;
    }

    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.08); /* Increased opacity slightly */
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
    }

    /* METRIC BOXES */
    .metric-box {
        text-align: center;
        background: rgba(0, 0, 0, 0.4); /* Darker background for contrast */
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #dddddd !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }

    /* PROGRESS BAR CONTAINER */
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 30px;
        height: 40px;
        width: 100%;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
    }
    .prob-bar-fill {
        height: 100%;
        transition: width 0.6s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    /* INPUT FIELDS STYLING */
    /* Dropdown and Number Input Containers */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Text Inside Inputs */
    .stNumberInput input {
    color: #ff2e63 !important;
    font-weight: 800 !important;
    text-shadow: 0 0 8px rgba(255, 46, 99, 0.8);
}

.stSelectbox div[data-baseweb="select"] span {
    color: #ff2e63 !important;
    font-weight: 800 !important;
    text-shadow: 0 0 8px rgba(255, 46, 99, 0.8);
}


    /* BUTTONS */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 15px 0;
    }
    .primary-btn > button {
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%);
        color: white !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.4);
    }
    .primary-btn > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(221, 36, 118, 0.6);
    }

    /* ACTION BUTTONS (WHAT IF) */
    .action-btn > button {
        background: rgba(255, 255, 255, 0.15);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3);
        width: 100%;
    }
    .action-btn > button:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: white;
    }
    </style>
""", unsafe_allow_html=True)


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('ipl_win_predictor.pkl', 'rb'))
    except FileNotFoundError:
        return None


pipe = load_model()

# --- CONSTANTS ---
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
])

cities = sorted([
    'Mumbai', 'Kolkata', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai',
    'Jaipur', 'Mohali', 'Ahmedabad', 'Pune', 'Visakhapatnam', 'Indore',
    'Dubai', 'Abu Dhabi', 'Sharjah', 'Raipur', 'Ranchi', 'Durban', 'Centurion'
])

team_colors = {
    'Sunrisers Hyderabad': '#F78125', 'Mumbai Indians': '#004BA0',
    'Royal Challengers Bengaluru': '#EC1C24', 'Kolkata Knight Riders': '#3A225D',
    'Punjab Kings': '#DD1F2D', 'Chennai Super Kings': '#F9CD05',
    'Rajasthan Royals': '#EA1A85', 'Delhi Capitals': '#00008B',
    'Gujarat Titans': '#1B2133', 'Lucknow Super Giants': '#3FD5EA'
}

# --- SESSION STATE INITIALIZATION ---
if 'score' not in st.session_state: st.session_state.score = 100
if 'wickets' not in st.session_state: st.session_state.wickets = 2
if 'overs' not in st.session_state: st.session_state.overs = 10
if 'balls' not in st.session_state: st.session_state.balls = 0
if 'target' not in st.session_state: st.session_state.target = 180


# --- HELPER: UPDATE STATE ---
def update_match(runs, wkt):
    # Don't update if match is over
    if st.session_state.wickets >= 10 or st.session_state.score >= st.session_state.target:
        return

    st.session_state.score += runs
    st.session_state.wickets += wkt

    # Ball logic
    st.session_state.balls += 1
    if st.session_state.balls == 6:
        st.session_state.balls = 0
        st.session_state.overs += 1


# --- HEADER ---
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("<h1>🏏 IPL WIN PREDICTOR 2025</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ddd; margin-top: -15px; font-size: 1.1rem;'>Real-time AI Probability Engine</p>",
                unsafe_allow_html=True)

st.markdown("---")

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.2], gap="large")

# === LEFT COLUMN: INPUTS ===
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏟️ Match Settings")

    # Dropdowns
    batting_team = st.selectbox('🏏 Batting (Chasing)', teams, index=5)
    bowling_team = st.selectbox('⚾ Bowling (Defending)', [t for t in teams if t != batting_team], index=1)
    selected_city = st.selectbox('📍 Venue', cities, index=3)

    st.session_state.target = st.number_input('🎯 Target Score', min_value=1, value=st.session_state.target)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔢 Current Situation")

    # Live Inputs
    # WRAPPED IN TRY-CATCH TO PREVENT WIDGET OVERFLOW ERRORS
    try:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.score = st.number_input('Runs Scored', min_value=0, value=st.session_state.score, step=1)
        with c2:
            st.session_state.wickets = st.number_input('Wickets Lost', min_value=0, max_value=10,
                                                       value=st.session_state.wickets, step=1)

        c3, c4 = st.columns(2)
        with c3:
            st.session_state.overs = st.number_input('Overs Completed', min_value=0, max_value=19,
                                                     value=st.session_state.overs, step=1)
        with c4:
            st.session_state.balls = st.number_input('Balls (Current Over)', min_value=0, max_value=5,
                                                     value=st.session_state.balls, step=1)
    except Exception:
        st.error("⚠️ Invalid input detected. Please check your score and overs.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Primary Action Button
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    if st.button('🚀 PREDICT PROBABILITY'):
        pass  # Just triggers rerun
    st.markdown('</div>', unsafe_allow_html=True)

# === RIGHT COLUMN: RESULTS ===
with col2:
    if pipe is None:
        st.error("⚠️ Model file not found. Please upload 'ipl_win_predictor.pkl'.")
    else:
        # WRAPPED ENTIRE CALCULATION IN TRY-CATCH
        try:
            # 1. Calculations
            total_balls = (st.session_state.overs * 6) + st.session_state.balls
            balls_left = 120 - total_balls
            runs_left = st.session_state.target - st.session_state.score
            wickets_left = 10 - st.session_state.wickets

            # Avoid division by zero
            crr = st.session_state.score / (total_balls / 6) if total_balls > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            # 2. Prediction
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'current_score': [st.session_state.score],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'crr': [crr],
                'rrr': [rrr]
            })

            result = pipe.predict_proba(input_df)
            loss_prob = result[0][0]
            win_prob = result[0][1]

            # 3. Colors
            bat_color = team_colors.get(batting_team, '#4CAF50')
            bowl_color = team_colors.get(bowling_team, '#FF5252')

            # 4. Display Metrics
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(
                f"<div class='metric-box'><div class='metric-value'>{int(runs_left)}</div><div class='metric-label'>Need</div></div>",
                unsafe_allow_html=True)
            m2.markdown(
                f"<div class='metric-box'><div class='metric-value'>{int(balls_left)}</div><div class='metric-label'>Balls</div></div>",
                unsafe_allow_html=True)
            m3.markdown(
                f"<div class='metric-box'><div class='metric-value'>{crr:.1f}</div><div class='metric-label'>CRR</div></div>",
                unsafe_allow_html=True)
            m4.markdown(
                f"<div class='metric-box'><div class='metric-value'>{rrr:.1f}</div><div class='metric-label'>RRR</div></div>",
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 5. Probability Bar
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom:5px; font-weight:bold;">
                    <span style="color:{bat_color}; font-size: 1.1rem;">{batting_team} ({round(win_prob * 100)}%)</span>
                    <span style="color:{bowl_color}; font-size: 1.1rem;">{bowling_team} ({round(loss_prob * 100)}%)</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {win_prob * 100}%; background: {bat_color};"></div>
                </div>
            """, unsafe_allow_html=True)

            # 6. Commentary Card
            if win_prob > 0.6:
                status, status_color = f"🔥 {batting_team} is cruising!", "#4ade80"
            elif win_prob < 0.4:
                status, status_color = f"🛡️ {bowling_team} is tightening the grip!", "#f87171"
            else:
                status, status_color = "⚖️ It's a nail-biter! Anyone's game.", "#facc15"

            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); margin-top: 20px; padding: 15px; border-left: 5px solid {status_color}; border-radius: 5px;">
                    <h3 style="margin:0; color: {status_color} !important; font-size: 1.2rem;">{status}</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception:
            st.error("🚨 Invalid Match Scenario: Probabilities cannot be calculated for the current inputs.")

        # 7. "WHAT IF" SCENARIO SIMULATOR
        st.markdown("### 🎮 Simulate Next Ball")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        b1, b2, b3, b4, b5 = st.columns(5)

        with b1:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("⚪ Dot"): update_match(0, 0); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("1️⃣ Single"): update_match(1, 0); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("4️⃣ Four"): update_match(4, 0); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b4:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("6️⃣ Six"): update_match(6, 0); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b5:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("☝️ OUT"): update_match(0, 1); st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)