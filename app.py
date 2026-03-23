"""
=============================================================================
APP.PY - Modern Streamlit Web Application for Deepfake Detection
=============================================================================
"""

import os
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import google.generativeai as genai
import base64
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from model import get_model
from gradcam import generate_gradcam_visualization

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - CREATIVE ARTISTIC UI INSPIRED BY YUCCA
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* ===== CSS VARIABLES ===== */
    :root {
        --bg-dark: #0c0c0c;
        --bg-card: rgba(20, 20, 20, 0.6);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --text-muted: rgba(255, 255, 255, 0.3);
        --accent-1: #c8ff00;
        --accent-2: #00ffc8;
        --accent-3: #ff6b35;
        --accent-4: #a855f7;
        --gradient-1: linear-gradient(135deg, #c8ff00 0%, #00ffc8 100%);
        --gradient-2: linear-gradient(135deg, #ff6b35 0%, #a855f7 100%);
        --gradient-3: linear-gradient(135deg, #00ffc8 0%, #0ea5e9 100%);
    }
    
    /* ===== BASE ===== */
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        background: var(--bg-dark) !important;
        overflow-x: hidden;
    }
    
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    /* ===== CREATIVE ANIMATED BACKGROUND ===== */
    .creative-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
        background: var(--bg-dark);
    }
    
    /* Large morphing gradient blobs */
    .morph-blob {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.6;
        mix-blend-mode: screen;
    }
    
    .blob-1 {
        width: 800px;
        height: 800px;
        background: radial-gradient(circle at 30% 30%, rgba(200, 255, 0, 0.4) 0%, rgba(0, 255, 200, 0.2) 50%, transparent 70%);
        top: -300px;
        right: -200px;
        animation: morph-1 20s infinite ease-in-out;
    }
    
    .blob-2 {
        width: 600px;
        height: 600px;
        background: radial-gradient(circle at 70% 70%, rgba(255, 107, 53, 0.35) 0%, rgba(168, 85, 247, 0.2) 50%, transparent 70%);
        bottom: -200px;
        left: -150px;
        animation: morph-2 25s infinite ease-in-out;
    }
    
    .blob-3 {
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0, 255, 200, 0.3) 0%, rgba(14, 165, 233, 0.15) 50%, transparent 70%);
        top: 40%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: morph-3 18s infinite ease-in-out;
    }
    
    @keyframes morph-1 {
        0%, 100% { 
            transform: translate(0, 0) scale(1) rotate(0deg);
            border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
        }
        25% { 
            transform: translate(50px, 100px) scale(1.1) rotate(90deg);
            border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%;
        }
        50% { 
            transform: translate(-50px, 50px) scale(0.95) rotate(180deg);
            border-radius: 50% 60% 30% 60% / 30% 60% 70% 40%;
        }
        75% { 
            transform: translate(30px, -50px) scale(1.05) rotate(270deg);
            border-radius: 60% 40% 60% 30% / 70% 30% 50% 60%;
        }
    }
    
    @keyframes morph-2 {
        0%, 100% { 
            transform: translate(0, 0) scale(1);
            border-radius: 40% 60% 60% 40% / 70% 30% 70% 30%;
        }
        33% { 
            transform: translate(80px, -60px) scale(1.15);
            border-radius: 70% 30% 50% 50% / 30% 70% 30% 70%;
        }
        66% { 
            transform: translate(-40px, 80px) scale(0.9);
            border-radius: 30% 70% 70% 30% / 60% 40% 60% 40%;
        }
    }
    
    @keyframes morph-3 {
        0%, 100% { 
            transform: translate(-50%, -50%) scale(1);
            opacity: 0.4;
        }
        50% { 
            transform: translate(-50%, -50%) scale(1.3);
            opacity: 0.6;
        }
    }
    
    /* Floating 3D shapes */
    .floating-shapes {
        position: absolute;
        width: 100%;
        height: 100%;
        perspective: 1000px;
    }
    
    .shape-3d {
        position: absolute;
        transform-style: preserve-3d;
    }
    
    /* Rotating ring */
    .ring-shape {
        width: 300px;
        height: 300px;
        border: 2px solid rgba(200, 255, 0, 0.3);
        border-radius: 50%;
        top: 15%;
        right: 10%;
        animation: ring-rotate 30s linear infinite;
    }
    
    .ring-shape::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        border: 2px solid rgba(0, 255, 200, 0.2);
        border-radius: 50%;
        transform: rotateX(60deg);
    }
    
    .ring-shape::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        border: 2px solid rgba(255, 107, 53, 0.2);
        border-radius: 50%;
        transform: rotateY(60deg);
    }
    
    @keyframes ring-rotate {
        0% { transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }
        100% { transform: rotateX(360deg) rotateY(360deg) rotateZ(360deg); }
    }
    
    /* Floating cube */
    .cube-container {
        width: 80px;
        height: 80px;
        top: 60%;
        left: 8%;
        animation: cube-float 15s ease-in-out infinite;
    }
    
    .cube {
        width: 100%;
        height: 100%;
        position: relative;
        transform-style: preserve-3d;
        animation: cube-rotate 20s linear infinite;
    }
    
    .cube-face {
        position: absolute;
        width: 80px;
        height: 80px;
        border: 1px solid rgba(200, 255, 0, 0.3);
        background: rgba(200, 255, 0, 0.03);
        backdrop-filter: blur(5px);
    }
    
    .cube-face:nth-child(1) { transform: rotateY(0deg) translateZ(40px); }
    .cube-face:nth-child(2) { transform: rotateY(90deg) translateZ(40px); }
    .cube-face:nth-child(3) { transform: rotateY(180deg) translateZ(40px); }
    .cube-face:nth-child(4) { transform: rotateY(-90deg) translateZ(40px); }
    .cube-face:nth-child(5) { transform: rotateX(90deg) translateZ(40px); }
    .cube-face:nth-child(6) { transform: rotateX(-90deg) translateZ(40px); }
    
    @keyframes cube-rotate {
        0% { transform: rotateX(0deg) rotateY(0deg); }
        100% { transform: rotateX(360deg) rotateY(360deg); }
    }
    
    @keyframes cube-float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-30px); }
    }
    
    /* Flowing wave lines */
    .wave-container {
        position: absolute;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    
    .wave-line {
        position: absolute;
        width: 200%;
        height: 2px;
        left: -50%;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(200, 255, 0, 0.1) 20%,
            rgba(0, 255, 200, 0.3) 50%,
            rgba(200, 255, 0, 0.1) 80%,
            transparent 100%
        );
        animation: wave-flow 8s linear infinite;
    }
    
    .wave-line:nth-child(1) { top: 20%; animation-delay: 0s; }
    .wave-line:nth-child(2) { top: 35%; animation-delay: -2s; opacity: 0.6; }
    .wave-line:nth-child(3) { top: 50%; animation-delay: -4s; opacity: 0.4; }
    .wave-line:nth-child(4) { top: 65%; animation-delay: -6s; opacity: 0.3; }
    .wave-line:nth-child(5) { top: 80%; animation-delay: -1s; opacity: 0.5; }
    
    @keyframes wave-flow {
        0% { transform: translateX(0) scaleY(1); }
        50% { transform: translateX(25%) scaleY(2); }
        100% { transform: translateX(50%) scaleY(1); }
    }
    
    /* Floating orbs with trail */
    .orb-trail {
        position: absolute;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        filter: blur(1px);
    }
    
    .orb-trail-1 {
        background: var(--accent-1);
        box-shadow: 
            0 0 20px var(--accent-1),
            0 0 40px var(--accent-1),
            0 0 60px rgba(200, 255, 0, 0.3);
        animation: orb-path-1 12s ease-in-out infinite;
    }
    
    .orb-trail-2 {
        background: var(--accent-2);
        width: 15px;
        height: 15px;
        box-shadow: 
            0 0 15px var(--accent-2),
            0 0 30px var(--accent-2);
        animation: orb-path-2 15s ease-in-out infinite;
    }
    
    .orb-trail-3 {
        background: var(--accent-3);
        width: 12px;
        height: 12px;
        box-shadow: 
            0 0 12px var(--accent-3),
            0 0 25px var(--accent-3);
        animation: orb-path-3 10s ease-in-out infinite;
    }
    
    @keyframes orb-path-1 {
        0%, 100% { top: 20%; left: 10%; }
        25% { top: 30%; left: 80%; }
        50% { top: 70%; left: 70%; }
        75% { top: 60%; left: 20%; }
    }
    
    @keyframes orb-path-2 {
        0%, 100% { top: 80%; left: 85%; }
        33% { top: 20%; left: 60%; }
        66% { top: 50%; left: 15%; }
    }
    
    @keyframes orb-path-3 {
        0%, 100% { top: 50%; left: 50%; }
        25% { top: 10%; left: 30%; }
        50% { top: 40%; left: 90%; }
        75% { top: 85%; left: 40%; }
    }
    
    /* Grid pattern with perspective */
    .perspective-grid {
        position: absolute;
        width: 200%;
        height: 100%;
        left: -50%;
        bottom: 0;
        background: 
            linear-gradient(90deg, rgba(200, 255, 0, 0.03) 1px, transparent 1px),
            linear-gradient(rgba(200, 255, 0, 0.03) 1px, transparent 1px);
        background-size: 100px 100px;
        transform: perspective(500px) rotateX(60deg);
        transform-origin: center bottom;
        mask-image: linear-gradient(to top, rgba(0,0,0,0.5) 0%, transparent 60%);
        -webkit-mask-image: linear-gradient(to top, rgba(0,0,0,0.5) 0%, transparent 60%);
        animation: grid-scroll 20s linear infinite;
    }
    
    @keyframes grid-scroll {
        0% { background-position: 0 0; }
        100% { background-position: 100px 100px; }
    }
    
    /* Decorative corner elements */
    .corner-decor {
        position: absolute;
        width: 150px;
        height: 150px;
        opacity: 0.4;
    }
    
    .corner-tl {
        top: 20px;
        left: 20px;
        border-left: 2px solid var(--accent-1);
        border-top: 2px solid var(--accent-1);
        animation: corner-pulse 3s infinite ease-in-out;
    }
    
    .corner-br {
        bottom: 20px;
        right: 20px;
        border-right: 2px solid var(--accent-2);
        border-bottom: 2px solid var(--accent-2);
        animation: corner-pulse 3s infinite ease-in-out 1.5s;
    }
    
    @keyframes corner-pulse {
        0%, 100% { opacity: 0.4; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    /* DNA helix effect */
    .dna-helix {
        position: absolute;
        right: 5%;
        top: 30%;
        height: 40%;
        width: 60px;
    }
    
    .dna-strand {
        position: absolute;
        width: 8px;
        height: 8px;
        background: var(--accent-1);
        border-radius: 50%;
        animation: dna-move 3s ease-in-out infinite;
    }
    
    .dna-strand:nth-child(odd) {
        background: var(--accent-2);
        animation-delay: -1.5s;
    }
    
    .dna-strand:nth-child(1) { top: 0%; animation-delay: 0s; }
    .dna-strand:nth-child(2) { top: 10%; animation-delay: -0.3s; }
    .dna-strand:nth-child(3) { top: 20%; animation-delay: -0.6s; }
    .dna-strand:nth-child(4) { top: 30%; animation-delay: -0.9s; }
    .dna-strand:nth-child(5) { top: 40%; animation-delay: -1.2s; }
    .dna-strand:nth-child(6) { top: 50%; animation-delay: -1.5s; }
    .dna-strand:nth-child(7) { top: 60%; animation-delay: -1.8s; }
    .dna-strand:nth-child(8) { top: 70%; animation-delay: -2.1s; }
    .dna-strand:nth-child(9) { top: 80%; animation-delay: -2.4s; }
    .dna-strand:nth-child(10) { top: 90%; animation-delay: -2.7s; }
    
    @keyframes dna-move {
        0%, 100% { transform: translateX(0px); opacity: 0.3; }
        50% { transform: translateX(50px); opacity: 1; }
    }
    
    /* ===== TYPOGRAPHY ===== */
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 6rem;
        font-weight: 800;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -4px;
        line-height: 0.9;
        position: relative;
        animation: title-reveal 1.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .main-title .highlight {
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        display: inline-block;
    }
    
    .main-title .highlight::after {
        content: '';
        position: absolute;
        bottom: 10px;
        left: 0;
        width: 100%;
        height: 8px;
        background: var(--gradient-1);
        opacity: 0.3;
        transform: skewX(-12deg);
        animation: highlight-slide 0.8s ease-out 0.5s backwards;
    }
    
    @keyframes title-reveal {
        0% { 
            opacity: 0; 
            transform: translateY(60px);
            filter: blur(20px);
        }
        100% { 
            opacity: 1; 
            transform: translateY(0);
            filter: blur(0);
        }
    }
    
    @keyframes highlight-slide {
        0% { width: 0; }
        100% { width: 100%; }
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        animation: subtitle-fade 1s ease-out 0.3s backwards;
    }
    
    @keyframes subtitle-fade {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--accent-1);
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-label::before {
        content: '';
        width: 30px;
        height: 2px;
        background: var(--gradient-1);
        animation: line-expand 0.6s ease-out;
    }
    
    @keyframes line-expand {
        0% { width: 0; }
        100% { width: 30px; }
    }
    
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2rem;
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(200, 255, 0, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(200, 255, 0, 0.2);
        transform: translateY(-8px);
        box-shadow: 
            0 30px 60px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(200, 255, 0, 0.05);
    }
    
    .glass-card:hover::before {
        opacity: 1;
    }
    
    /* ===== RESULT DISPLAYS ===== */
    .result-container {
        padding: 2.5rem;
        border-radius: 28px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: result-emerge 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    @keyframes result-emerge {
        0% { 
            opacity: 0; 
            transform: scale(0.8) rotateX(10deg);
        }
        100% { 
            opacity: 1; 
            transform: scale(1) rotateX(0);
        }
    }
    
    .result-fake {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(255, 107, 53, 0.3);
        box-shadow: 
            0 0 60px rgba(255, 107, 53, 0.15),
            inset 0 0 60px rgba(255, 107, 53, 0.05);
    }
    
    .result-real {
        background: linear-gradient(135deg, rgba(200, 255, 0, 0.1) 0%, rgba(0, 255, 200, 0.08) 100%);
        border: 1px solid rgba(200, 255, 0, 0.3);
        box-shadow: 
            0 0 60px rgba(200, 255, 0, 0.15),
            inset 0 0 60px rgba(200, 255, 0, 0.05);
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: icon-pop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s backwards;
    }
    
    @keyframes icon-pop {
        0% { transform: scale(0) rotate(-180deg); }
        100% { transform: scale(1) rotate(0); }
    }
    
    .result-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    
    .result-fake .result-label { color: var(--accent-3); }
    .result-real .result-label { color: var(--accent-1); }
    
    .result-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    
    .result-fake .result-title { 
        background: var(--gradient-2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .result-real .result-title { 
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .result-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
        position: relative;
    }
    
    .result-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: var(--progress);
        border-radius: 3px;
        animation: bar-fill 1s ease-out 0.4s backwards;
    }
    
    .result-fake .result-bar::after {
        background: var(--gradient-2);
    }
    
    .result-real .result-bar::after {
        background: var(--gradient-1);
    }
    
    @keyframes bar-fill {
        0% { width: 0; }
    }
    
    /* ===== STAT CARDS ===== */
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(200, 255, 0, 0.05), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { left: -100%; }
        50% { left: 100%; }
    }
    
    .stat-card:hover {
        border-color: rgba(200, 255, 0, 0.2);
        transform: scale(1.02);
    }
    
    .stat-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.75rem;
    }
    
    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1;
    }
    
    .stat-value.success {
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-value.danger {
        background: var(--gradient-2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ===== FEATURE CARDS ===== */
    .feature-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(200, 255, 0, 0.1), transparent 30%);
        animation: rotate-bg 10s linear infinite paused;
        opacity: 0;
    }
    
    .feature-card:hover::before {
        animation-play-state: running;
        opacity: 1;
    }
    
    .feature-card:hover {
        border-color: rgba(200, 255, 0, 0.3);
        transform: translateY(-12px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes rotate-bg {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .feature-num {
        font-family: 'Syne', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, rgba(200, 255, 0, 0.2) 0%, rgba(0, 255, 200, 0.1) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 1rem;
    }
    
    .feature-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background: rgba(200, 255, 0, 0.05);
        border: 1px solid rgba(200, 255, 0, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--gradient-1);
    }
    
    .warning-box {
        background: rgba(255, 107, 53, 0.05);
        border: 1px solid rgba(255, 107, 53, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .warning-box::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--gradient-2);
    }
    
    .box-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .info-box .box-title { color: var(--accent-1); }
    .warning-box .box-title { color: var(--accent-3); }
    
    /* ===== DIVIDER ===== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 4rem 0;
        position: relative;
    }
    
    .divider::before {
        content: '◆';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background: var(--bg-dark);
        padding: 0 20px;
        color: var(--accent-1);
        font-size: 0.8rem;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: var(--gradient-1) !important;
        color: #000 !important;
        border: none !important;
        padding: 1.2rem 3rem !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border-radius: 16px !important;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent) !important;
        transition: left 0.5s ease !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 
            0 20px 40px rgba(200, 255, 0, 0.3),
            0 0 60px rgba(200, 255, 0, 0.2) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        transition: all 0.4s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-1) !important;
        background: rgba(200, 255, 0, 0.02) !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: var(--gradient-1) !important;
        border-radius: 4px;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: rgba(12, 12, 12, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* ===== INPUTS ===== */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 20px rgba(200, 255, 0, 0.1) !important;
    }
    
    /* ===== IMAGES ===== */
    .stImage {
        border-radius: 20px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.4s ease !important;
    }
    
    .stImage:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* ===== JSON ===== */
    .stJson {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 4rem 0 2rem;
    }
    
    .footer-brand {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .footer-brand .highlight {
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer-text {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    /* ===== UPLOAD HINT ===== */
    .upload-area {
        text-align: center;
        padding: 5rem 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.01);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-area::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 50% 50%, rgba(200, 255, 0, 0.03) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .upload-area:hover {
        border-color: var(--accent-1);
    }
    
    .upload-area:hover::before {
        opacity: 1;
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    
    .upload-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .upload-sub {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }
    
    .upload-formats {
        font-size: 0.75rem;
        color: var(--text-muted);
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div > div > div {
        background: var(--accent-1) !important;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>

<!-- CREATIVE ANIMATED BACKGROUND -->
<div class="creative-bg">
    <!-- Morphing gradient blobs -->
    <div class="morph-blob blob-1"></div>
    <div class="morph-blob blob-2"></div>
    <div class="morph-blob blob-3"></div>
    
    <!-- 3D Floating shapes -->
    <div class="floating-shapes">
        <div class="shape-3d ring-shape"></div>
        
        <div class="shape-3d cube-container">
            <div class="cube">
                <div class="cube-face"></div>
                <div class="cube-face"></div>
                <div class="cube-face"></div>
                <div class="cube-face"></div>
                <div class="cube-face"></div>
                <div class="cube-face"></div>
            </div>
        </div>
    </div>
    
    <!-- Wave lines -->
    <div class="wave-container">
        <div class="wave-line"></div>
        <div class="wave-line"></div>
        <div class="wave-line"></div>
        <div class="wave-line"></div>
        <div class="wave-line"></div>
    </div>
    
    <!-- Floating orbs with trails -->
    <div class="orb-trail orb-trail-1"></div>
    <div class="orb-trail orb-trail-2"></div>
    <div class="orb-trail orb-trail-3"></div>
    
    <!-- Perspective grid -->
    <div class="perspective-grid"></div>
    
    <!-- Corner decorations -->
    <div class="corner-decor corner-tl"></div>
    <div class="corner-decor corner-br"></div>
    
    <!-- DNA helix -->
    <div class="dna-helix">
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
        <div class="dna-strand"></div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# CACHING - Load model once
# =============================================================================

@st.cache_resource
def load_model_cached():
    """Load trained model with caching."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'model.pth'
    
    try:
        if os.path.exists(model_path):
            model = get_model(pretrained=False, freeze_backbone=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model, device, True
        else:
            model = get_model(pretrained=True, freeze_backbone=False)
            model = model.to(device)
            model.eval()
            return model, device, False
    except Exception as e:
        return None, device, False


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(model, image, device, threshold=0.5):
    """Run prediction on image."""
    input_tensor = preprocess_image(image).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob_fake = output.item()
    
    if prob_fake > threshold:
        prediction = "FAKE"
        confidence = prob_fake * 100
    else:
        prediction = "REAL"
        confidence = (1 - prob_fake) * 100
    
    return prediction, confidence, prob_fake


def summarize_heatmap_focus(heatmap: np.ndarray) -> dict:
    """Create compact Grad-CAM evidence for explanations.

    Returns focus percentages by region so explanations can be tied
    to what the model actually attended to.
    """
    if heatmap is None or heatmap.size == 0:
        return {
            "max_activation": 0.0,
            "high_attention_pct": 0.0,
            "regions": {}
        }

    h, w = heatmap.shape
    h_mid, w_mid = h // 2, w // 2

    # Fraction of pixels with strong activation.
    high_attention_mask = heatmap >= 0.65
    high_attention_pct = float(np.mean(high_attention_mask) * 100.0)

    # Region map uses percentages so LLM prompt stays concise and grounded.
    regions = {
        "top_left": float(np.mean(heatmap[:h_mid, :w_mid]) * 100.0),
        "top_right": float(np.mean(heatmap[:h_mid, w_mid:]) * 100.0),
        "bottom_left": float(np.mean(heatmap[h_mid:, :w_mid]) * 100.0),
        "bottom_right": float(np.mean(heatmap[h_mid:, w_mid:]) * 100.0),
        "center": float(np.mean(heatmap[h // 4:3 * h // 4, w // 4:3 * w // 4]) * 100.0)
    }

    return {
        "max_activation": float(np.max(heatmap) * 100.0),
        "high_attention_pct": high_attention_pct,
        "regions": regions
    }


def build_local_explanation(prediction: str, confidence: float, prob_fake: float, focus_summary: dict) -> str:
    """Fallback explanation when Gemini is not available.

    This keeps the app explainable even without an API key.
    """
    regions = focus_summary.get("regions", {})
    if regions:
        top_region, top_region_score = max(regions.items(), key=lambda item: item[1])
        region_text = f"Primary model focus: {top_region.replace('_', ' ')} ({top_region_score:.1f}% avg activation)."
    else:
        region_text = "Primary model focus could not be extracted."

    high_attention_pct = focus_summary.get("high_attention_pct", 0.0)

    if prediction == "FAKE":
        verdict = (
            f"The image was classified as FAKE with {confidence:.1f}% confidence "
            f"(P(fake)={prob_fake:.2%})."
        )
        cues = (
            "This often aligns with synthetic artifacts such as inconsistent skin texture, "
            "edge blending around facial boundaries, or geometry/lighting mismatches."
        )
    else:
        verdict = (
            f"The image was classified as REAL with {confidence:.1f}% confidence "
            f"(P(fake)={prob_fake:.2%})."
        )
        cues = (
            "The model likely found coherent textures and transitions without strong signs "
            "of compositing or generative artifacts."
        )

    attention_line = (
        f"High-attention area coverage is {high_attention_pct:.1f}% of pixels; "
        f"{region_text}"
    )

    recommendation = (
        "Recommendation: treat this as probabilistic evidence and verify with metadata, "
        "source credibility, and cross-check tools when the decision is important."
    )

    return f"{verdict}\n\n{cues}\n\n{attention_line}\n\n{recommendation}"


# =============================================================================
# GEMINI EXPLAINABILITY
# =============================================================================

def get_gemini_explanation(
    image: Image.Image,
    prediction: str,
    confidence: float,
    prob_fake: float,
    api_key: str,
    focus_summary: dict
) -> str:
    """Generate natural language explanation using Google Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        regions = focus_summary.get("regions", {})
        if regions:
            top_region, top_region_score = max(regions.items(), key=lambda item: item[1])
            region_line = f"Top activation region: {top_region} ({top_region_score:.1f}%)."
        else:
            region_line = "Top activation region: unavailable."

        prompt = f"""You are an expert in deepfake detection and digital forensics. Analyze this image that has been classified by our AI model.

**Model Prediction:** {prediction}
**Confidence:** {confidence:.1f}%
**Probability of being fake:** {prob_fake:.2%}
**Max Grad-CAM activation:** {focus_summary.get('max_activation', 0.0):.1f}%
**Strong attention coverage (>=0.65):** {focus_summary.get('high_attention_pct', 0.0):.1f}%
**{region_line}**

Use the Grad-CAM evidence above to ground your explanation in model focus.

Based on the image and the model's prediction, provide a detailed but concise explanation (3-4 sentences) of:
1. What visual indicators might have led to this classification
2. How the attention map supports or weakens the model's confidence
3. Common deepfake artifacts to look for (if FAKE) or signs of authenticity (if REAL)
4. A recommendation for the user

Be informative but cautious - remind users that AI detection is not 100% accurate.
Format your response in a clear, easy-to-read manner."""

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_base64}
        ])
        
        return response.text
    except Exception as e:
        return f"⚠️ Could not generate explanation: {str(e)}"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">DEEP<span class="highlight">FAKE</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Authenticity Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        model, device, trained = load_model_cached()
        
        if model:
            st.success("✅ Model Loaded" if trained else "⚠️ Demo Mode")
        else:
            st.error("❌ Model Error")
        
        st.markdown("---")
        
        show_gradcam = st.toggle("🔥 Show Grad-CAM", value=True)
        show_ai_explanation = st.toggle("🤖 AI Explanation (Gemini)", value=False)
        
        gemini_api_key = None
        if show_ai_explanation:
            # Try to load from environment first
            env_api_key = os.getenv("GEMINI_API_KEY", "")
            
            if env_api_key:
                st.success("🔑 API Key loaded from .env")
                gemini_api_key = env_api_key
            
            # Allow manual override
            manual_key = st.text_input(
                "🔑 Gemini API Key" + (" (override)" if env_api_key else ""),
                type="password",
                help="Enter your Google Gemini API key (or set GEMINI_API_KEY in .env)"
            )
            if manual_key:
                gemini_api_key = manual_key
            
            if not gemini_api_key:
                st.caption("⚠️ Add GEMINI_API_KEY to .env file")
        
        threshold = st.slider(
            "🎯 Threshold",
            min_value=0.3, max_value=0.8, value=0.5, step=0.05
        )
        st.caption(f"If P(Fake) > {threshold} → FAKE")
        
        st.markdown("---")
        
        st.markdown("### 📖 How It Works")
        st.markdown("""
        1. **Upload** an image
        2. **AI analyzes** patterns
        3. **View results** & confidence
        4. **Grad-CAM** shows hot spots
        """)
        
        st.markdown("---")
        st.markdown(f"**Device:** `{device}`")
    
    # Main Content
    if model is None:
        st.error("Model failed to load!")
        return
    
    # Upload Section
    st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyze Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop image here", 
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Analyze Button
        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with st.spinner("🔄 Analyzing..."):
                prediction, confidence, prob_fake = predict(model, image, device, threshold)
                
                overlay = None
                heatmap = None
                if show_gradcam:
                    try:
                        overlay, heatmap, _, _ = generate_gradcam_visualization(model, image, device)
                    except:
                        pass

                focus_summary = summarize_heatmap_focus(heatmap) if heatmap is not None else {
                    "max_activation": 0.0,
                    "high_attention_pct": 0.0,
                    "regions": {}
                }
                
                # Generate Gemini explanation
                ai_explanation = None
                if show_ai_explanation and gemini_api_key:
                    with st.spinner("🤖 Generating AI explanation..."):
                        ai_explanation = get_gemini_explanation(
                            image, prediction, confidence, prob_fake, gemini_api_key, focus_summary
                        )
                elif show_ai_explanation:
                    ai_explanation = build_local_explanation(
                        prediction, confidence, prob_fake, focus_summary
                    )
            
            # Results
            with col1:
                st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Original</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Verdict</div>', unsafe_allow_html=True)
                
                if prediction == "FAKE":
                    st.markdown(f'''
                    <div class="result-container result-fake">
                        <div class="result-icon">⚠</div>
                        <div class="result-label">Analysis Complete</div>
                        <div class="result-title">Manipulation Detected</div>
                        <div class="result-bar" style="--progress: {prob_fake*100}%"></div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="result-container result-real">
                        <div class="result-icon">✓</div>
                        <div class="result-label">Analysis Complete</div>
                        <div class="result-title">Verified Authentic</div>
                        <div class="result-bar" style="--progress: {(1-prob_fake)*100}%"></div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-label">Confidence Score</div>
                    <div class="stat-value {"danger" if prediction == "FAKE" else "success"}">{confidence:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="stat-card" style="margin-top: 0.75rem;">
                    <div class="stat-label">Fake Probability</div>
                    <div class="stat-value">{prob_fake:.2%}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                if show_gradcam and overlay is not None:
                    st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Focus Map</div>', unsafe_allow_html=True)
                    st.image(overlay, use_container_width=True)
                    with st.expander("What am I looking at?"):
                        st.markdown("""
                        **Attention Levels:**
                        - 🔴 **Red** = Critical focus area
                        - 🟡 **Yellow** = Moderate attention
                        - 🔵 **Blue** = Low relevance
                        
                        The AI highlights regions that influenced its decision.
                        """)
                else:
                    st.markdown('<div class="section-label">Metrics</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Scores</div>', unsafe_allow_html=True)
                    st.metric("Fake Score", f"{prob_fake:.1%}")
                    st.metric("Real Score", f"{1-prob_fake:.1%}")
            
            # Detailed Analysis
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            dcol1, dcol2 = st.columns(2)
            
            with dcol1:
                st.markdown('<div class="section-label">Technical</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Raw Output</div>', unsafe_allow_html=True)
                st.json({
                    "prediction": prediction,
                    "confidence": f"{confidence:.2f}%",
                    "prob_fake": round(prob_fake, 4),
                    "prob_real": round(1 - prob_fake, 4),
                    "threshold": threshold
                })
            
            with dcol2:
                st.markdown('<div class="section-label">Insights</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
                
                # Show AI-powered explanation if available
                if ai_explanation:
                    st.markdown(f'''
                    <div class="{"warning-box" if prediction == "FAKE" else "info-box"}">
                        <div class="box-title">🤖 AI Analysis</div>
                        {ai_explanation}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    # Fallback to static explanation
                    if prediction == "FAKE":
                        st.markdown('''
                        <div class="warning-box">
                            <div class="box-title">⚠ Manipulation Indicators</div>
                            The model detected patterns commonly associated with synthetic or manipulated media:
                            <br><br>
                            • Unnatural facial features or expressions<br>
                            • Blending artifacts around edges<br>
                            • Texture inconsistencies in skin or hair
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                        <div class="info-box">
                            <div class="box-title">✓ Authenticity Indicators</div>
                            No significant manipulation patterns detected. The image appears to be genuine.
                            <br><br>
                            <em>Note: Always verify important media through multiple sources.</em>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Tip to enable AI explanation
                if not show_ai_explanation:
                    st.caption("💡 Enable 'AI Explanation' in sidebar for detailed analysis")
                elif show_ai_explanation and not gemini_api_key:
                    st.caption("Using local explanation (set GEMINI_API_KEY for richer AI analysis)")
    
    else:
        # Features when no image
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-label">Capabilities</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What We Detect</div>', unsafe_allow_html=True)
        
        f1, f2, f3, f4 = st.columns(4)
        
        with f1:
            st.markdown('''
            <div class="feature-card">
                <div class="feature-num">01</div>
                <div class="feature-name">Neural Network</div>
                <div class="feature-desc">ResNet18 deep learning architecture trained on facial manipulation data</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with f2:
            st.markdown('''
            <div class="feature-card">
                <div class="feature-num">02</div>
                <div class="feature-name">Real-Time</div>
                <div class="feature-desc">Instant analysis with results delivered in under 2 seconds</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with f3:
            st.markdown('''
            <div class="feature-card">
                <div class="feature-num">03</div>
                <div class="feature-name">Explainable</div>
                <div class="feature-desc">Visual heatmaps and AI-powered explanations for transparency</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with f4:
            st.markdown('''
            <div class="feature-card">
                <div class="feature-num">04</div>
                <div class="feature-name">Precision</div>
                <div class="feature-desc">High accuracy detection with adjustable confidence thresholds</div>
            </div>
            ''', unsafe_allow_html=True)
        
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">DEEP<span class="highlight">FAKE</span></div>
        <div class="footer-text">Powered by PyTorch • Streamlit • Gemini AI</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
