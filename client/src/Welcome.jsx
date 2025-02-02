import React from 'react';
import './welcome.css';
import { useNavigate } from 'react-router-dom';

const Welcome = () => {
    const navigate = useNavigate();

    return (
        <div className="welcome-container">
            <div className="speech-bubble">
                <h1 className="logo-text">Babble</h1>
                <p className="subtitle">Your Personal Speech Therapist</p>
            </div>
            <div className="auth-buttons">
                <button className="login-btn" onClick={() => navigate('/login')}>Login</button>
                <button className="signup-btn" onClick={() => navigate('/signup')}>Sign Up</button>
            </div>
        </div>
    );
};

export default Welcome;
