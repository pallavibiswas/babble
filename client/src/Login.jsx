import React, { useState } from 'react';
import { auth } from './firebase'; 
import { signInWithEmailAndPassword } from 'firebase/auth'; 
import { Link, useNavigate } from 'react-router-dom'; 
import './styles.css'; 

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      await signInWithEmailAndPassword(auth, email, password);
      console.log('Login successful!');
      navigate('/');
    } catch (error) {
      setError(error.message);
      console.error('Login error:', error);
    }
  };

  return (
    <div className="login-container">
      {/* Left Side - Login Form */}
      <div className="left-panel">
        <h2>Login</h2>
        <p>
          Don’t have an account yet? <Link to="/signup" className="link">Sign Up</Link>
        </p>

        {error && <p className="error-message">{error}</p>}

        <form className="login-form" onSubmit={handleSubmit}>
          <label>Email Address</label>
          <input
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          
          <label>Password</label>
          <input
            type="password"
            placeholder="Enter 6 characters or more"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <Link to="/forgot-password" className="forgot-password">Forgot Password?</Link>

          <button type="submit">Login</button>
        </form>
      </div>

      {/* Right Side - Two Separate Images with Spacing */}
      <div className="right-panel">
        <div className="image-container">
          <img src="/images/speech_therapist_1.png" alt="Person 1" className="right-image" />
          <img src="/images/speech_therapist_2.png" alt="Person 2" className="right-image" />
        </div>
      </div>
    </div>
  );
};

export default Login;
