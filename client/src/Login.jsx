import React, { useState } from 'react';
import { auth } from './firebase'; 
import { signInWithEmailAndPassword } from 'firebase/auth'; 
import { Link, useNavigate } from 'react-router-dom'; 
import './styles.css'; 

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate(); // For navigation after successful login

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); // Clear any previous errors

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
      <div className="left-panel">
        <form className="login-form" onSubmit={handleSubmit}>
          <h2>Login</h2>
          {error && <p className="error-message">{error}</p>} 
          <label htmlFor="email">
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>
          <label htmlFor="password">
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          <button type="submit">Login</button>
        </form>
      </div>
      <div className="right-panel">
        <h2>Don't Have An Account?</h2>
        <p>Sign up and start building your projects with ease</p>
        <button type="button">
          <Link to="/signup" className="link">Sign Up</Link>
        </button>
      </div>
    </div>
  );
};

export default Login;