import React, { useState } from 'react';
import { auth, db } from './firebase';
import { createUserWithEmailAndPassword } from 'firebase/auth';
import { doc, setDoc } from 'firebase/firestore';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import './styles.css';

const SignUp = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [FName, setFName] = useState('');
  const [LName, setLName] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = auth.currentUser

      if (user) {
        await setDoc(doc(db, 'Users', user.uid), {
          email: user.email,
          firstName: FName,
          lastName: LName,
        });

        console.log("User signed up successfully!");
        navigate("/section", { state: { firstName: FName } });
      }
    } catch (error) {
      console.error("Signup error:", error.message);
      setError(error.message);
    }
  };

  return (
    <div className="signup-container">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="signup-card"
      >
        <h2 className="signup-title">Sign Up</h2>
        {error && <p className="error-message">{error}</p>}
        <form onSubmit={handleSubmit} className="signup-form">
          <input
            type="text"
            placeholder="First Name"
            onChange={(e) => setFName(e.target.value)}
            className="signup-input"
            required
          />
          <input
            type="text"
            placeholder="Last Name"
            onChange={(e) => setLName(e.target.value)}
            className="signup-input"
            required
          />
          <input
            type="email"
            placeholder="Email"
            onChange={(e) => setEmail(e.target.value)}
            className="signup-input"
            required
          />
          <input
            type="password"
            placeholder="Password"
            onChange={(e) => setPassword(e.target.value)}
            className="signup-input"
            required
            minLength="6"
          />
          <button type="submit" className="signup-button">
            Sign Up
          </button>
        </form>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="login-box"
      >
        <p className="login-title">Already Have An Account?</p>
        <p className="login-text">
          Welcome back! Login and continue working on your progress!
        </p>
        <Link to="/login" className="login-button">
          Sign In
        </Link>
      </motion.div>
    </div>
  );
};

export default SignUp;