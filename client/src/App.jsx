import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SignUp from './SignUp';
import Login from './Login';
import Section from './section';


const App = () => {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/section" element={<Section />} />
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<SignUp />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;