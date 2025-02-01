import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import './section.css';

const Section = ({ user }) => {
  const navigate = useNavigate(); // Add this line - actually initialize navigate
//   const location = useLocation();
//   const { firstName } = location.state || {};

  return (
    <div className="section-container">
      <button 
        className="section-button"
        onClick={() => console.log("Navigation works!")} // Add click handler for testing
      >
        Start AI Speech Analysis
      </button>
      <button
        className="section-button"
        onClick={() => console.log("Navigation works!")} // Add click handler for testing
      >
        I Already Know My Speech Issue
      </button>
    </div>
  );
};

export default Section;