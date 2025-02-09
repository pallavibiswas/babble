import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { jsPDF } from "jspdf";
import "./section.css";

const Section = ({ user }) => {
  const navigate = useNavigate();
  const [lesson, setLesson] = useState("");
  const [loading, setLoading] = useState(false);
  const [analysisInProgress, setAnalysisInProgress] = useState(false);
  const [error, setError] = useState("");

  // Handle generating a lesson plan
  const handleGenerateLesson = async () => {
    const userId = "test_user";
    const userSpeechIssue = prompt("Describe your speech issue (e.g., 'trouble pronouncing R and W'):");

    if (!userSpeechIssue) {
      alert("Please enter a speech issue.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await axios.post("http://127.0.0.1:5000/generate_lesson", {
        user_id: userId,
        speech_issue: userSpeechIssue,
      });

      setLesson(response.data.lesson);
    } catch (error) {
      console.error("Error generating lesson:", error);
      setError("Failed to generate lesson. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Handle downloading the lesson as a PDF
  const handleDownloadPDF = () => {
    if (!lesson) {
      alert("No lesson to download.");
      return;
    }

    const pdf = new jsPDF();
    pdf.setFontSize(14);
    pdf.text("Generated Lesson Plan", 10, 10);
    pdf.setFontSize(12);
    pdf.text(lesson, 10, 20, { maxWidth: 180 });
    pdf.save("Speech_Lesson.pdf");
  };

  // Start AI Speech Analysis (from first file)
  const handleStartAnalysis = async () => {
    setAnalysisInProgress(true);
    setError("");

    try {
      const response = await axios.post("http://127.0.0.1:5000/start-analysis");

      if (response.data.status === "started") {
        alert("AI Speech Analysis has started!");
      } else {
        alert("Failed to start the analysis.");
      }
    } catch (error) {
      console.error("Error starting analysis:", error);
      setError("Failed to start AI Speech Analysis. Please try again.");
    } finally {
      setAnalysisInProgress(false);
    }
  };

  return (
    <div className="section-container">
      <div className="content-wrapper">
        {/* 🎭 Show the pointing person only if a lesson is generated */}
        {lesson && (
          <img
            className="pointing-person"
            src="/images/pointing_image.png"
            alt="Person Pointing"
          />
        )}

        {/* 📜 Lesson Plan Container */}
        {lesson && (
          <div className="lesson-container">
            <h3>Generated Lesson:</h3>
            <div className="lesson-box">
              <p style={{ whiteSpace: "pre-wrap" }}>{lesson}</p>
            </div>
            <button className="pdf-button" onClick={handleDownloadPDF}>
              Download as PDF
            </button>
          </div>
        )}
      </div>

      {/* 🟦 Start AI Speech Analysis Button (merged functionality) */}
      <button
        className="section-button"
        onClick={handleStartAnalysis}
        disabled={analysisInProgress}
      >
        {analysisInProgress ? "Starting Analysis..." : "Start AI Speech Analysis"}
      </button>

      {/* 📝 Generate Lesson Button */}
      <button className="section-button" onClick={handleGenerateLesson} disabled={loading}>
        {loading ? "Generating Lesson..." : "I Already Know My Speech Issue"}
      </button>

      {/* Display error messages if any */}
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default Section;