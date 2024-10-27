import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import Brain from './components/Brain';

function App() {
  const [text, setText] = useState("");
  const [submittedText, setSubmittedText] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleSubmit = async () => {
    setSubmittedText(text);

    // Send POST request to Flask backend
    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      setAnalysisResult(data); // Store analysis result in state
    } catch (error) {
      console.error("Error fetching analysis:", error);
    }

    setText("");
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden',
      padding: '1rem'
    }}>
      {/* Background 3D Brain */}
      <Canvas style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1,
      }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[2, 2, 5]} intensity={1} />
        <Brain/>
      </Canvas>

      {/* Main Content */}
      <h1>How are you feeling today?</h1>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter your feelings here..."
        style={{
          padding: '0.5rem',
          fontSize: '1rem',
          width: '80%',
          maxWidth: '400px',
          marginBottom: '1rem',
          textAlign: 'center'
        }}
      />
      <button
        onClick={handleSubmit}
        style={{
          padding: '0.5rem 1rem',
          fontSize: '1rem',
          cursor: 'pointer',
          marginTop: '0.5rem'
        }}
      >
        Submit
      </button>

      {submittedText && (
        <p style={{ marginTop: '1rem' }}>Your Submitted Input: {submittedText}</p>
      )}

      {analysisResult && (
        <div style={{ marginTop: '1rem', textAlign: 'left', maxWidth: '400px' }}>
          <h3>Analysis Result:</h3>
          <p><strong>Polarity:</strong> {analysisResult.polarity.label} (Score: {analysisResult.polarity.score})</p>
          <p><strong>Concerns:</strong> {analysisResult.concerns.join(', ') || 'None'}</p>
          <p><strong>Intensity:</strong> {analysisResult.intensity}</p>
          <h4>Timeline:</h4>
          <ul>
            {analysisResult.timeline.map((entry, index) => (
              <li key={index}>{entry.description}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
