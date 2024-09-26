import React, { useState } from 'react';
import api from '../utils/api';

function TextGeneration() {
  const [prompt, setPrompt] = useState('');
  const [maxLength, setMaxLength] = useState(100);
  const [generatedText, setGeneratedText] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/api/generate-text/', { prompt, max_length: maxLength });
      setGeneratedText(response.data.generated_text);
    } catch (error) {
      console.error('Error generating text:', error);
    }
  };

  const renderGeneratedText = () => {
    if (!generatedText) return null;

    if (typeof generatedText === 'string') {
      return <p>{generatedText}</p>;
    }

    if (typeof generatedText === 'object') {
      return (
        <div>
          {Object.entries(generatedText).map(([key, value]) => (
            <p key={key}><strong>{key}:</strong> {value}</p>
          ))}
        </div>
      );
    }

    return <p>Unexpected format for generated text.</p>;
  };

  return (
    <div>
      <h2>Text Generation</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          required
        />
        <input
          type="number"
          placeholder="Max Length"
          value={maxLength}
          onChange={(e) => setMaxLength(Number(e.target.value))}
          required
        />
        <button type="submit">Generate Text</button>
      </form>
      {generatedText && (
        <div>
          <h3>Generated Text</h3>
          {renderGeneratedText()}
        </div>
      )}
    </div>
  );
}

export default TextGeneration;