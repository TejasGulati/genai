import React, { useState } from 'react';
import api from '../utils/api';
import { Loader2, Type, Sliders } from 'lucide-react';

const buttonStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '0.75rem 1.5rem',
  fontSize: '1rem',
  fontWeight: '500',
  borderRadius: '0.375rem',
  color: 'white',
  backgroundColor: '#10B981',
  transition: 'background-color 0.3s',
  border: 'none',
  cursor: 'pointer',
};

const cardStyle = {
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  borderRadius: '0.75rem',
  padding: '1.5rem',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  transition: 'all 0.3s',
  marginBottom: '1.5rem',
};

const formatKey = (key) => {
  return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

const RenderValue = ({ value }) => {
  if (typeof value === 'object' && value !== null) {
    if (Array.isArray(value)) {
      return (
        <ul style={{ listStyleType: 'disc', paddingLeft: '1.5rem', color: '#D1D5DB' }}>
          {value.map((item, index) => (
            <li key={index}>
              {typeof item === 'object' ? <RenderValue value={item} /> : item}
            </li>
          ))}
        </ul>
      );
    } else {
      return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
          {Object.entries(value).map(([key, val]) => (
            <div key={key} style={{ ...cardStyle, flex: '1 1 300px' }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: 'white', marginBottom: '0.5rem' }}>{formatKey(key)}</h4>
              <div style={{ color: '#D1D5DB' }}><RenderValue value={val} /></div>
            </div>
          ))}
        </div>
      );
    }
  }
  return <span style={{ color: '#D1D5DB' }}>{value}</span>;
};

const Section = ({ title, data }) => {
  if (!data) return null;
  return (
    <div style={cardStyle}>
      <h3 style={{ fontSize: '1.5rem', fontWeight: '700', color: 'white', marginBottom: '1rem' }}>{formatKey(title)}</h3>
      <RenderValue value={data} />
    </div>
  );
};

export default function TextGeneration() {
  const [prompt, setPrompt] = useState('');
  const [maxLength, setMaxLength] = useState(100);
  const [generatedText, setGeneratedText] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await api.post('/api/generate-text/', { prompt, max_length: maxLength });
      setGeneratedText(response.data.generated_text);
    } catch (error) {
      console.error('Error generating text:', error);
      setError('Failed to generate text. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(to bottom right, #065F46, #0F766E, #1E40AF)',
      color: 'white',
      padding: '4rem 1rem'
    }}>
      <div style={{ maxWidth: '80rem', margin: '0 auto' }}>
        <h2 style={{ fontSize: 'clamp(2rem, 5vw, 4rem)', fontWeight: '800', marginBottom: '2rem', textAlign: 'center' }}>Text Generation</h2>
        <form onSubmit={handleSubmit} style={{ marginBottom: '2rem', maxWidth: '600px', margin: '0 auto' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <textarea
              placeholder="Enter Prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              required
              style={{
                width: '100%',
                height: '8rem',
                padding: '0.75rem',
                fontSize: '1rem',
                borderRadius: '0.375rem',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                color: 'white',
                border: '1px solid rgba(255, 255, 255, 0.2)',
              }}
            />
            <div style={{ display: 'flex', gap: '1rem' }}>
              <div style={{ 
                flex: 1, 
                display: 'flex', 
                alignItems: 'center', 
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '0.375rem',
                padding: '0.5rem'
              }}>
                <Sliders style={{ color: '#D1D5DB', marginRight: '0.5rem' }} size={20} />
                <input
                  type="number"
                  placeholder="Max Length"
                  value={maxLength}
                  onChange={(e) => setMaxLength(Number(e.target.value))}
                  required
                  style={{
                    width: '100%',
                    backgroundColor: 'transparent',
                    border: 'none',
                    color: 'white',
                    fontSize: '1rem',
                  }}
                />
              </div>
              <button type="submit" style={buttonStyle} disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 style={{ marginRight: '0.5rem', animation: 'spin 1s linear infinite' }} size={20} />
                    Generating...
                  </>
                ) : (
                  <>
                    <Type style={{ marginRight: '0.5rem' }} size={20} />
                    Generate Text
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
        
        {error && (
          <div style={{ ...cardStyle, backgroundColor: 'rgba(220, 38, 38, 0.1)', color: '#FCA5A5', marginBottom: '2rem' }}>
            <p style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Error</p>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <Loader2 style={{ animation: 'spin 1s linear infinite' }} size={40} />
            <p style={{ marginTop: '1rem', color: '#D1D5DB' }}>Generating text...</p>
          </div>
        )}

        {generatedText && (
          <Section title="Generated Text" data={generatedText} />
        )}
      </div>
    </div>
  );
}