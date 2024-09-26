import React, { useState } from 'react';
import api from '../utils/api';
import { Loader2, Type, Sliders } from 'lucide-react';

const RenderValue = ({ value }) => {
  return <span className="text-gray-300 whitespace-pre-wrap">{value}</span>;
};

const Section = ({ title, data }) => {
  if (!data) return null;
  return (
    <div className="border border-white border-opacity-20 rounded-lg shadow-sm p-6 mb-6 bg-white bg-opacity-5 transition-all duration-300 ease-in-out hover:bg-opacity-10">
      <h3 className="text-xl font-semibold mb-4 text-white border-b border-white border-opacity-20 pb-2">{title}</h3>
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
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-800 to-blue-800 text-white p-8 pt-24">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center">Text Generation</h2>
        <form onSubmit={handleSubmit} className="mb-8 max-w-2xl mx-auto">
          <div className="flex flex-col gap-4">
            <textarea
              placeholder="Enter prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              required
              className="w-full h-32 bg-white bg-opacity-10 border border-white border-opacity-20 p-3 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm transition-all duration-300 ease-in-out text-white placeholder-gray-300"
            />
            <div className="flex gap-4">
              <div className="flex-grow flex items-center bg-white bg-opacity-10 border border-white border-opacity-20 rounded-md p-2">
                <Sliders className="text-gray-300 mr-2" size={20} />
                <input
                  type="number"
                  placeholder="Max Length"
                  value={maxLength}
                  onChange={(e) => setMaxLength(Number(e.target.value))}
                  required
                  className="w-full bg-transparent p-1 focus:outline-none text-white placeholder-gray-300"
                />
              </div>
              <button
                type="submit"
                className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 transition duration-300 ease-in-out flex items-center justify-center shadow-sm hover:shadow"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin mr-2" size={20} />
                    Generating...
                  </>
                ) : (
                  <>
                    <Type className="mr-2" size={20} />
                    Generate Text
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
        
        {error && (
          <div className="bg-red-900 bg-opacity-50 border-l-4 border-red-500 text-white p-4 mb-6 rounded-md max-w-2xl mx-auto" role="alert">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="flex justify-center items-center mb-8">
            <div className="animate-pulse flex flex-col items-center">
              <Loader2 className="animate-spin mb-2" size={40} />
              <p className="text-gray-300">Generating text...</p>
            </div>
          </div>
        )}

        {generatedText && (
          <div className="space-y-6">
            <Section title="Generated Text" data={generatedText} />
          </div>
        )}
      </div>
    </div>
  );
}