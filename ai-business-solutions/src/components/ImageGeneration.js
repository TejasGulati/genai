import React, { useState, useCallback } from 'react';
import api from '../utils/api';
import { Loader2, Image as ImageIcon } from 'lucide-react';

const RenderValue = ({ value }) => {
  if (typeof value === 'object' && value !== null) {
    if (Array.isArray(value)) {
      return (
        <ul className="list-disc pl-5 space-y-2">
          {value.map((item, index) => (
            <li key={index} className="text-gray-200">
              {typeof item === 'object' ? <RenderValue value={item} /> : item}
            </li>
          ))}
        </ul>
      );
    } else {
      return (
        <div className="grid grid-cols-1 gap-4">
          {Object.entries(value).map(([key, val]) => (
            <div key={key} className="bg-white bg-opacity-10 p-4 rounded-lg">
              <h4 className="text-lg font-semibold text-white mb-2">{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</h4>
              <div className="text-gray-300"><RenderValue value={val} /></div>
            </div>
          ))}
        </div>
      );
    }
  }
  return <span className="text-gray-200 whitespace-pre-wrap">{value}</span>;
};

const Section = ({ title, children }) => {
  return (
    <div className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg border border-white border-opacity-20 rounded-lg shadow-lg p-6 mb-6 transition-all duration-300 ease-in-out hover:shadow-xl">
      <h3 className="text-xl font-semibold mb-4 text-white border-b border-white border-opacity-20 pb-2">{title}</h3>
      {children}
    </div>
  );
};

export default function ImageGeneration() {
  const [prompt, setPrompt] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [aiDescription, setAiDescription] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchImage = useCallback(async (imageUrl, retries = 3) => {
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Failed to fetch image');
      const blob = await response.blob();
      return URL.createObjectURL(blob);
    } catch (error) {
      if (retries > 0) {
        console.log(`Retrying image fetch. Attempts left: ${retries - 1}`);
        await new Promise(resolve => setTimeout(resolve, 1000));
        return fetchImage(imageUrl, retries - 1);
      }
      throw error;
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setGeneratedImage(null);
    setAiDescription(null);

    try {
      const response = await api.post('/api/generate-image/', { prompt });
      console.log('Server response:', response.data);

      if (!response.data || !response.data.image_url) {
        throw new Error('Invalid server response: missing image_url');
      }

      try {
        const imageUrl = await fetchImage(response.data.image_url);
        setGeneratedImage(imageUrl);
      } catch (fetchError) {
        console.error('Error fetching image:', fetchError);
        setError('Failed to load the generated image. Please try again.');
      }

      if (response.data.ai_description) {
        setAiDescription(response.data.ai_description);
      } else {
        console.warn('AI description not provided in the response');
      }
    } catch (error) {
      console.error('Error generating or fetching image:', error);
      setError(`Failed to generate or fetch image: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageError = (e) => {
    console.error('Error loading image:', e);
    setError('Failed to load image. Please try again.');
    e.target.style.display = 'none';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-800 to-blue-800 text-white p-8 pt-24">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl font-bold mb-8 text-center">AI Image Generation</h2>
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex flex-col sm:flex-row gap-4">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter image prompt"
              required
              className="flex-grow bg-white bg-opacity-10 border border-white border-opacity-20 p-3 rounded-md focus:ring-2 focus:ring-emerald-400 focus:border-transparent placeholder-gray-400 text-white"
            />
            <button
              type="submit"
              className="bg-emerald-500 text-white px-6 py-3 rounded-md hover:bg-emerald-600 transition duration-300 ease-in-out flex items-center justify-center min-w-[150px] shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin mr-2" size={20} />
                  Generating...
                </>
              ) : (
                <>
                  <ImageIcon className="mr-2" size={20} />
                  Generate Image
                </>
              )}
            </button>
          </div>
        </form>
        
        {error && (
          <div className="bg-red-500 bg-opacity-20 border-l-4 border-red-500 text-white p-4 mb-6 rounded-md" role="alert">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="flex justify-center items-center mb-8">
            <div className="animate-pulse flex flex-col items-center">
              <Loader2 className="animate-spin mb-2" size={40} />
              <p className="text-gray-300">Generating image...</p>
            </div>
          </div>
        )}

        {generatedImage && (
          <Section title="Generated Image">
            <img
              src={generatedImage}
              alt={prompt}
              className="w-full mb-4 rounded-lg shadow-lg"
              onError={handleImageError}
            />
          </Section>
        )}

        {aiDescription && (
          <Section title="AI Description">
            <RenderValue value={aiDescription} />
          </Section>
        )}
      </div>
    </div>
  );
}