import React, { useState, useCallback } from 'react';
import api from '../utils/api';

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
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Image Generation</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter image prompt"
          className="w-full p-2 border rounded"
          required
        />
        <button
          type="submit"
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          disabled={isLoading}
        >
          {isLoading ? 'Generating...' : 'Generate Image'}
        </button>
      </form>
      {error && <p className="text-red-500 mb-4">{error}</p>}
      {generatedImage && (
        <div>
          <h2 className="text-xl font-semibold mb-2">Generated Image</h2>
          <img
            src={generatedImage}
            alt={prompt}
            className="w-full mb-4 rounded shadow"
            onError={handleImageError}
          />
        </div>
      )}
      {aiDescription && (
        <div>
          <h2 className="text-xl font-semibold mb-2">AI Description</h2>
          <pre className="whitespace-pre-wrap bg-gray-100 p-4 rounded shadow">
            {typeof aiDescription === 'string' ? aiDescription : JSON.stringify(aiDescription, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}