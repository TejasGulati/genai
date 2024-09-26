import React, { useState } from 'react';
import api from '../utils/api';
import { Loader2, TrendingUp } from 'lucide-react';

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

const Section = ({ title, data }) => {
  if (!data) return null;
  return (
    <div className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg border border-white border-opacity-20 rounded-lg shadow-lg p-6 mb-6 transition-all duration-300 ease-in-out hover:shadow-xl">
      <h3 className="text-xl font-semibold mb-4 text-white border-b border-white border-opacity-20 pb-2">{title}</h3>
      <RenderValue value={data} />
    </div>
  );
};

export default function Prediction() {
  const [data, setData] = useState('');
  const [datasetKey, setDatasetKey] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await api.post('/api/predict/', { data: JSON.parse(data), dataset_key: datasetKey });
      setPredictions(response.data);
    } catch (error) {
      console.error('Error making predictions:', error);
      setError('Failed to make predictions. Please check your input and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-800 to-blue-800 text-white p-8 pt-24">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl font-bold mb-8 text-center">Predictive Analytics</h2>
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex flex-col gap-4">
            <textarea
              placeholder="Enter JSON data"
              value={data}
              onChange={(e) => setData(e.target.value)}
              required
              className="w-full h-48 bg-white bg-opacity-10 border border-white border-opacity-20 p-3 rounded-md focus:ring-2 focus:ring-emerald-400 focus:border-transparent placeholder-gray-400 text-white"
            />
            <input
              type="text"
              placeholder="Dataset Key"
              value={datasetKey}
              onChange={(e) => setDatasetKey(e.target.value)}
              required
              className="w-full bg-white bg-opacity-10 border border-white border-opacity-20 p-3 rounded-md focus:ring-2 focus:ring-emerald-400 focus:border-transparent placeholder-gray-400 text-white"
            />
            <button
              type="submit"
              className="bg-emerald-500 text-white px-6 py-3 rounded-md hover:bg-emerald-600 transition duration-300 ease-in-out flex items-center justify-center shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin mr-2" size={20} />
                  Making Predictions...
                </>
              ) : (
                <>
                  <TrendingUp className="mr-2" size={20} />
                  Make Prediction
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

        {loading && (
          <div className="flex justify-center items-center mb-8">
            <div className="animate-pulse flex flex-col items-center">
              <Loader2 className="animate-spin mb-2" size={40} />
              <p className="text-gray-300">Making predictions...</p>
            </div>
          </div>
        )}

        {predictions && (
          <div className="space-y-6">
            <Section title="Predictions" data={predictions} />
          </div>
        )}
      </div>
    </div>
  );
}