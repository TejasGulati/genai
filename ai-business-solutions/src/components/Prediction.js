import React, { useState } from 'react';
import api from '../utils/api';

function Prediction() {
  const [data, setData] = useState('');
  const [datasetKey, setDatasetKey] = useState('');
  const [predictions, setPredictions] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/api/predict/', { data: JSON.parse(data), dataset_key: datasetKey });
      setPredictions(response.data);
    } catch (error) {
      console.error('Error making predictions:', error);
    }
  };

  return (
    <div>
      <h2>Predictive Analytics</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter JSON data"
          value={data}
          onChange={(e) => setData(e.target.value)}
          required
        />
        <input
          type="text"
          placeholder="Dataset Key"
          value={datasetKey}
          onChange={(e) => setDatasetKey(e.target.value)}
          required
        />
        <button type="submit">Make Prediction</button>
      </form>
      {predictions && (
        <div>
          <h3>Predictions</h3>
          <pre>{JSON.stringify(predictions, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default Prediction;