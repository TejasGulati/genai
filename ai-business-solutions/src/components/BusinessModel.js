import React, { useState } from 'react';
import api from '../utils/api';

function BusinessModel() {
  const [company, setCompany] = useState('');
  const [year, setYear] = useState('');
  const [model, setModel] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/api/business-model/', { company, year });
      setModel(response.data);
    } catch (error) {
      console.error('Error generating business model:', error);
    }
  };

  return (
    <div>
      <h2>Innovative Business Model Generator</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Company"
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          required
        />
        <input
          type="number"
          placeholder="Year"
          value={year}
          onChange={(e) => setYear(e.target.value)}
          required
        />
        <button type="submit">Generate Model</button>
      </form>
      {model && (
        <div>
          <h3>Business Model</h3>
          <pre>{JSON.stringify(model, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default BusinessModel;