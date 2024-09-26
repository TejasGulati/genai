import React, { useState } from 'react';
import api from '../utils/api';

function EnvironmentalImpact() {
  const [company, setCompany] = useState('');
  const [year, setYear] = useState('');
  const [impact, setImpact] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/api/environmental-impact/', { company, year });
      setImpact(response.data);
    } catch (error) {
      console.error('Error analyzing environmental impact:', error);
    }
  };

  return (
    <div>
      <h2>Environmental Impact Analysis</h2>
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
        <button type="submit">Analyze Impact</button>
      </form>
      {impact && (
        <div>
          <h3>Impact Analysis</h3>
          <pre>{JSON.stringify(impact, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default EnvironmentalImpact;
