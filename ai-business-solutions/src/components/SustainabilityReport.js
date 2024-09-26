import React, { useState } from 'react';
import api from '../utils/api';

function SustainabilityReport() {
  const [companyName, setCompanyName] = useState('');
  const [report, setReport] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/api/sustainability-report/', { company_name: companyName });
      setReport(response.data);
    } catch (error) {
      console.error('Error generating sustainability report:', error);
    }
  };

  return (
    <div>
      <h2>Sustainability Report</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Company Name"
          value={companyName}
          onChange={(e) => setCompanyName(e.target.value)}
          required
        />
        <button type="submit">Generate Report</button>
      </form>
      {report && (
        <div>
          <h3>Report</h3>
          <pre>{JSON.stringify(report, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default SustainabilityReport;