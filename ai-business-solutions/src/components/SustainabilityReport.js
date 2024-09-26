import React, { useState } from 'react';
import api from '../utils/api';
import { Loader2, FileText } from 'lucide-react';

const RenderValue = ({ value }) => {
  if (typeof value === 'object' && value !== null) {
    if (Array.isArray(value)) {
      return (
        <ul className="list-disc pl-5 space-y-2">
          {value.map((item, index) => (
            <li key={index} className="text-gray-300">
              {typeof item === 'object' ? <RenderValue value={item} /> : item}
            </li>
          ))}
        </ul>
      );
    } else {
      return (
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(value).map(([key, val]) => (
            <div key={key} className="bg-white bg-opacity-10 p-3 rounded-md">
              <dt className="font-semibold text-gray-200 mb-1">{key.replace(/_/g, ' ')}</dt>
              <dd className="text-gray-300"><RenderValue value={val} /></dd>
            </div>
          ))}
        </dl>
      );
    }
  }
  return <span className="text-gray-300">{value}</span>;
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

export default function SustainabilityReport() {
  const [companyName, setCompanyName] = useState('');
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await api.post('/api/sustainability-report/', { company_name: companyName });
      setReport(response.data);
    } catch (error) {
      console.error('Error generating sustainability report:', error);
      setError('Failed to generate report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-800 to-blue-800 text-white p-8 pt-24">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center">Sustainability Report</h2>
        <form onSubmit={handleSubmit} className="mb-8 max-w-2xl mx-auto">
          <div className="flex flex-col sm:flex-row gap-4">
            <input
              type="text"
              placeholder="Company Name"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              required
              className="flex-grow bg-white bg-opacity-10 border border-white border-opacity-20 p-3 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm transition-all duration-300 ease-in-out text-white placeholder-gray-300"
            />
            <button
              type="submit"
              className="bg-green-600 text-white px-6 py-3 rounded-md hover:bg-green-700 transition duration-300 ease-in-out flex items-center justify-center min-w-[150px] shadow-sm hover:shadow"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin mr-2" size={20} />
                  Generating...
                </>
              ) : (
                <>
                  <FileText className="mr-2" size={20} />
                  Generate Report
                </>
              )}
            </button>
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
              <p className="text-gray-300">Generating report...</p>
            </div>
          </div>
        )}

        {report && (
          <div className="space-y-6">
            {Object.entries(report).map(([key, value]) => (
              <Section key={key} title={key.replace(/_/g, ' ')} data={value} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}