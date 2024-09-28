import React, { useState, useEffect } from 'react';
import api from '../utils/api';
import { Loader2, TrendingUp, BarChart2, FileText, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const styles = {
  button: {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '0.75rem 1.5rem',
    fontSize: '1rem',
    fontWeight: '500',
    borderRadius: '0.375rem',
    color: 'white',
    backgroundColor: '#10B981',
    transition: 'all 0.3s',
    border: 'none',
    cursor: 'pointer',
  },
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderRadius: '0.75rem',
    padding: '1.5rem',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    transition: 'all 0.3s',
    marginBottom: '1.5rem',
  },
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(to bottom right, #065F46, #0F766E, #1E40AF)',
    color: 'white',
    padding: '4rem 1rem',
    position: 'relative',
    overflow: 'hidden',
  },
  content: {
    maxWidth: '80rem',
    margin: '0 auto',
  },
  title: {
    fontSize: 'clamp(2rem, 5vw, 4rem)',
    fontWeight: '800',
    marginBottom: '2rem',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: '1.5rem',
    fontWeight: '600',
    marginBottom: '1rem',
    textAlign: 'center',
  },
  featureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1rem',
    marginBottom: '2rem',
  },
  form: {
    marginBottom: '5rem',
    maxWidth: '600px',
    margin: '0 auto',
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    fontSize: '1rem',
    borderRadius: '0.375rem',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    color: 'white',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    marginBottom: '1rem',
  },
  error: {
    backgroundColor: 'rgba(220, 38, 38, 0.1)',
    color: '#FCA5A5',
    marginBottom: '2rem',
  },
  flexContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1.5rem',
    justifyContent: 'flex-start',
  },
  flexItem: {
    flex: '1 1 300px',
    maxWidth: '100%',
  },
};

const datasets = [
  { key: 'ai_esg_alignment', label: 'AI ESG Alignment' },
  { key: 'ai_impact', label: 'AI Impact on Traditional Industries' },
  { key: 'gen_ai_business', label: 'Generative AI Business Models' },
];

const datasetFields = {
  ai_esg_alignment: [
    { name: 'company', label: 'Company Name', type: 'text' },
    { name: 'industry', label: 'Industry', type: 'text' },
    { name: 'year', label: 'Year', type: 'number' },
    { name: 'ai_esg_investment_percentage', label: 'AI ESG Investment Percentage', type: 'number', step: '0.01' },
    { name: 'primary_esg_initiative', label: 'Primary ESG Initiative', type: 'text' },
    { name: 'ai_contribution', label: 'AI Contribution', type: 'text' },
    { name: 'esg_performance_score', label: 'ESG Performance Score', type: 'number', step: '0.01', min: '0', max: '100' },
    { name: 'carbon_footprint_reduction', label: 'Carbon Footprint Reduction', type: 'number', step: '0.01' },
    { name: 'resource_efficiency_improvement', label: 'Resource Efficiency Improvement', type: 'number', step: '0.01' },
    { name: 'stakeholder_trust_index', label: 'Stakeholder Trust Index', type: 'number', step: '0.01', min: '0', max: '100' },
    { name: 'regulatory_compliance_score', label: 'Regulatory Compliance Score', type: 'number', step: '0.01', min: '0', max: '100' },
    { name: 'social_impact_score', label: 'Social Impact Score', type: 'number', step: '0.01', min: '0', max: '100' },
  ],
  ai_impact: [
    { name: 'company', label: 'Company Name', type: 'text' },
    { name: 'industry', label: 'Industry', type: 'text' },
    { name: 'year', label: 'Year', type: 'number' },
    { name: 'ai_investment_percentage', label: 'AI Investment Percentage', type: 'number', step: '0.01' },
    { name: 'traditional_process_impacted', label: 'Traditional Process Impacted', type: 'text' },
    { name: 'ai_technology_used', label: 'AI Technology Used', type: 'text' },
    { name: 'process_efficiency_improvement', label: 'Process Efficiency Improvement', type: 'number', step: '0.01' },
    { name: 'cost_savings', label: 'Cost Savings', type: 'number', step: '0.01' },
    { name: 'jobs_automated', label: 'Jobs Automated', type: 'number', step: '0.01' },
    { name: 'new_jobs_created', label: 'New Jobs Created', type: 'number', step: '0.01' },
    { name: 'product_quality_improvement', label: 'Product Quality Improvement', type: 'number', step: '0.01' },
    { name: 'time_to_market_reduction', label: 'Time to Market Reduction', type: 'number', step: '0.01' },
  ],
  gen_ai_business: [
    { name: 'company', label: 'Company Name', type: 'text' },
    { name: 'country', label: 'Country', type: 'text' },
    { name: 'industry', label: 'Industry', type: 'text' },
    { name: 'year', label: 'Year', type: 'number' },
    { name: 'ai_adoption_percentage', label: 'AI Adoption Percentage', type: 'number', step: '0.01' },
    { name: 'primary_ai_application', label: 'Primary AI Application', type: 'text' },
    { name: 'disruption_level', label: 'Disruption Level', type: 'text' },
    { name: 'revenue_growth', label: 'Revenue Growth', type: 'number', step: '0.01' },
    { name: 'cost_reduction', label: 'Cost Reduction', type: 'number', step: '0.01' },
    { name: 'esg_score', label: 'ESG Score', type: 'number', step: '0.01', min: '0', max: '100' },
    { name: 'primary_esg_impact', label: 'Primary ESG Impact', type: 'text' },
    { name: 'sustainable_growth_index', label: 'Sustainable Growth Index', type: 'number', step: '0.01', min: '0', max: '1' },
    { name: 'innovation_index', label: 'Innovation Index', type: 'number', step: '0.01', min: '0', max: '100' },
    { name: 'employee_satisfaction', label: 'Employee Satisfaction', type: 'number', step: '0.01', min: '0', max: '100' },
    { name: 'market_share_change', label: 'Market Share Change', type: 'number', step: '0.01' },
  ],
};

const formatKey = (key) => {
  return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

const cleanText = (text) => {
  if (typeof text !== 'string') return text;
  return text.replace(/\*\*/g, '').replace(/\\n/g, '\n').trim();
};
const RenderValue = ({ value }) => {
  if (value === null || value === undefined) {
    return <span style={{ color: '#D1D5DB' }}>N/A</span>;
  }

  if (typeof value === 'object') {
    if (Array.isArray(value)) {
      return (
        <ul style={{ listStyleType: 'disc', paddingLeft: '1.5rem', color: '#D1D5DB' }}>
          {value.map((item, index) => (
            <li key={index}>
              {typeof item === 'object' ? <RenderValue value={item} /> : cleanText(item)}
            </li>
          ))}
        </ul>
      );
    } else {
      return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
          {Object.entries(value).map(([key, val]) => (
            <motion.div
              key={key}
              style={{ ...styles.card, flex: '1 1 300px' }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: 'white', marginBottom: '0.5rem' }}>
                {formatKey(key)}
              </h4>
              <div style={{ color: '#D1D5DB' }}><RenderValue value={val} /></div>
            </motion.div>
          ))}
        </div>
      );
    }
  }

  if (typeof value === 'number') {
    return <span style={{ color: '#D1D5DB' }}>{value.toFixed(2)}</span>;
  }

  return <span style={{ color: '#D1D5DB', wordBreak: 'break-word' }}>{cleanText(value.toString())}</span>;
};

const Section = ({ title, data }) => {
  if (!data) return null;
  return (
    <motion.div
      style={{...styles.card, marginBottom: '2rem'}}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3 style={{ fontSize: '1.5rem', fontWeight: '700', color: 'white', marginBottom: '1.5rem' }}>
        {formatKey(title)}
      </h3>
      <div style={styles.flexContainer}>
        {Object.entries(data).map(([key, val]) => (
          <motion.div key={key} style={{ ...styles.card, ...styles.flexItem, margin: 0 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: 'white', marginBottom: '0.75rem' }}>
              {formatKey(key)}
            </h4>
            <div style={{ color: '#D1D5DB' }}><RenderValue value={val} /></div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

const FeatureCard = ({ title, description, icon: Icon }) => {
  return (
    <motion.div
      style={{
        ...styles.card,
        ...styles.flexItem,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
      }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <div style={{
        width: '60px',
        height: '60px',
        borderRadius: '50%',
        backgroundColor: '#10B981',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: '1rem'
      }}>
        <Icon size={30} color="white" />
      </div>
      <h4 style={{ fontSize: '1.2rem', fontWeight: '600', color: 'white', marginBottom: '0.5rem' }}>
        {title}
      </h4>
      <p style={{ color: '#D1D5DB' }}>{description}</p>
    </motion.div>
  );
};
export default function Prediction() {
  const [formData, setFormData] = useState({});
  const [datasetKey, setDatasetKey] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleDatasetChange = (e) => {
    const newDatasetKey = e.target.value;
    setDatasetKey(newDatasetKey);
    setFormData({}); // Reset form data when changing datasets
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const dataForApi = { data: [formData], dataset_key: datasetKey };
      const response = await api.post('/api/predict/', dataForApi);
      if (!response.data) {
        throw new Error('No data received from the server');
      }
      setResult(response.data);
    } catch (error) {
      console.error('Error making predictions:', error);
      setError('Failed to make predictions. Please check your input and try again.');
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      title: "Advanced Predictions",
      description: "Generate accurate predictions using state-of-the-art machine learning models.",
      icon: TrendingUp
    },
    {
      title: "Comprehensive Analysis",
      description: "Receive detailed analysis and insights on your predictions.",
      icon: BarChart2
    },
    {
      title: "AI-Powered Insights",
      description: "Leverage AI to interpret predictions and provide actionable recommendations.",
      icon: FileText
    },
    {
      title: "Multiple Datasets",
      description: "Support for various datasets to cater to different prediction needs.",
      icon: AlertTriangle
    },
  ];

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <motion.h2
          style={{...styles.title, marginBottom: '3rem'}}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          AI-Powered Predictive Analytics
        </motion.h2>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          style={{marginBottom: '4rem'}}
        >
          <h3 style={{...styles.subtitle, marginBottom: '2rem'}}>Our Features</h3>
          <div style={styles.flexContainer}>
            {features.map((feature, index) => (
              <FeatureCard key={index} {...feature} />
            ))}
          </div>
        </motion.div>

        <motion.form 
          onSubmit={handleSubmit} 
          style={{...styles.form, marginBottom: '4rem'}}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div>
              <label htmlFor="dataset" style={styles.label}>Select Dataset</label>
              <select
                id="dataset"
                value={datasetKey}
                onChange={handleDatasetChange}
                required
                style={styles.input}
              >
                <option value="">Select Dataset</option>
                {datasets.map(dataset => (
                  <option key={dataset.key} value={dataset.key}>{dataset.label}</option>
                ))}
              </select>
            </div>
            {datasetKey && datasetFields[datasetKey].map(field => (
              <div key={field.name}>
                <label htmlFor={field.name} style={styles.label}>
                  {field.label}
                  {field.type === 'number' && field.max && ` (${field.min} to ${field.max})`}
                </label>
                <input
                  id={field.name}
                  type={field.type}
                  name={field.name}
                  value={formData[field.name] || ''}
                  onChange={handleInputChange}
                  required
                  style={styles.input}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                />
              </div>
            ))}
            <motion.button 
              type="submit" 
              style={styles.button} 
              disabled={loading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {loading ? (
                <>
                  <Loader2 style={{ marginRight: '0.75rem', animation: 'spin 1s linear infinite' }} size={24} />
                  Making Predictions...
                </>
              ) : (
                <>
                  <TrendingUp style={{ marginRight: '0.75rem' }} size={24} />
                  Make Prediction
                </>
              )}
            </motion.button>
          </div>
        </motion.form>
        
        <AnimatePresence>
          {error && (
            <motion.div 
              style={{ ...styles.card, ...styles.error, marginBottom: '2rem' }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <p style={{ fontWeight: '600', marginBottom: '0.75rem' }}>Error</p>
              <p>{error}</p>
            </motion.div>
          )}

          {loading && (
            <motion.div 
              style={{ textAlign: 'center', marginBottom: '3rem' }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              <p style={{ marginTop: '1.5rem', color: '#D1D5DB' }}>Generating predictions and insights...</p>
            </motion.div>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              style={{marginTop: '3rem'}}
            >
              <Section title="Predictions" data={result.predictions} />
              <Section title="Feature Importance" data={result.predictions.feature_importance} />
              <Section title="AI Insights" data={result.ai_insights} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div
        style={{
          position: 'fixed',
          inset: 0,
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath opacity='.5' d='M96 95h4v1h-4v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9zm-1 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9h9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9z'/%3E%3Cpath d='M6 5V0H5v5H0v1h5v94h1V6h94V5H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          transform: `translateY(${scrollY * 0.5}px)`,
          pointerEvents: 'none',
        }}
      />
    </div>
  );
}