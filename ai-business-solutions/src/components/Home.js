import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Leaf, Users, TrendingUp, Zap, BarChart2, FileText, Image } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext'; // Assuming you have this context

const buttonStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '0.75rem 1.5rem',
  fontSize: '1rem',
  fontWeight: '500',
  borderRadius: '0.375rem',
  color: 'white',
  backgroundColor: '#10B981',
  transition: 'background-color 0.3s',
  border: 'none',
  cursor: 'pointer',
};

const cardStyle = {
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  borderRadius: '0.75rem',
  padding: '1.5rem',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  transition: 'all 0.3s',
  height: '100%',
};

function FeatureCard({ title, description, icon: Icon }) {
  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ 
          width: '3rem', 
          height: '3rem', 
          borderRadius: '50%', 
          backgroundColor: '#10B981', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          marginRight: '1rem' 
        }}>
          <Icon size={24} color="white" />
        </div>
        <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: 'white', margin: 0 }}>{title}</h3>
      </div>
      <p style={{ color: '#D1D5DB', margin: 0 }}>{description}</p>
    </div>
  );
}

function Home() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const handleGetStarted = () => {
    if (isAuthenticated) {
      navigate('/dashboard');
    } else {
      navigate('/register');
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(to bottom right, #065F46, #0F766E, #1E40AF)',
      color: 'white',
      padding: '4rem 1rem'
    }}>
      <div style={{ maxWidth: '80rem', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '5rem' }}>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 4rem)', fontWeight: '800', marginBottom: '1rem' }}>
            {isAuthenticated ? "Welcome to Your AI-Enhanced Dashboard" : "AI-Enhanced Sustainable Business Solutions"}
          </h1>
          <p style={{ fontSize: 'clamp(1rem, 2vw, 1.25rem)', color: '#D1D5DB', marginBottom: '2rem' }}>
            {isAuthenticated
              ? "Explore powerful AI tools to revolutionize your business while prioritizing sustainability"
              : "Revolutionize your business with cutting-edge generative AI technologies while prioritizing environmental and social responsibility"}
          </p>
          <button onClick={handleGetStarted} style={buttonStyle}>
            {isAuthenticated ? "Go to Dashboard" : "Get Started"}
            <ArrowRight style={{ marginLeft: '0.5rem' }} size={20} />
          </button>
        </div>

        <div style={{ marginBottom: '5rem' }}>
          <h2 style={{ fontSize: '2rem', fontWeight: '700', textAlign: 'center', marginBottom: '3rem' }}>
            {isAuthenticated ? "Your AI-Powered Tools" : "Our Sustainable AI Approach"}
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
            {isAuthenticated ? (
              <>
                <FeatureCard
                  title="Sustainability Reports"
                  description="Generate comprehensive sustainability reports for your company with AI-driven insights."
                  icon={FileText}
                />
                <FeatureCard
                  title="Environmental Impact Analysis"
                  description="Analyze and visualize your company's environmental impact using advanced AI algorithms."
                  icon={Leaf}
                />
                <FeatureCard
                  title="Innovative Business Models"
                  description="Explore AI-generated innovative and sustainable business models tailored to your industry."
                  icon={TrendingUp}
                />
                <FeatureCard
                  title="Predictive Analytics"
                  description="Leverage AI-powered predictive analytics to forecast trends and make data-driven decisions."
                  icon={BarChart2}
                />
                <FeatureCard
                  title="AI Text Generation"
                  description="Create high-quality, context-aware text content for various business needs."
                  icon={FileText}
                />
                <FeatureCard
                  title="AI Image Generation"
                  description="Generate unique images and visual content to enhance your marketing and branding efforts."
                  icon={Image}
                />
              </>
            ) : (
              <>
                <FeatureCard
                  title="Environmental Impact"
                  description="Reduce your carbon footprint with AI-optimized resource management and predictive maintenance."
                  icon={Leaf}
                />
                <FeatureCard
                  title="Social Responsibility"
                  description="Implement AI-driven strategies to enhance diversity, inclusion, and positive community impact."
                  icon={Users}
                />
                <FeatureCard
                  title="Innovative Business Models"
                  description="Leverage generative AI to create disruptive, sustainable business models aligned with ESG objectives."
                  icon={TrendingUp}
                />
              </>
            )}
          </div>
        </div>

        {!isAuthenticated && (
          <div style={{ marginBottom: '5rem' }}>
            <h2 style={{ fontSize: '2rem', fontWeight: '700', textAlign: 'center', marginBottom: '3rem' }}>Why Choose Our AI Solutions?</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
              <div style={cardStyle}>
                <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: 'white', marginBottom: '1rem' }}>Sustainable Growth</h3>
                <p style={{ color: '#D1D5DB' }}>
                  Our AI solutions are designed not just for economic gain, but to promote sustainable and beneficial growth for society as a whole. We focus on long-term impacts and ensure that progress doesn't come at the expense of environmental health or social equity.
                </p>
              </div>
              <div style={cardStyle}>
                <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: 'white', marginBottom: '1rem' }}>Cutting-Edge Technology</h3>
                <p style={{ color: '#D1D5DB' }}>
                  Stay ahead of the curve with our state-of-the-art generative AI technologies. Our solutions are continuously updated to leverage the latest advancements in artificial intelligence, ensuring your business remains competitive and innovative.
                </p>
              </div>
            </div>
          </div>
        )}

        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: '2rem', fontWeight: '700', marginBottom: '2rem' }}>
            {isAuthenticated ? "Ready to Explore Your AI Tools?" : "Ready to Transform Your Business?"}
          </h2>
          <button onClick={handleGetStarted} style={{...buttonStyle, padding: '1rem 2rem', fontSize: '1.125rem'}}>
            {isAuthenticated ? "Go to Dashboard" : "Start Your Sustainable AI Journey"}
            <Zap style={{ marginLeft: '0.5rem' }} size={24} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default Home;