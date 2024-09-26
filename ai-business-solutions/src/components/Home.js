import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Leaf, Users, TrendingUp, Zap, BarChart2, FileText, Image } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';

const FeatureCard = ({ title, description, icon: Icon }) => {
  return (
    <motion.div
      className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 h-full transition-all duration-300 hover:shadow-lg hover:shadow-emerald-500/20"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <div className="flex items-center mb-4">
        <div className="w-12 h-12 rounded-full bg-emerald-500 flex items-center justify-center mr-4">
          <Icon size={24} className="text-white" />
        </div>
        <h3 className="text-xl font-semibold text-white">{title}</h3>
      </div>
      <p className="text-gray-300">{description}</p>
    </motion.div>
  );
};

const Home = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleGetStarted = () => {
    if (isAuthenticated) {
      // Dispatch a custom event to open the tools dropdown
      const event = new CustomEvent('openToolsDropdown');
      window.dispatchEvent(event);
      
      // Scroll to the top of the page to ensure the navbar is visible
      window.scrollTo(0, 0);
    } else {
      navigate('/register');
    }
  };

  const features = isAuthenticated
    ? [
        { title: "Sustainability Reports", description: "Generate comprehensive sustainability reports with AI-driven insights.", icon: FileText },
        { title: "Environmental Impact Analysis", description: "Analyze and visualize your company's environmental impact using advanced AI algorithms.", icon: Leaf },
        { title: "Innovative Business Models", description: "Explore AI-generated innovative and sustainable business models tailored to your industry.", icon: TrendingUp },
        { title: "Predictive Analytics", description: "Leverage AI-powered predictive analytics to forecast trends and make data-driven decisions.", icon: BarChart2 },
        { title: "AI Text Generation", description: "Create high-quality, context-aware text content for various business needs.", icon: FileText },
        { title: "AI Image Generation", description: "Generate unique images and visual content to enhance your marketing and branding efforts.", icon: Image },
      ]
    : [
        { title: "Environmental Impact", description: "Reduce your carbon footprint with AI-optimized resource management and predictive maintenance.", icon: Leaf },
        { title: "Social Responsibility", description: "Implement AI-driven strategies to enhance diversity, inclusion, and positive community impact.", icon: Users },
        { title: "Innovative Business Models", description: "Leverage generative AI to create disruptive, sustainable business models aligned with ESG objectives.", icon: TrendingUp },
      ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-800 via-teal-700 to-blue-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-4 leading-tight">
            {isAuthenticated ? "Welcome to Your AI-Enhanced Tools" : "AI-Enhanced Sustainable Business Solutions"}
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
            {isAuthenticated
              ? "Explore powerful AI tools to revolutionize your business while prioritizing sustainability"
              : "Revolutionize your business with cutting-edge generative AI technologies while prioritizing environmental and social responsibility"}
          </p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleGetStarted}
            className="inline-flex items-center px-6 py-3 text-lg font-medium rounded-full text-white bg-emerald-500 hover:bg-emerald-600 transition-colors duration-300"
          >
            {isAuthenticated ? "Explore Tools" : "Get Started"}
            <ArrowRight className="ml-2" size={20} />
          </motion.button>
        </motion.div>

        <div className="mb-20">
          <h2 className="text-3xl font-bold text-center mb-12">
            {isAuthenticated ? "Your AI-Powered Tools" : "Our Sustainable AI Approach"}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <FeatureCard {...feature} />
              </motion.div>
            ))}
          </div>
        </div>

        {!isAuthenticated && (
          <div className="mb-20">
            <h2 className="text-3xl font-bold text-center mb-12">Why Choose Our AI Solutions?</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <motion.div
                className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <h3 className="text-2xl font-semibold mb-4">Sustainable Growth</h3>
                <p className="text-gray-300">
                  Our AI solutions are designed not just for economic gain, but to promote sustainable and beneficial growth for society as a whole. We focus on long-term impacts and ensure that progress doesn't come at the expense of environmental health or social equity.
                </p>
              </motion.div>
              <motion.div
                className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <h3 className="text-2xl font-semibold mb-4">Cutting-Edge Technology</h3>
                <p className="text-gray-300">
                  Stay ahead of the curve with our state-of-the-art generative AI technologies. Our solutions are continuously updated to leverage the latest advancements in artificial intelligence, ensuring your business remains competitive and innovative.
                </p>
              </motion.div>
            </div>
          </div>
        )}

        <motion.div
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          <h2 className="text-3xl font-bold mb-8">
            {isAuthenticated ? "Ready to Explore Your AI Tools?" : "Ready to Transform Your Business?"}
          </h2>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleGetStarted}
            className="inline-flex items-center px-8 py-4 text-xl font-medium rounded-full text-white bg-emerald-500 hover:bg-emerald-600 transition-colors duration-300"
          >
            {isAuthenticated ? "Open Tools" : "Start Your Sustainable AI Journey"}
            <Zap className="ml-2" size={24} />
          </motion.button>
        </motion.div>
      </div>

      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath opacity='.5' d='M96 95h4v1h-4v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9zm-1 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9z'/%3E%3Cpath d='M6 5V0H5v5H0v1h5v94h1V6h94V5H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          transform: `translateY(${scrollY * 0.5}px)`,
        }}
      />
    </div>
  );
};

export default Home;