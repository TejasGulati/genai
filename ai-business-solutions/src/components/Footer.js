import React from 'react';
import { Link } from 'react-router-dom';
import { GithubIcon, TwitterIcon, LinkedinIcon } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gradient-to-r from-green-700 to-green-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-4">
            <h3 className="text-xl font-bold">EcoPulse</h3>
            <p className="text-sm text-green-100">Empowering businesses with AI-driven sustainability solutions.</p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><Link to="/" className="text-sm hover:text-green-300 transition-colors duration-300">Home</Link></li>
              <li><Link to="/sustainability-report" className="text-sm hover:text-green-300 transition-colors duration-300">Sustainability Report</Link></li>
              <li><Link to="/environmental-impact" className="text-sm hover:text-green-300 transition-colors duration-300">Environmental Impact</Link></li>
              <li><Link to="/business-model" className="text-sm hover:text-green-300 transition-colors duration-300">Business Model</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Connect With Us</h3>
            <div className="flex space-x-4">
              <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="hover:text-green-300 transition-colors duration-300" aria-label="GitHub">
                <GithubIcon size={24} />
              </a>
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="hover:text-green-300 transition-colors duration-300" aria-label="Twitter">
                <TwitterIcon size={24} />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="hover:text-green-300 transition-colors duration-300" aria-label="LinkedIn">
                <LinkedinIcon size={24} />
              </a>
            </div>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-green-600 text-center">
          <p className="text-sm">&copy; {currentYear} EcoPulse. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;