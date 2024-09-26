import React from 'react';
import { Link } from 'react-router-dom';
import { GithubIcon, TwitterIcon, LinkedinIcon } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-green-700 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-4">
            <Link to="/" className="text-white font-bold text-xl flex items-center">
              <span className="mr-2 text-2xl">🌿</span>
              <span className="hidden sm:inline">EcoPulse</span>
              <span className="sm:hidden">AI BS</span>
            </Link>
            <p className="text-sm text-gray-300">Empowering businesses with AI-driven sustainability solutions.</p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <FooterLink to="/" label="Home" />
              <FooterLink to="/sustainability-report" label="Sustainability Report" />
              <FooterLink to="/environmental-impact" label="Environmental Impact" />
              <FooterLink to="/business-model" label="Business Model" />
              <FooterLink to="/predict" label="Prediction" />
              <FooterLink to="/generate-text" label="Generate Text" />
              <FooterLink to="/generate-image" label="Generate Image" />
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Connect With Us</h3>
            <div className="flex space-x-4">
              <SocialLink href="https://github.com" Icon={GithubIcon} label="GitHub" />
              <SocialLink href="https://twitter.com" Icon={TwitterIcon} label="Twitter" />
              <SocialLink href="https://linkedin.com" Icon={LinkedinIcon} label="LinkedIn" />
            </div>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-green-600 text-center">
          <p className="text-sm text-gray-300">&copy; {currentYear} EcoPulse. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

const FooterLink = ({ to, label }) => (
  <li>
    <Link
      to={to}
      className="text-gray-300 hover:bg-green-600 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300"
    >
      {label}
    </Link>
  </li>
);

const SocialLink = ({ href, Icon, label }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className="text-gray-300 hover:bg-green-600 hover:text-white p-2 rounded-md transition-colors duration-300"
    aria-label={label}
  >
    <Icon className="h-6 w-6" />
  </a>
);

export default Footer;