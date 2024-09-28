import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Twitter, Linkedin } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-green-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
          <div className="space-y-4">
            <Link to="/" className="text-white font-bold text-2xl flex items-center">
              <span className="mr-2 text-3xl">🌿</span>
              <span className="hidden sm:inline">EcoPulse</span>
              <span className="sm:hidden">AI BS</span>
            </Link>
            <p className="text-sm text-green-200">Empowering businesses with AI-driven sustainability solutions.</p>
            <div className="flex space-x-4 mt-4">
              <SocialLink href="https://github.com" Icon={Github} label="GitHub" />
              <SocialLink href="https://twitter.com" Icon={Twitter} label="Twitter" />
              <SocialLink href="https://linkedin.com" Icon={Linkedin} label="LinkedIn" />
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4 text-green-300">Quick Links</h3>
            <ul className="space-y-2">
              <FooterLink to="/" label="Home" />
              <FooterLink to="/sustainability-report" label="Sustainability Report" />
              <FooterLink to="/environmental-impact" label="Environmental Impact" />
              <FooterLink to="/business-model" label="Business Model" />
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4 text-green-300">Our Tools</h3>
            <ul className="space-y-2">
              <FooterLink to="/predict" label="Prediction" />
              <FooterLink to="/generate-text" label="Generate Text" />
              <FooterLink to="/generate-image" label="Generate Image" />
            </ul>
          </div>
        </div>
        <div className="mt-12 pt-8 border-t border-green-700 text-center">
          <p className="text-sm text-green-200">&copy; {currentYear} EcoPulse. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

const FooterLink = ({ to, label }) => (
  <li>
    <Link
      to={to}
      className="text-green-200 hover:text-white transition-colors duration-300"
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
    className="text-green-200 hover:text-white transition-colors duration-300"
    aria-label={label}
  >
    <Icon className="h-5 w-5" />
  </a>
);

export default Footer;