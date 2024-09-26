import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

function Navbar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${scrolled ? 'bg-green-800 bg-opacity-90 backdrop-filter backdrop-blur-lg' : 'bg-transparent'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-white font-bold text-xl">
              <span className="mr-2">🌿</span>
              AI Business Solutions
            </Link>
          </div>
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <NavLink to="/" label="Home" />
              {isAuthenticated ? (
                <>
                  <NavLink to="/profile" label="Profile" />
                  <NavLink to="/sustainability-report" label="Sustainability Report" />
                  <NavLink to="/environmental-impact" label="Environmental Impact" />
                  <NavLink to="/business-model" label="Business Model" />
                  <NavLink to="/predict" label="Prediction" />
                  <NavLink to="/generate-text" label="Generate Text" />
                  <NavLink to="/generate-image" label="Generate Image" />
                  <button onClick={handleLogout} className="text-gray-300 hover:bg-green-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300">Logout</button>
                </>
              ) : (
                <>
                  <NavLink to="/register" label="Register" />
                  <NavLink to="/login" label="Login" />
                </>
              )}
            </div>
          </div>
          <div className="-mr-2 flex md:hidden">
            <button
              onClick={toggleMenu}
              type="button"
              className="bg-green-800 inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-green-800 focus:ring-white"
              aria-controls="mobile-menu"
              aria-expanded={isMenuOpen}
            >
              <span className="sr-only">Open main menu</span>
              {!isMenuOpen ? (
                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              ) : (
                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {isMenuOpen && (
        <div className="md:hidden" id="mobile-menu">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <MobileNavLink to="/" label="Home" />
            {isAuthenticated ? (
              <>
                <MobileNavLink to="/profile" label="Profile" />
                <MobileNavLink to="/sustainability-report" label="Sustainability Report" />
                <MobileNavLink to="/environmental-impact" label="Environmental Impact" />
                <MobileNavLink to="/business-model" label="Business Model" />
                <MobileNavLink to="/predict" label="Prediction" />
                <MobileNavLink to="/generate-text" label="Generate Text" />
                <MobileNavLink to="/generate-image" label="Generate Image" />
                <button onClick={handleLogout} className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-gray-300 hover:text-white hover:bg-green-700 transition-colors duration-300">Logout</button>
              </>
            ) : (
              <>
                <MobileNavLink to="/register" label="Register" />
                <MobileNavLink to="/login" label="Login" />
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}

function NavLink({ to, label }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={`px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300 ${
        isActive ? 'bg-green-700 text-white' : 'text-gray-300 hover:bg-green-600 hover:text-white'
      }`}
    >
      {label}
    </Link>
  );
}

function MobileNavLink({ to, label }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={`block px-3 py-2 rounded-md text-base font-medium transition-colors duration-300 ${
        isActive ? 'bg-green-700 text-white' : 'text-gray-300 hover:bg-green-600 hover:text-white'
      }`}
    >
      {label}
    </Link>
  );
}

export default Navbar;