import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ChevronDown } from 'lucide-react';

function Navbar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

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

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  useEffect(() => {
    const handleOpenToolsDropdown = () => {
      setDropdownOpen(true);
    };

    window.addEventListener('openToolsDropdown', handleOpenToolsDropdown);

    return () => {
      window.removeEventListener('openToolsDropdown', handleOpenToolsDropdown);
    };
  }, []);

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${
      scrolled ? 'bg-green-700 bg-opacity-90 backdrop-filter backdrop-blur-lg shadow-lg' : 'bg-transparent'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-white font-bold text-xl flex items-center">
              <span className="mr-2 text-2xl">🌿</span>
              <span className="hidden sm:inline">EcoPulse</span>
              <span className="sm:hidden">AI BS</span>
            </Link>
          </div>
          <div className="hidden md:block">
            <div className="ml-10 flex items-center space-x-4">
              <NavLink to="/" label="Home" />
              {isAuthenticated ? (
                <>
                  <div className="relative" ref={dropdownRef}>
                    <button
                      onClick={() => setDropdownOpen(!dropdownOpen)}
                      className="text-gray-300 hover:bg-green-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300 flex items-center"
                    >
                      Tools <ChevronDown className="ml-1 h-4 w-4" />
                    </button>
                    <AnimatePresence>
                      {dropdownOpen && (
                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          transition={{ duration: 0.2 }}
                          className="absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 divide-y divide-gray-100 focus:outline-none"
                        >
                          <div className="py-1">
                            <DropdownLink to="/sustainability-report" label="Sustainability Report" />
                            <DropdownLink to="/environmental-impact" label="Environmental Impact" />
                            <DropdownLink to="/business-model" label="Business Model" />
                            <DropdownLink to="/predict" label="Prediction" />
                            <DropdownLink to="/generate-text" label="Generate Text" />
                            <DropdownLink to="/generate-image" label="Generate Image" />
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  <NavLink to="/profile" label="Profile" />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleLogout}
                    className="text-gray-300 hover:bg-green-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300"
                  >
                    Logout
                  </motion.button>
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
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleMenu}
              type="button"
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
              aria-controls="mobile-menu"
              aria-expanded={isMenuOpen}
            >
              <span className="sr-only">Open main menu</span>
              {!isMenuOpen ? <Menu className="block h-6 w-6" /> : <X className="block h-6 w-6" />}
            </motion.button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="md:hidden"
            id="mobile-menu"
          >
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
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleLogout}
                    className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-gray-300 hover:text-white hover:bg-green-700 transition-colors duration-300"
                  >
                    Logout
                  </motion.button>
                </>
              ) : (
                <>
                  <MobileNavLink to="/register" label="Register" />
                  <MobileNavLink to="/login" label="Login" />
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}

function NavLink({ to, label }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
      <Link
        to={to}
        className={`px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300 ${
          isActive ? 'bg-green-700 text-white' : 'text-gray-300 hover:bg-green-600 hover:text-white'
        }`}
      >
        {label}
      </Link>
    </motion.div>
  );
}

function MobileNavLink({ to, label }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
      <Link
        to={to}
        className={`block px-3 py-2 rounded-md text-base font-medium transition-colors duration-300 ${
          isActive ? 'bg-green-700 text-white' : 'text-gray-300 hover:bg-green-600 hover:text-white'
        }`}
      >
        {label}
      </Link>
    </motion.div>
  );
}

function DropdownLink({ to, label }) {
  return (
    <Link
      to={to}
      className="block px-4 py-2 text-sm text-gray-800 hover:bg-gray-100 hover:text-gray-900"
      role="menuitem"
    >
      {label}
    </Link>
  );
}

export default Navbar;