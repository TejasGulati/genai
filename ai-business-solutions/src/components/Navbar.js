import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ChevronDown, User, LogOut } from 'lucide-react';

function Navbar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const mobileMenuRef = useRef(null);

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

    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setIsMenuOpen(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
      if (mobileMenuRef.current && !mobileMenuRef.current.contains(event.target) && !event.target.closest('button')) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${
      scrolled || isMenuOpen ? 'bg-green-800 shadow-lg' : 'bg-transparent'
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
                    <motion.button
                      onClick={() => setDropdownOpen(!dropdownOpen)}
                      className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300 flex items-center"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Tools <ChevronDown className="ml-1 h-4 w-4" />
                    </motion.button>
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
                  <NavLink to="/profile" icon={<User className="w-4 h-4 mr-1" />} label="Profile" />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleLogout}
                    className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300 flex items-center"
                  >
                    <LogOut className="w-4 h-4 mr-1" /> Logout
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
              className="inline-flex items-center justify-center p-2 rounded-md text-white hover:text-white hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
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
            ref={mobileMenuRef}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="md:hidden bg-green-800 shadow-lg overflow-hidden"
            id="mobile-menu"
          >
            <div className="px-4 pt-2 pb-3 space-y-1 sm:px-3">
              <MobileNavLink to="/" label="Home" onClick={() => setIsMenuOpen(false)} />
              {isAuthenticated ? (
                <>
                  <MobileNavLink to="/profile" label="Profile" icon={<User className="w-4 h-4 mr-2" />} onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/sustainability-report" label="Sustainability Report" onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/environmental-impact" label="Environmental Impact" onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/business-model" label="Business Model" onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/predict" label="Prediction" onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/generate-text" label="Generate Text" onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/generate-image" label="Generate Image" onClick={() => setIsMenuOpen(false)} />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => {
                      handleLogout();
                      setIsMenuOpen(false);
                    }}
                    className="flex items-center w-full text-left px-3 py-2 rounded-md text-base font-medium text-white hover:bg-green-700 transition-colors duration-300"
                  >
                    <LogOut className="w-4 h-4 mr-2" /> Logout
                  </motion.button>
                </>
              ) : (
                <>
                  <MobileNavLink to="/register" label="Register" onClick={() => setIsMenuOpen(false)} />
                  <MobileNavLink to="/login" label="Login" onClick={() => setIsMenuOpen(false)} />
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}

function NavLink({ to, label, icon }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
      <Link
        to={to}
        className={`px-3 py-2 rounded-md text-sm font-medium transition-colors duration-300 flex items-center ${
          isActive ? 'bg-green-700 text-white' : 'text-gray-300 hover:text-white hover:bg-green-700'
        }`}
      >
        {icon && <span className="mr-1">{icon}</span>}
        {label}
      </Link>
    </motion.div>
  );
}

function MobileNavLink({ to, label, icon, onClick }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
      <Link
        to={to}
        onClick={onClick}
        className={`flex items-center px-3 py-2 rounded-md text-base font-medium transition-colors duration-300 ${
          isActive ? 'bg-green-700 text-white' : 'text-white hover:bg-green-700'
        }`}
      >
        {icon && <span className="mr-2">{icon}</span>}
        {label}
      </Link>
    </motion.div>
  );
}

function DropdownLink({ to, label }) {
  return (
    <Link
      to={to}
      className="block px-4 py-2 text-sm text-gray-800 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-300"
      role="menuitem"
    >
      {label}
    </Link>
  );
}

export default Navbar;