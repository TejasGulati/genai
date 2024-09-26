import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

function Navbar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav>
      <ul>
        <li><Link to="/">Home</Link></li>
        {isAuthenticated ? (
          <>
            <li><Link to="/profile">Profile</Link></li>
            <li><Link to="/sustainability-report">Sustainability Report</Link></li>
            <li><Link to="/environmental-impact">Environmental Impact</Link></li>
            <li><Link to="/business-model">Business Model</Link></li>
            <li><Link to="/predict">Prediction</Link></li>
            <li><Link to="/generate-text">Generate Text</Link></li>
            <li><Link to="/generate-image">Generate Image</Link></li>
            <li><button onClick={handleLogout}>Logout</button></li>
          </>
        ) : (
          <>
            <li><Link to="/register">Register</Link></li>
            <li><Link to="/login">Login</Link></li>
          </>
        )}
      </ul>
    </nav>
  );
}

export default Navbar;