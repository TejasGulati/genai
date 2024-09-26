import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import api from '../utils/api';

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  const logout = useCallback(async () => {
    try {
      await api.post('/api/users/logout/', {}, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      localStorage.removeItem('token');
      delete api.defaults.headers.common['Authorization'];
      setUser(null);
      setIsAuthenticated(false);
    }
  }, []);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      if (token) {
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        try {
          const response = await api.get('/api/users/user/');
          setUser(response.data);
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Auth check failed:', error);
          logout();
        }
      }
      setLoading(false);
    };

    checkAuth();
  }, [logout]);

  const login = async (email, password) => {
    try {
      const response = await api.post('/api/users/login/', { email, password });
      const { access } = response.data;
      localStorage.setItem('token', access);
      api.defaults.headers.common['Authorization'] = `Bearer ${access}`;
      const userResponse = await api.get('/api/users/user/');
      setUser(userResponse.data);
      setIsAuthenticated(true);
      return "Login successful!";
    } catch (error) {
      console.error('Login error:', error.response?.data || error.message);
      throw error.response?.data?.detail || 'An unexpected error occurred during login.';
    }
  };

  const register = async (email, password, name, username, location, company, phone) => {
    try {
      const response = await api.post('/api/users/register/', { 
        email, 
        password, 
        name, 
        username, 
        location, 
        company, 
        phone 
      });
      return `Registration successful for ${response.data.email || email}! Please log in.`;
    } catch (error) {
      console.error('Registration error:', error.response?.data || error.message);
      if (error.response && error.response.data) {
        throw error.response.data;
      } else {
        throw new Error('An unexpected error occurred during registration.');
      }
    }
  };

  const value = {
    user,
    isAuthenticated,
    login,
    register,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}