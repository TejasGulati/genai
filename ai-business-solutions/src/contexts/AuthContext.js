import React, { createContext, useState, useContext, useEffect, useCallback, useRef } from 'react';
import api from '../utils/api';

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  const logoutRef = useRef(null);

  logoutRef.current = async () => {
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
      localStorage.removeItem('refreshToken');
      delete api.defaults.headers.common['Authorization'];
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  const refreshToken = useCallback(async () => {
    const refreshToken = localStorage.getItem('refreshToken');
    if (refreshToken) {
      try {
        const response = await api.post('/api/users/token/refresh/', { refresh: refreshToken });
        const { access } = response.data;
        localStorage.setItem('token', access);
        api.defaults.headers.common['Authorization'] = `Bearer ${access}`;
        setIsAuthenticated(true);
        return true;
      } catch (error) {
        console.error('Token refresh failed:', error);
        logoutRef.current();
        return false;
      }
    }
    return false;
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
          await refreshToken();
        }
      }
      setLoading(false);
    };

    checkAuth();
  }, [refreshToken]);

  const login = async (email, password) => {
    try {
      const response = await api.post('/api/users/login/', { email, password });
      const { access, refresh } = response.data;
      localStorage.setItem('token', access);
      localStorage.setItem('refreshToken', refresh);
      api.defaults.headers.common['Authorization'] = `Bearer ${access}`;
      setUser(response.data.user);
      setIsAuthenticated(true);
      return response.data;
    } catch (error) {
      console.error('Login error:', error.response?.data || error.message);
      throw error.response?.data || { message: 'An unexpected error occurred during login.' };
    }
  };

  const register = async (email, password, name) => {
    try {
      const response = await api.post('/api/users/register/', { email, password, name });
      return response.data;
    } catch (error) {
      console.error('Registration error:', error.response?.data || error.message);
      if (error.response && error.response.data) {
        // Throw an Error object with the response data as its message
        throw new Error(JSON.stringify(error.response.data));
      } else {
        throw new Error('An unexpected error occurred during registration.');
      }
    }
  };

  useEffect(() => {
    const interceptor = api.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401 && !error.config._retry) {
          error.config._retry = true;
          if (await refreshToken()) {
            return api(error.config);
          }
        }
        return Promise.reject(error);
      }
    );

    return () => api.interceptors.response.eject(interceptor);
  }, [refreshToken]);

  const value = {
    user,
    isAuthenticated,
    login,
    register,
    logout: useCallback(() => logoutRef.current(), []),
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}