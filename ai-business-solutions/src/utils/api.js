import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',  // Update this to your Django backend URL
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Token has expired or is invalid
      // Clear local storage
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      
      // Remove Authorization header
      delete api.defaults.headers.common['Authorization'];
      
      // Redirect to login page
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;