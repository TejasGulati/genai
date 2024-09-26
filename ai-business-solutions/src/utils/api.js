import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',  // Update this to your Django backend URL
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    if (error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      try {
        const refreshToken = localStorage.getItem('refreshToken');
        const response = await api.post('/api/users/refresh/', { refresh: refreshToken });
        const { access } = response.data;
        localStorage.setItem('token', access);
        api.defaults.headers.common['Authorization'] = `Bearer ${access}`;
        return api(originalRequest);
      } catch (refreshError) {
        // Refresh token is invalid, logout the user
        localStorage.removeItem('token');
        localStorage.removeItem('refreshToken');
        delete api.defaults.headers.common['Authorization'];
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    return Promise.reject(error);
  }
);

export default api;
