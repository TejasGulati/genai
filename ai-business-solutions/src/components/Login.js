import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    try {
      const successMessage = await login(email, password);
      setMessage(successMessage);
      setTimeout(() => navigate('/'), 2000);
    } catch (error) {
      setMessage(error.toString());
    }
  };

  if (isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-700 to-blue-800 flex flex-col justify-center py-12 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-black opacity-50 z-0"></div>
      <div className="absolute inset-0 z-10">
        <div className="h-full w-full bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1IiBoZWlnaHQ9IjUiPgo8cmVjdCB3aWR0aD0iNSIgaGVpZ2h0PSI1IiBmaWxsPSIjMDAwMDAwNjYiPjwvcmVjdD4KPHBhdGggZD0iTTAgNUw1IDBaTTYgNEw0IDZaTS0xIDFMMSAtMVoiIHN0cm9rZT0iIzg4ODg4ODY2IiBzdHJva2Utd2lkdGg9IjEiPjwvcGF0aD4KPC9zdmc+')] opacity-10"></div>
      </div>
      <div className="sm:mx-auto sm:w-full sm:max-w-md z-20 relative">
        <h2 className="mt-6 text-center text-4xl font-extrabold text-white">Login to Your Sustainable AI Future</h2>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md z-20 relative">
        <div className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg py-8 px-4 shadow-2xl sm:rounded-lg sm:px-10 border border-white border-opacity-20">
          <form className="space-y-6" onSubmit={handleSubmit}>
            {message && (
              <div className={`rounded-md p-4 ${message.includes('successful') ? 'bg-green-400 bg-opacity-20 text-green-100' : 'bg-red-400 bg-opacity-20 text-red-100'}`}>
                {message}
              </div>
            )}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-200">
                Email address
              </label>
              <div className="mt-1">
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm bg-white bg-opacity-10 text-white"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-200">
                Password
              </label>
              <div className="mt-1">
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm bg-white bg-opacity-10 text-white"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-all duration-300 ease-in-out transform hover:scale-105"
              >
                Login
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Login;