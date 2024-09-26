import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [errors, setErrors] = useState({});
  const { register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors({});
    try {
      await register(email, password, name);
      // Redirect or show success message
      console.log('Registration successful');
    } catch (error) {
      console.error('Registration error:', error);
      try {
        // Try to parse the error message as JSON
        const errorData = JSON.parse(error.message);
        setErrors(errorData);
      } catch {
        // If parsing fails, it's a generic error
        setErrors({ message: error.message });
      }
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Register</h2>
      {errors.message && <p style={{ color: 'red' }}>{errors.message}</p>}
      <div>
        <input
          type="text"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
        {errors.name && <p style={{ color: 'red' }}>{errors.name}</p>}
      </div>
      <div>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        {errors.email && <p style={{ color: 'red' }}>{errors.email}</p>}
      </div>
      <div>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {errors.password && <p style={{ color: 'red' }}>{errors.password}</p>}
      </div>
      <button type="submit">Register</button>
    </form>
  );
}

export default Register;