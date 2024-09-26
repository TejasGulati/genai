// File: src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './components/Home';
import Register from './components/Register';
import Login from './components/Login';
import UserProfile from './components/UserProfile';
import SustainabilityReport from './components/SustainabilityReport';
import EnvironmentalImpact from './components/EnvironmentalImpact';
import BusinessModel from './components/BusinessModel';
import Prediction from './components/Prediction';
import TextGeneration from './components/TextGeneration';
import ImageGeneration from './components/ImageGeneration';
import PrivateRoute from './components/PrivateRoute';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/register" element={<Register />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/profile"
            element={
              <PrivateRoute>
                <UserProfile />
              </PrivateRoute>
            }
          />
          <Route
            path="/sustainability-report"
            element={
              <PrivateRoute>
                <SustainabilityReport />
              </PrivateRoute>
            }
          />
          <Route
            path="/environmental-impact"
            element={
              <PrivateRoute>
                <EnvironmentalImpact />
              </PrivateRoute>
            }
          />
          <Route
            path="/business-model"
            element={
              <PrivateRoute>
                <BusinessModel />
              </PrivateRoute>
            }
          />
          <Route
            path="/predict"
            element={
              <PrivateRoute>
                <Prediction />
              </PrivateRoute>
            }
          />
          <Route
            path="/generate-text"
            element={
              <PrivateRoute>
                <TextGeneration />
              </PrivateRoute>
            }
          />
          <Route
            path="/generate-image"
            element={
              <PrivateRoute>
                <ImageGeneration />
              </PrivateRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;