import React, { useEffect, useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import api from '../utils/api';
import { User, Mail, Loader2, Calendar, MapPin, Briefcase, Phone, Edit2, Check, X } from 'lucide-react';
import { motion } from 'framer-motion';

const EditableProfileItem = ({ icon: Icon, label, value, onSave, isEditable = true }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedValue, setEditedValue] = useState(value);

  useEffect(() => {
    setEditedValue(value);
  }, [value]);

  const handleSave = () => {
    onSave(label.toLowerCase(), editedValue);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedValue(value);
    setIsEditing(false);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg rounded-xl p-6 border border-white border-opacity-20 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-500/20"
    >
      <h4 className="text-lg font-semibold text-white mb-2 flex items-center justify-between">
        <span className="flex items-center">
          <Icon className="mr-2" size={20} />
          {label}
        </span>
        {isEditable && !isEditing && (
          <button onClick={() => setIsEditing(true)} className="text-emerald-300 hover:text-emerald-100">
            <Edit2 size={16} />
          </button>
        )}
      </h4>
      {isEditing ? (
        <div className="flex items-center">
          <input
            type="text"
            value={editedValue}
            onChange={(e) => setEditedValue(e.target.value)}
            className="bg-white bg-opacity-20 text-white border border-white border-opacity-30 rounded px-2 py-1 mr-2 flex-grow"
          />
          <button onClick={handleSave} className="text-emerald-300 hover:text-emerald-100 mr-2">
            <Check size={16} />
          </button>
          <button onClick={handleCancel} className="text-red-300 hover:text-red-100">
            <X size={16} />
          </button>
        </div>
      ) : (
        <div className="text-gray-300">{value || 'Not set'}</div>
      )}
    </motion.div>
  );
};

function UserProfile() {
  const [profile, setProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const { isAuthenticated, logout } = useAuth();

  const fetchProfile = useCallback(async () => {
    if (!isAuthenticated) {
      setIsLoading(false);
      return;
    }
    try {
      setIsLoading(true);
      const response = await api.get('/api/users/user/');
      setProfile(response.data);
      setError(null);
    } catch (error) {
      console.error('Error fetching profile:', error);
      setError(error.message || 'An error occurred while fetching your profile.');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const handleSave = async (field, value) => {
    try {
      const response = await api.patch('/api/users/user/', { [field]: value });
      setProfile(response.data);
      setError(null);
    } catch (error) {
      console.error('Error updating profile:', error);
      setError('Failed to update profile. Please try again.');
    }
  };

  const handleRefresh = () => {
    fetchProfile();
  };

  const handleLogout = async () => {
    try {
      await logout();
      setProfile(null);
    } catch (error) {
      console.error('Error logging out:', error);
      setError('Failed to log out. Please try again.');
    }
  };


  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-700 to-blue-800 flex justify-center items-center">
        <Loader2 className="animate-spin text-white" size={48} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-700 to-blue-800 flex justify-center items-center">
        <div className="bg-red-500 text-white p-4 rounded-md">
          <h2 className="text-xl font-bold mb-2">Error</h2>
          <p>{error}</p>
          <button onClick={handleRefresh} className="mt-4 bg-white text-red-500 px-4 py-2 rounded">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !profile) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-700 to-blue-800 flex justify-center items-center">
        <div className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg rounded-xl p-8 border border-white border-opacity-20 text-white">
          <h2 className="text-2xl font-bold mb-4">Authentication Error</h2>
          <p>There was an issue accessing your profile. Please try logging in again.</p>
          <button onClick={handleRefresh} className="mt-4 bg-teal-500 text-white px-4 py-2 rounded">
            Refresh Profile
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-700 to-blue-800 text-white py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <motion.h2 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-4xl font-extrabold text-center mb-10"
        >
          User Profile
        </motion.h2>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg rounded-xl p-8 border border-white border-opacity-20 shadow-xl"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <EditableProfileItem icon={User} label="Username" value={profile.username} onSave={handleSave} />
            <EditableProfileItem icon={Mail} label="Email" value={profile.email} onSave={handleSave} />
            <EditableProfileItem icon={Calendar} label="Joined" value={new Date(profile.date_joined).toLocaleDateString()} isEditable={false} />
            <EditableProfileItem icon={MapPin} label="Location" value={profile.location} onSave={handleSave} />
            <EditableProfileItem icon={Briefcase} label="Company" value={profile.company} onSave={handleSave} />
            <EditableProfileItem icon={Phone} label="Phone" value={profile.phone} onSave={handleSave} />
          </div>
          <div className="mt-8 flex justify-between">
            <button onClick={handleRefresh} className="bg-teal-500 hover:bg-teal-600 text-white px-4 py-2 rounded transition-colors duration-300">
              Refresh Profile
            </button>
            <button onClick={handleLogout} className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded transition-colors duration-300">
              Logout
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default UserProfile;