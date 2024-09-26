import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import api from '../utils/api';
import { User, Mail, Loader2 } from 'lucide-react';

const cardStyle = {
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  borderRadius: '0.75rem',
  padding: '1.5rem',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  transition: 'all 0.3s',
};

function UserProfile() {
  const [profile, setProfile] = useState(null);
  const { user } = useAuth();

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await api.get('/api/users/user/');
        setProfile(response.data);
      } catch (error) {
        console.error('Error fetching profile:', error);
      }
    };
    if (user) {
      fetchProfile();
    }
  }, [user]);

  if (!profile) return (
    <div className="flex justify-center items-center h-screen bg-gradient-to-br from-green-800 via-teal-800 to-blue-800">
      <Loader2 className="animate-spin text-white" size={48} />
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-teal-800 to-blue-800 text-white p-8 pt-24">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center">User Profile</h1>
        <div style={cardStyle}>
          <h2 className="text-2xl font-semibold mb-6 flex items-center">
            <User className="mr-2" />
            Profile Information
          </h2>
          <div className="space-y-4">
            <div className="flex items-center">
              <Mail className="mr-2" />
              <span className="font-semibold mr-2">Email:</span>
              <span>{profile.email}</span>
            </div>
            {/* Add more profile information as needed */}
          </div>
        </div>
      </div>
    </div>
  );
}

export default UserProfile;