import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import api from '../utils/api';

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

  if (!profile) return <div>Loading...</div>;

  return (
    <div>
      <h2>User Profile</h2>
      <p>Email: {profile.email}</p>
      {/* Add more profile information as needed */}
    </div>
  );
}

export default UserProfile;