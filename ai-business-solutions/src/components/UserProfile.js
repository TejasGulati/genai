import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import api from '../utils/api';
import { User, Mail, Loader2, Calendar, MapPin } from 'lucide-react';

const cardStyle = {
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  borderRadius: '0.75rem',
  padding: '1.5rem',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  transition: 'all 0.3s',
  marginBottom: '1.5rem',
};

const ProfileItem = ({ icon: Icon, label, value }) => (
  <div style={cardStyle}>
    <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: 'white', marginBottom: '0.5rem', display: 'flex', alignItems: 'center' }}>
      <Icon style={{ marginRight: '0.5rem' }} size={20} />
      {label}
    </h4>
    <div style={{ color: '#D1D5DB' }}>{value}</div>
  </div>
);

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
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(to bottom right, #065F46, #0F766E, #1E40AF)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      <Loader2 style={{ animation: 'spin 1s linear infinite' }} size={48} />
    </div>
  );

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(to bottom right, #065F46, #0F766E, #1E40AF)',
      color: 'white',
      padding: '4rem 1rem'
    }}>
      <div style={{ maxWidth: '48rem', margin: '0 auto' }}>
        <h2 style={{ fontSize: 'clamp(2rem, 5vw, 4rem)', fontWeight: '800', marginBottom: '2rem', textAlign: 'center' }}>User Profile</h2>
        <div style={{ ...cardStyle, marginBottom: '2rem' }}>
          <h3 style={{ fontSize: '1.5rem', fontWeight: '700', color: 'white', marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
            <User style={{ marginRight: '0.5rem' }} size={24} />
            Profile Information
          </h3>
          <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))' }}>
            <ProfileItem icon={Mail} label="Email" value={profile.email} />
            <ProfileItem icon={User} label="Username" value={profile.username || 'Not set'} />
            <ProfileItem icon={Calendar} label="Joined" value={new Date(profile.date_joined).toLocaleDateString()} />
            <ProfileItem icon={MapPin} label="Location" value={profile.location || 'Not set'} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default UserProfile;