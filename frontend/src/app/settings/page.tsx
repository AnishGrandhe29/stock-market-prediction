'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import {
    Settings,
    User,
    Bell,
    Moon,
    Sun,
    Shield,
    Palette,
    Save,
    LogOut,
    CheckCircle,
    Mail,
    RefreshCw
} from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';
import { usersAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

export default function SettingsPage() {
    const { user, logout, isAuthenticated, accessToken } = useAuthStore();
    const [activeTab, setActiveTab] = useState('profile');
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [notifications, setNotifications] = useState({
        predictions: true,
        alerts: true,
        newsletter: false,
    });
    const [profile, setProfile] = useState({
        fullName: '',
        email: '',
        avatarUrl: '',
    });
    const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');

    // Fetch user data from API
    const fetchUserData = async () => {
        if (!accessToken) return;

        setIsRefreshing(true);
        try {
            const response = await usersAPI.getMe();
            const userData = response.data;
            setProfile({
                fullName: userData.full_name || '',
                email: userData.email || '',
                avatarUrl: userData.avatar_url || '',
            });
        } catch (error) {
            console.error('Failed to fetch user data:', error);
        } finally {
            setIsRefreshing(false);
        }
    };

    useEffect(() => {
        // Check system theme
        if (typeof window !== 'undefined') {
            setIsDarkMode(document.documentElement.classList.contains('dark'));
        }

        // Fetch fresh user data from API
        if (isAuthenticated) {
            fetchUserData();
        }
    }, [isAuthenticated, accessToken]);

    const toggleDarkMode = () => {
        const newMode = !isDarkMode;
        setIsDarkMode(newMode);
        if (typeof window !== 'undefined') {
            document.documentElement.classList.toggle('dark', newMode);
            localStorage.setItem('theme', newMode ? 'dark' : 'light');
        }
    };

    const handleSaveProfile = async () => {
        setSaveStatus('saving');
        try {
            await usersAPI.updateProfile({ full_name: profile.fullName });
            setSaveStatus('saved');
            setTimeout(() => setSaveStatus('idle'), 2000);
        } catch (error) {
            console.error('Failed to save profile:', error);
            setSaveStatus('idle');
        }
    };

    const handleLogout = () => {
        logout();
        window.location.href = '/auth/login';
    };

    const tabs = [
        { id: 'profile', label: 'Profile', icon: User },
        { id: 'appearance', label: 'Appearance', icon: Palette },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'security', label: 'Security', icon: Shield },
    ];

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3">
                <Settings className="w-8 h-8 text-primary-500" />
                <div>
                    <h1 className="text-3xl font-bold text-surface-900 dark:text-white">
                        Settings
                    </h1>
                    <p className="text-surface-500">
                        Manage your account preferences
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar Tabs */}
                <div className="lg:col-span-1">
                    <div className="card p-2 space-y-1">
                        {tabs.map((tab) => {
                            const Icon = tab.icon;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${activeTab === tab.id
                                            ? 'bg-primary-500 text-white'
                                            : 'text-surface-600 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-surface-700'
                                        }`}
                                >
                                    <Icon className="w-5 h-5" />
                                    <span className="font-medium">{tab.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Content */}
                <div className="lg:col-span-3">
                    <div className="card p-6">
                        {/* Profile Tab */}
                        {activeTab === 'profile' && (
                            <div className="space-y-6">
                                <div className="flex items-center justify-between">
                                    <h2 className="text-xl font-semibold text-surface-900 dark:text-white">
                                        Profile Settings
                                    </h2>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={fetchUserData}
                                            disabled={isRefreshing}
                                            className="p-2 text-surface-500 hover:text-primary-500 transition-colors"
                                            title="Refresh profile data"
                                        >
                                            <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
                                        </button>
                                        <InfoTooltip
                                            title="Profile"
                                            content="Your profile information from Google account."
                                        />
                                    </div>
                                </div>

                                {!isAuthenticated ? (
                                    <div className="text-center py-8">
                                        <User className="w-16 h-16 mx-auto text-surface-300 mb-4" />
                                        <p className="text-surface-500 mb-4">
                                            Please log in to manage your profile
                                        </p>
                                        <a
                                            href="/auth/login"
                                            className="inline-flex items-center gap-2 px-6 py-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
                                        >
                                            Sign In
                                        </a>
                                    </div>
                                ) : (
                                    <>
                                        {/* Profile Picture from Google */}
                                        <div className="flex items-center gap-6 p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                                            <div className="relative">
                                                {profile.avatarUrl ? (
                                                    <img
                                                        src={profile.avatarUrl}
                                                        alt="Profile"
                                                        className="w-20 h-20 rounded-full object-cover border-4 border-primary-500"
                                                    />
                                                ) : (
                                                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center border-4 border-primary-500">
                                                        <span className="text-2xl font-bold text-white">
                                                            {profile.fullName?.[0]?.toUpperCase() || profile.email?.[0]?.toUpperCase() || '?'}
                                                        </span>
                                                    </div>
                                                )}
                                                <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-success-500 rounded-full border-2 border-white dark:border-surface-800 flex items-center justify-center">
                                                    <svg className="w-3 h-3 text-white" viewBox="0 0 24 24">
                                                        <path fill="currentColor" d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                                                    </svg>
                                                </div>
                                            </div>
                                            <div>
                                                <p className="text-sm text-surface-500 mb-1">Signed in via Google</p>
                                                <p className="font-semibold text-surface-900 dark:text-white text-lg">
                                                    {profile.fullName || 'No name set'}
                                                </p>
                                                <p className="text-surface-500 text-sm flex items-center gap-1">
                                                    <Mail className="w-4 h-4" />
                                                    {profile.email}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="space-y-4">
                                            <div>
                                                <label className="block text-sm font-medium text-surface-700 dark:text-surface-300 mb-2">
                                                    Display Name
                                                </label>
                                                <input
                                                    type="text"
                                                    value={profile.fullName}
                                                    onChange={(e) => setProfile({ ...profile, fullName: e.target.value })}
                                                    className="w-full px-4 py-3 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700 text-surface-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                                    placeholder="Enter your display name"
                                                />
                                                <p className="text-xs text-surface-400 mt-1">
                                                    This is how your name appears in the app
                                                </p>
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-surface-700 dark:text-surface-300 mb-2">
                                                    Email Address
                                                </label>
                                                <div className="flex items-center gap-2 px-4 py-3 rounded-lg border border-surface-200 dark:border-surface-600 bg-surface-100 dark:bg-surface-800">
                                                    <Mail className="w-5 h-5 text-surface-400" />
                                                    <span className="text-surface-500">{profile.email}</span>
                                                    <span className="ml-auto text-xs bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 px-2 py-1 rounded">
                                                        Google
                                                    </span>
                                                </div>
                                                <p className="text-xs text-surface-400 mt-1">
                                                    Email is managed by your Google account
                                                </p>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-4 pt-4 border-t border-surface-200 dark:border-surface-700">
                                            <button
                                                onClick={handleSaveProfile}
                                                disabled={saveStatus === 'saving'}
                                                className="flex items-center gap-2 px-6 py-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 transition-colors"
                                            >
                                                {saveStatus === 'saved' ? (
                                                    <CheckCircle className="w-5 h-5" />
                                                ) : (
                                                    <Save className="w-5 h-5" />
                                                )}
                                                {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'Saved!' : 'Save Changes'}
                                            </button>
                                        </div>
                                    </>
                                )}
                            </div>
                        )}

                        {/* Appearance Tab */}
                        {activeTab === 'appearance' && (
                            <div className="space-y-6">
                                <h2 className="text-xl font-semibold text-surface-900 dark:text-white">
                                    Appearance
                                </h2>

                                <div className="flex items-center justify-between p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                                    <div className="flex items-center gap-4">
                                        {isDarkMode ? (
                                            <Moon className="w-6 h-6 text-primary-500" />
                                        ) : (
                                            <Sun className="w-6 h-6 text-warning-500" />
                                        )}
                                        <div>
                                            <p className="font-medium text-surface-900 dark:text-white">
                                                Dark Mode
                                            </p>
                                            <p className="text-sm text-surface-500">
                                                {isDarkMode ? 'Currently using dark theme' : 'Currently using light theme'}
                                            </p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={toggleDarkMode}
                                        className={`relative w-14 h-7 rounded-full transition-colors ${isDarkMode ? 'bg-primary-500' : 'bg-surface-300'
                                            }`}
                                    >
                                        <span
                                            className={`absolute top-1 w-5 h-5 bg-white rounded-full transition-transform ${isDarkMode ? 'translate-x-8' : 'translate-x-1'
                                                }`}
                                        />
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Notifications Tab */}
                        {activeTab === 'notifications' && (
                            <div className="space-y-6">
                                <h2 className="text-xl font-semibold text-surface-900 dark:text-white">
                                    Notification Preferences
                                </h2>

                                <div className="space-y-4">
                                    {[
                                        { key: 'predictions', label: 'Daily Predictions', desc: 'Get notified when new predictions are available' },
                                        { key: 'alerts', label: 'Price Alerts', desc: 'Receive alerts when your watchlist stocks hit targets' },
                                        { key: 'newsletter', label: 'Weekly Newsletter', desc: 'Weekly market insights and analysis' },
                                    ].map((item) => (
                                        <div
                                            key={item.key}
                                            className="flex items-center justify-between p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl"
                                        >
                                            <div>
                                                <p className="font-medium text-surface-900 dark:text-white">
                                                    {item.label}
                                                </p>
                                                <p className="text-sm text-surface-500">
                                                    {item.desc}
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => setNotifications({
                                                    ...notifications,
                                                    [item.key]: !notifications[item.key as keyof typeof notifications]
                                                })}
                                                className={`relative w-14 h-7 rounded-full transition-colors ${notifications[item.key as keyof typeof notifications]
                                                        ? 'bg-primary-500'
                                                        : 'bg-surface-300'
                                                    }`}
                                            >
                                                <span
                                                    className={`absolute top-1 w-5 h-5 bg-white rounded-full transition-transform ${notifications[item.key as keyof typeof notifications]
                                                            ? 'translate-x-8'
                                                            : 'translate-x-1'
                                                        }`}
                                                />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Security Tab */}
                        {activeTab === 'security' && (
                            <div className="space-y-6">
                                <h2 className="text-xl font-semibold text-surface-900 dark:text-white">
                                    Security
                                </h2>

                                {isAuthenticated ? (
                                    <div className="space-y-4">
                                        <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                                            <div className="flex items-center gap-3 mb-3">
                                                <svg className="w-6 h-6" viewBox="0 0 24 24">
                                                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                                                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                                                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                                                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                                                </svg>
                                                <span className="font-medium text-surface-900 dark:text-white">
                                                    Connected with Google
                                                </span>
                                                <span className="ml-auto text-xs bg-success-100 dark:bg-success-900/30 text-success-600 dark:text-success-400 px-2 py-1 rounded">
                                                    Active
                                                </span>
                                            </div>
                                            <p className="text-sm text-surface-500">
                                                Your account is securely connected via Google OAuth
                                            </p>
                                        </div>

                                        <div className="p-4 bg-danger-50 dark:bg-danger-900/20 rounded-xl border border-danger-200 dark:border-danger-800">
                                            <p className="font-medium text-danger-700 dark:text-danger-400 mb-1">
                                                Sign Out
                                            </p>
                                            <p className="text-sm text-danger-600 dark:text-danger-400/80 mb-4">
                                                Sign out of your account on this device
                                            </p>
                                            <button
                                                onClick={handleLogout}
                                                className="flex items-center gap-2 px-4 py-2 bg-danger-500 text-white rounded-lg hover:bg-danger-600 transition-colors"
                                            >
                                                <LogOut className="w-4 h-4" />
                                                Sign Out
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-center py-8">
                                        <Shield className="w-16 h-16 mx-auto text-surface-300 mb-4" />
                                        <p className="text-surface-500">
                                            Please log in to view security settings
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
