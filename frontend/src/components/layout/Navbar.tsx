'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import {
    TrendingUp,
    Bell,
    User,
    Moon,
    Sun,
    LogOut,
    Settings,
    ChevronDown
} from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { isNSETradingDay, isNSEHoliday } from '@/lib/tradingDays';

// Check if market is currently open (during trading hours on a trading day)
function getMarketStatus(): { isOpen: boolean; message: string } {
    const now = new Date();
    const day = now.getDay(); // 0 = Sunday, 6 = Saturday
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const currentTime = hours * 60 + minutes; // Convert to minutes since midnight

    // NSE trading hours: 9:15 AM - 3:30 PM IST
    const marketOpen = 9 * 60 + 15;  // 9:15 AM = 555 minutes
    const marketClose = 15 * 60 + 30; // 3:30 PM = 930 minutes

    // Check if it's a weekend
    if (day === 0 || day === 6) {
        return {
            isOpen: false,
            message: day === 0 ? 'Closed (Sunday)' : 'Closed (Saturday)'
        };
    }

    // Check if it's a holiday
    if (isNSEHoliday(now)) {
        return {
            isOpen: false,
            message: 'Closed (Holiday)'
        };
    }

    // Check trading hours
    if (currentTime < marketOpen) {
        return {
            isOpen: false,
            message: 'Pre-Market'
        };
    }

    if (currentTime > marketClose) {
        return {
            isOpen: false,
            message: 'After Hours'
        };
    }

    // Market is open
    return {
        isOpen: true,
        message: 'Market Open'
    };
}

export function Navbar() {
    const [isDark, setIsDark] = useState(false);
    const [showUserMenu, setShowUserMenu] = useState(false);
    const [marketStatus, setMarketStatus] = useState({ isOpen: false, message: 'Loading...' });
    const { user, isAuthenticated, logout } = useAuthStore();

    // Update market status every minute
    useEffect(() => {
        const updateStatus = () => {
            setMarketStatus(getMarketStatus());
        };

        updateStatus(); // Initial update
        const interval = setInterval(updateStatus, 60000); // Update every minute

        return () => clearInterval(interval);
    }, []);

    // Initialize theme
    useEffect(() => {
        if (typeof window !== 'undefined') {
            const savedTheme = localStorage.getItem('theme');
            const isDarkMode = savedTheme === 'dark' ||
                (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
            setIsDark(isDarkMode);
            document.documentElement.classList.toggle('dark', isDarkMode);
        }
    }, []);

    const toggleTheme = () => {
        const newTheme = !isDark;
        setIsDark(newTheme);
        document.documentElement.classList.toggle('dark', newTheme);
        localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    };

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-white dark:bg-surface-800 border-b border-surface-200 dark:border-surface-700 px-6">
            <div className="flex items-center justify-between h-full">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                        <TrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-lg font-bold gradient-text">NIFTY 50 Index Predictor</h1>
                        <p className="text-xs text-surface-500">AI-Powered Predictions</p>
                    </div>
                </Link>

                {/* Right side */}
                <div className="flex items-center gap-4">
                    {/* Market Status */}
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${marketStatus.isOpen
                            ? 'bg-success-100 dark:bg-success-900/30'
                            : 'bg-surface-100 dark:bg-surface-700'
                        }`}>
                        <span className={`w-2 h-2 rounded-full ${marketStatus.isOpen
                                ? 'bg-success-500 animate-pulse'
                                : 'bg-surface-400'
                            }`} />
                        <span className={`text-sm font-medium ${marketStatus.isOpen
                                ? 'text-success-700 dark:text-success-400'
                                : 'text-surface-600 dark:text-surface-300'
                            }`}>
                            {marketStatus.message}
                        </span>
                        <InfoTooltip
                            title="Market Status"
                            content="NSE trading hours: 9:15 AM - 3:30 PM IST, Monday to Friday (excluding holidays)"
                        />
                    </div>

                    {/* Notifications */}
                    <button className="relative p-2 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors">
                        <Bell className="w-5 h-5 text-surface-600 dark:text-surface-300" />
                        <span className="absolute top-1 right-1 w-2 h-2 bg-danger-500 rounded-full" />
                    </button>

                    {/* Theme Toggle */}
                    <button
                        onClick={toggleTheme}
                        className="p-2 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                        aria-label="Toggle theme"
                    >
                        {isDark ? (
                            <Sun className="w-5 h-5 text-surface-600 dark:text-surface-300" />
                        ) : (
                            <Moon className="w-5 h-5 text-surface-600 dark:text-surface-300" />
                        )}
                    </button>

                    {/* User Menu */}
                    {isAuthenticated ? (
                        <div className="relative">
                            <button
                                onClick={() => setShowUserMenu(!showUserMenu)}
                                className="flex items-center gap-2 p-2 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                            >
                                {user?.avatar_url ? (
                                    <img
                                        src={user.avatar_url}
                                        alt="Profile"
                                        className="w-8 h-8 rounded-full object-cover"
                                    />
                                ) : (
                                    <div className="w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center text-white font-medium">
                                        {user?.full_name?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || 'U'}
                                    </div>
                                )}
                                <ChevronDown className="w-4 h-4 text-surface-500" />
                            </button>

                            {showUserMenu && (
                                <div className="absolute right-0 mt-2 w-48 py-2 bg-white dark:bg-surface-800 rounded-xl shadow-lg border border-surface-200 dark:border-surface-700">
                                    <div className="px-4 py-2 border-b border-surface-200 dark:border-surface-700">
                                        <p className="text-sm font-medium text-surface-900 dark:text-white truncate">
                                            {user?.full_name || 'User'}
                                        </p>
                                        <p className="text-xs text-surface-500 truncate">
                                            {user?.email}
                                        </p>
                                    </div>
                                    <Link
                                        href="/settings"
                                        className="flex items-center gap-2 px-4 py-2 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                                        onClick={() => setShowUserMenu(false)}
                                    >
                                        <Settings className="w-4 h-4" />
                                        Settings
                                    </Link>
                                    <button
                                        onClick={() => {
                                            logout();
                                            setShowUserMenu(false);
                                        }}
                                        className="flex items-center gap-2 px-4 py-2 w-full text-left text-danger-500 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                                    >
                                        <LogOut className="w-4 h-4" />
                                        Logout
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <Link
                            href="/auth/login"
                            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors font-medium"
                        >
                            Sign In
                        </Link>
                    )}
                </div>
            </div>
        </nav>
    );
}
